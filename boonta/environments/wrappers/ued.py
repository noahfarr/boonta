from typing import Protocol

import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, PyTree, Timestep, broadcast

from .wrapper import Wrapper, WrapperState


class Underspecified(Protocol):
    def sample(self, key: Key) -> PyTree: ...

    def reset(self, key: Key, theta: PyTree) -> tuple[PyTree, Timestep]: ...


@struct.dataclass
class UEDState(WrapperState):
    theta: PyTree = struct.field(metadata={"axis": None})
    weights: Array = struct.field(metadata={"axis": None})
    restarting: Array = struct.field(metadata={"axis": None})
    assigned: Array = struct.field(metadata={"axis": None})
    upcoming: Array
    playing: Array


def tag(timestep: Timestep, playing: Array) -> Timestep:
    return timestep.replace(info={**(timestep.info or {}), "theta": playing})


class UED(Wrapper):
    def __init__(self, env, capacity: int = 1):
        super().__init__(env)
        self.capacity = capacity

    def init(self, key: Key) -> tuple[UEDState, Timestep]:
        env_state, timestep = self._env.init(key)
        template = jax.eval_shape(self._env.sample, key)
        playing = jnp.full(self.num_envs, -1, jnp.int32)
        state = UEDState(
            env_state,
            theta=jax.tree.map(
                lambda leaf: jnp.zeros((self.capacity, *leaf.shape), leaf.dtype),
                template,
            ),
            weights=jnp.zeros(self.capacity, jnp.float32),
            restarting=jnp.bool_(False),
            assigned=jnp.bool_(False),
            upcoming=playing,
            playing=playing,
        )
        return state, tag(timestep, playing)

    def restart(self, key: Key, state: UEDState, env_state, timestep: Timestep):
        draw_key, sample_key, reset_key = jax.random.split(key, 3)
        weighted = state.weights.sum() > 0
        drawn = jax.random.choice(
            draw_key,
            state.weights.shape[0],
            (self.num_envs,),
            p=jnp.where(weighted, state.weights, 1.0),
        ).astype(jnp.int32)
        index = jnp.where(
            state.assigned, state.upcoming, jnp.where(weighted, drawn, -1)
        )
        sampled = jax.vmap(self._env.sample)(
            jax.random.split(sample_key, self.num_envs)
        )

        def choose(stored, fresh):
            chosen = stored[jnp.maximum(index, 0)]
            kept = (index >= 0).reshape(-1, *([1] * (fresh.ndim - 1)))
            return jnp.where(kept, chosen, fresh)

        theta = jax.tree.map(choose, state.theta, sampled)
        env_state, initial = jax.vmap(self._env.reset)(
            jax.random.split(reset_key, self.num_envs), theta
        )
        timestep = timestep.replace(
            obs=initial.obs, truncated=timestep.truncated | ~timestep.terminated
        )
        return env_state, timestep, index

    def replay(self, key: Key, state: UEDState, env_state, timestep: Timestep):
        done = timestep.done.reshape(self.num_envs, -1).all(axis=-1)
        replayed = done & (state.playing >= 0)
        theta = jax.tree.map(
            lambda leaf: leaf[jnp.maximum(state.playing, 0)], state.theta
        )
        started, initial = jax.vmap(self._env.reset)(
            jax.random.split(key, self.num_envs), theta
        )

        def select(fresh, leaf):
            return jnp.where(broadcast(replayed, leaf), fresh, leaf)

        env_state = jax.tree.map(select, started, env_state)
        obs = jax.tree.map(select, initial.obs, timestep.obs)
        return env_state, timestep.replace(obs=obs)

    def step(self, key: Key, state: UEDState, action: Array) -> tuple[UEDState, Timestep]:
        step_key, replay_key, restart_key = jax.random.split(key, 3)
        env_state, timestep = self._env.step(step_key, state.env_state, action)
        env_state, timestep = jax.lax.cond(
            jnp.any(state.playing >= 0),
            lambda: self.replay(replay_key, state, env_state, timestep),
            lambda: (env_state, timestep),
        )
        env_state, restarted, playing = jax.lax.cond(
            state.restarting,
            lambda: self.restart(restart_key, state, env_state, timestep),
            lambda: (env_state, timestep, state.playing),
        )
        restarted = tag(restarted, state.playing)
        state = state.replace(
            env_state=env_state,
            playing=playing,
            restarting=jnp.bool_(False),
            assigned=jnp.bool_(False),
        )
        return state, restarted

    def update(
        self,
        state: UEDState,
        theta: PyTree = None,
        weights: Array = None,
        restart: bool = False,
        assign: Array = None,
        **kwargs,
    ) -> UEDState:
        if kwargs:
            state = state.replace(env_state=self._env.update(state.env_state, **kwargs))
        if theta is not None:
            state = state.replace(theta=theta)
        if weights is not None:
            state = state.replace(weights=jnp.asarray(weights, jnp.float32))
        if restart:
            state = state.replace(restarting=jnp.bool_(True), assigned=jnp.bool_(False))
        if assign is not None:
            state = state.replace(
                restarting=jnp.bool_(True),
                assigned=jnp.bool_(True),
                upcoming=jnp.asarray(assign, jnp.int32),
            )
        (capacity,) = state.weights.shape
        assert all(
            leaf.shape[0] == capacity for leaf in jax.tree.leaves(state.theta)
        ), "theta and weights must hold one entry per slot"
        return state
