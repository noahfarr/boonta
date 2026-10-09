import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, PyTree, Timestep, broadcast

from .wrapper import Wrapper, WrapperState


@struct.dataclass
class UEDState(WrapperState):
    theta: PyTree = struct.field(metadata={"axis": None})
    weights: Array = struct.field(metadata={"axis": None})
    restarting: Array = struct.field(metadata={"axis": None})
    assigned: Array = struct.field(metadata={"axis": None})
    upcoming: Array
    playing: Array


def tag_theta(timestep: Timestep, playing: Array) -> Timestep:
    return timestep.replace(info={**(timestep.info or {}), "theta": playing})


class UED(Wrapper):
    def init(self, key: Key) -> tuple[UEDState, Timestep]:
        env_state, timestep = self._env.init(key)
        playing = jnp.full(self.num_envs, -1, jnp.int32)
        state = UEDState(
            env_state,
            theta=None,
            weights=None,
            restarting=jnp.bool_(False),
            assigned=jnp.bool_(False),
            upcoming=playing,
            playing=playing,
        )
        return state, tag_theta(timestep, playing)

    def start_theta(self, state: UEDState, env_state, index: Array, chosen: Array):
        theta = jax.tree.map(lambda leaf: leaf[jnp.maximum(index, 0)], state.theta)
        started = jax.vmap(lambda inner, level: self._env.update(inner, theta=level))(
            env_state, theta
        )
        obs = self._env.observe(started)
        chosen = chosen & (index >= 0)

        def select(fresh, leaf):
            return jnp.where(broadcast(chosen, leaf), fresh, leaf)

        return jax.tree.map(select, started, env_state), chosen, obs

    def replay_finished(self, state: UEDState, env_state, timestep: Timestep):
        done = timestep.done.reshape(self.num_envs, -1).all(axis=-1)
        env_state, chosen, obs = self.start_theta(state, env_state, state.playing, done)
        obs = jax.tree.map(
            lambda fresh, leaf: jnp.where(broadcast(chosen, leaf), fresh, leaf),
            obs,
            timestep.obs,
        )
        return env_state, timestep.replace(obs=obs)

    def restart_theta(self, key: Key, state: UEDState, env_state, timestep: Timestep):
        drawn = jnp.full(self.num_envs, -1, jnp.int32)
        if state.weights is not None:
            weighted = state.weights.sum() > 0
            sampled = jax.random.choice(
                key,
                state.weights.shape[0],
                (self.num_envs,),
                p=jnp.where(weighted, state.weights, 1.0),
            ).astype(jnp.int32)
            drawn = jnp.where(weighted, sampled, drawn)
        index = jnp.where(state.assigned, state.upcoming, drawn)
        env_state, chosen, obs = self.start_theta(
            state, env_state, index, jnp.ones(self.num_envs, bool)
        )
        obs = jax.tree.map(
            lambda fresh, leaf: jnp.where(broadcast(chosen, leaf), fresh, leaf),
            obs,
            timestep.obs,
        )
        truncated = timestep.truncated | (
            broadcast(chosen, timestep.truncated) & ~timestep.terminated
        )
        timestep = timestep.replace(obs=obs, truncated=truncated)
        return env_state, timestep, jnp.where(chosen, index, state.playing)

    def step(self, key: Key, state: UEDState, action: Array) -> tuple[UEDState, Timestep]:
        env_state, timestep = self._env.step(key, state.env_state, action)
        if state.theta is not None:
            env_state, timestep = jax.lax.cond(
                jnp.any(state.playing >= 0),
                lambda: self.replay_finished(state, env_state, timestep),
                lambda: (env_state, timestep),
            )
            env_state, timestep, playing = jax.lax.cond(
                state.restarting,
                lambda: self.restart_theta(
                    jax.random.fold_in(key, 1), state, env_state, timestep
                ),
                lambda: (env_state, timestep, state.playing),
            )
        else:
            playing = state.playing
        timestep = tag_theta(timestep, state.playing)
        state = state.replace(
            env_state=env_state,
            playing=playing,
            restarting=jnp.bool_(False),
            assigned=jnp.bool_(False),
        )
        return state, timestep

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
        if state.theta is not None and state.weights is not None:
            (capacity,) = state.weights.shape
            assert all(
                leaf.shape[0] == capacity for leaf in jax.tree.leaves(state.theta)
            ), "theta and weights must hold one entry per slot"
        return state
