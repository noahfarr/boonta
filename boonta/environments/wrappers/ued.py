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

    def start_theta(
        self, key: Key, state: UEDState, env_state, index: Array, chosen: Array
    ):
        theta = jax.tree.map(lambda leaf: leaf[jnp.maximum(index, 0)], state.theta)
        keys = jax.random.split(key, self.num_envs)
        started = jax.vmap(
            lambda inner, key, level: self._env.update(inner, key, theta=level)
        )(env_state, keys, theta)
        obs = self._env.observe(started)
        chosen = chosen & (index >= 0)

        def select(fresh, leaf):
            return jnp.where(broadcast(chosen, leaf), fresh, leaf)

        return jax.tree.map(select, started, env_state), chosen, obs

    def draw_theta(self, key: Key, state: UEDState) -> Array:
        weighted = state.weights.sum() > 0
        num_slots, *_ = state.weights.shape
        sampled = jax.random.choice(
            key,
            num_slots,
            (self.num_envs,),
            p=jnp.where(weighted, state.weights, 1.0),
        ).astype(jnp.int32)
        return jnp.where(weighted, sampled, -1)

    def replay_finished(self, key: Key, state: UEDState, env_state, timestep: Timestep):
        done = timestep.done.reshape(self.num_envs, -1).all(axis=-1)
        draw_key, start_key = jax.random.split(key)
        index = state.playing
        if state.weights is not None:
            drawn = self.draw_theta(draw_key, state)
            index = jnp.where((drawn >= 0) & (state.playing >= 0), drawn, state.playing)
        env_state, chosen, obs = self.start_theta(
            start_key, state, env_state, index, done
        )
        obs = jax.tree.map(
            lambda fresh, leaf: jnp.where(broadcast(chosen, leaf), fresh, leaf),
            obs,
            timestep.obs,
        )
        playing = jnp.where(chosen, index, state.playing)
        return env_state, timestep.replace(obs=obs), playing

    def restart_theta(
        self, key: Key, state: UEDState, env_state, timestep: Timestep, playing: Array
    ):
        drawn = jnp.full(self.num_envs, -1, jnp.int32)
        if state.weights is not None:
            drawn = self.draw_theta(key, state)
        index = jnp.where(state.assigned, state.upcoming, drawn)
        env_state, chosen, obs = self.start_theta(
            jax.random.fold_in(key, 1),
            state,
            env_state,
            index,
            jnp.ones(self.num_envs, bool),
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
        return env_state, timestep, jnp.where(chosen, index, playing)

    def step(self, key: Key, state: UEDState, action: Array) -> tuple[UEDState, Timestep]:
        env_state, timestep = self._env.step(key, state.env_state, action)
        if state.theta is not None:
            env_state, timestep, playing = jax.lax.cond(
                jnp.any(state.playing >= 0),
                lambda: self.replay_finished(
                    jax.random.fold_in(key, 2), state, env_state, timestep
                ),
                lambda: (env_state, timestep, state.playing),
            )
            env_state, timestep, playing = jax.lax.cond(
                state.restarting,
                lambda: self.restart_theta(
                    jax.random.fold_in(key, 1), state, env_state, timestep, playing
                ),
                lambda: (env_state, timestep, playing),
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
        key: Key,
        theta: PyTree = None,
        weights: Array = None,
        restart: bool = False,
        assign: Array = None,
        **kwargs,
    ) -> UEDState:
        if kwargs:
            state = state.replace(
                env_state=self._env.update(state.env_state, key, **kwargs)
            )
        if theta is not None:
            state = state.replace(theta=theta)
        if weights is not None:
            state = state.replace(weights=jnp.asarray(weights, jnp.float32))
        if restart is not False:
            restart = jnp.asarray(restart, bool)
            state = state.replace(
                restarting=state.restarting | restart, assigned=state.assigned & ~restart
            )
        if assign is not None:
            state = state.replace(
                restarting=jnp.bool_(True),
                assigned=jnp.bool_(True),
                upcoming=jnp.asarray(assign, jnp.int32),
            )
        if state.theta is not None and state.weights is not None:
            num_slots, *_ = state.weights.shape
            for leaf in jax.tree.leaves(state.theta):
                leaf_slots, *_ = leaf.shape
                assert (
                    leaf_slots == num_slots
                ), "theta and weights must hold one entry per slot"
        return state
