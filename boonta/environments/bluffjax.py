from typing import Any

import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .environment import Environment
from .spaces import Space


class BluffJAX(Environment):
    def __init__(self, env):
        from bluffjax.environments.env import AECEnv

        self._env = env
        self.turn_based = isinstance(env, AECEnv)

    @property
    def num_agents(self) -> int:
        return self._env.num_agents

    def action_mask(self, state: Any) -> Array:
        avail = self._env.get_avail_actions(state).astype(bool)
        if self.turn_based:
            seats = jnp.arange(self.num_agents) == state.current_player_idx
            waiting = jnp.arange(avail.shape[-1]) == 0
            avail = jnp.where(seats[:, None], avail, waiting)
        stuck = ~avail.any(axis=-1, keepdims=True)
        return avail | (stuck & (jnp.arange(avail.shape[-1]) == 0))

    def observe_seats(self, state: Any, obs: Array) -> dict[str, Array]:
        if self.turn_based:
            seats = jnp.arange(self.num_agents) == state.current_player_idx
            obs = jnp.where(seats[:, None], obs, 0.0)
        return {"observation": obs.astype(jnp.float32), "mask": self.action_mask(state)}

    def observe(self, state: Any) -> dict[str, Array]:
        return self.observe_seats(state, self._env.obs_from_state(state))

    def init(self, key: Key) -> tuple[Any, Timestep]:
        state, obs = self._env.reset(key)
        seats = (self.num_agents,)
        return state, Timestep(
            obs=self.observe_seats(state, obs),
            action=jnp.zeros(seats, jnp.int32),
            reward=jnp.zeros(seats, jnp.float32),
            terminated=jnp.ones(seats, bool),
            truncated=jnp.zeros(seats, bool),
        )

    def step(self, key: Key, state: Any, action: Array) -> tuple[Any, Timestep]:
        move = jnp.take(action, state.current_player_idx) if self.turn_based else action
        state, obs, reward, absorbing, done, _ = self._env.step_env(key, state, move)
        seats = (self.num_agents,)
        ended = absorbing.all()
        return state, Timestep(
            obs=self.observe_seats(state, obs),
            action=action,
            reward=reward.astype(jnp.float32),
            terminated=jnp.broadcast_to(done & ended, seats),
            truncated=jnp.broadcast_to(done & ~ended, seats),
        )

    def observation_space(self) -> dict[str, Space]:
        _, obs = jax.eval_shape(self._env.reset, jax.random.key(0))
        width = obs.shape[-1]
        num_actions = self._env.action_space().n
        return {
            "observation": Space((self.num_agents, width), jnp.float32, -jnp.inf, jnp.inf),
            "mask": Space((self.num_agents, num_actions), jnp.bool_, 0, 1),
        }

    def action_space(self) -> Space:
        num_actions = self._env.action_space().n
        return Space((self.num_agents,), jnp.int32, 0, num_actions - 1)

    def time_limit(self) -> int:
        return int(self._env.horizon)


def make(env_id, **kwargs):
    import bluffjax

    return BluffJAX(bluffjax.make(env_id, **kwargs))
