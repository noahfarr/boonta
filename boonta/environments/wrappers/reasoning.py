from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from boonta.utils import Array, Key, PyTree, Timestep

from ..spaces import Space
from .wrapper import Wrapper, WrapperState


@struct.dataclass
class ReasoningState(WrapperState):
    obs: PyTree


class Reasoning(Wrapper):
    def __init__(self, env, num_tokens: int, cost: float = 0.0):
        super().__init__(env)
        space = env.action_space()
        assert jnp.issubdtype(
            space.dtype, jnp.integer
        ), f"Reasoning requires a discrete action space, got dtype {space.dtype}"
        assert num_tokens >= 1, f"num_tokens ({num_tokens}) must be >= 1"
        self._num_tokens = num_tokens
        self._cost = cost

    def action_space(self) -> Space:
        space = self._env.action_space()
        return Space(
            shape=space.shape,
            dtype=space.dtype,
            low=space.low,
            high=space.high + self._num_tokens,
        )

    def action_mask(self, state: ReasoningState) -> Array | None:
        mask = self._env.action_mask(state.env_state)
        if mask is None:
            return None
        tokens = jnp.ones((*mask.shape[:-1], self._num_tokens), mask.dtype)
        return jnp.concatenate([mask, tokens], axis=-1)

    def init(self, key: Key) -> tuple[ReasoningState, Timestep]:
        env_state, timestep = self._env.init(key)
        return ReasoningState(env_state=env_state, obs=timestep.obs), timestep

    def step(
        self, key: Key, state: ReasoningState, action: Array
    ) -> tuple[ReasoningState, Timestep]:
        space = self._env.action_space()
        high = jnp.asarray(space.high, space.dtype)
        reasoning = jnp.any(action > high)

        env_state, timestep = self._env.step(
            key, state.env_state, jnp.minimum(action, high)
        )

        env_state = jax.tree.map(
            partial(jnp.where, reasoning), state.env_state, env_state
        )
        obs = jax.tree.map(partial(jnp.where, reasoning), state.obs, timestep.obs)
        timestep = timestep.replace(
            obs=obs,
            action=action,
            reward=jnp.where(reasoning, -self._cost, timestep.reward),
            terminated=jnp.where(reasoning, False, timestep.terminated),
            truncated=jnp.where(reasoning, False, timestep.truncated),
            info=jax.tree.map(
                lambda leaf: jnp.where(reasoning, 0, leaf), timestep.info
            ),
        )
        return ReasoningState(env_state=env_state, obs=obs), timestep
