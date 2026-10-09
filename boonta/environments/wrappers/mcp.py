from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, Timestep

from ..spaces import Space
from .wrapper import Wrapper, WrapperState


@struct.dataclass
class MCPState(WrapperState):
    active: Array
    arguments: Array
    cursor: Array


class MCP(Wrapper):
    def __init__(
        self,
        env,
        *,
        to_action: Callable,
        to_tokens: Callable,
        start: int,
        end: int,
        pad: int,
        vocab_size: int,
        capacity: int,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...] = (),
        tokens_per_call: int = 32,
    ):
        super().__init__(env)
        self._to_action = to_action
        self._to_tokens = to_tokens
        self._start = start
        self._end = end
        self._pad = pad
        self._vocab_size = vocab_size
        self._capacity = capacity
        self._observation_shape = observation_shape
        self._action_shape = action_shape
        self._tokens_per_call = tokens_per_call

    def init(self, key: Key) -> tuple[MCPState, Timestep]:
        env_state, _ = self._env.init(key)
        state = MCPState(
            env_state=env_state,
            active=jnp.bool_(False),
            arguments=jnp.zeros(self._capacity, jnp.int32),
            cursor=jnp.int32(0),
        )
        action = jnp.int32(0)
        _, shapes = jax.eval_shape(self.step, key, state, action)
        timestep = Timestep(
            obs=jnp.full(self._observation_shape, self._pad, jnp.int32),
            action=action,
            reward=jnp.zeros(shapes.reward.shape, shapes.reward.dtype),
            terminated=jnp.ones(shapes.terminated.shape, shapes.terminated.dtype),
            truncated=jnp.zeros(shapes.truncated.shape, shapes.truncated.dtype),
            info=jax.tree.map(
                lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shapes.info
            ),
        )
        return state, timestep

    def step(
        self,
        key: Key,
        state: MCPState,
        action: Array,
    ) -> tuple[MCPState, Timestep]:
        token = jnp.asarray(action, jnp.int32)
        is_start = token == self._start
        is_end = token == self._end

        active = jnp.where(is_start, True, state.active)
        cursor = jnp.where(is_start, jnp.int32(0), state.cursor)
        arguments = jnp.where(
            is_start, jnp.zeros_like(state.arguments), state.arguments
        )

        collect = active & ~is_start & ~is_end & (cursor < self._capacity)
        index = jnp.clip(cursor, 0, self._capacity - 1)
        arguments = jnp.where(collect, arguments.at[index].set(token), arguments)
        cursor = jnp.where(collect, cursor + 1, cursor)

        fire = active & is_end

        def select(on_fire, otherwise):
            return jax.lax.select(fire, on_fire, otherwise)

        env_action = self._to_action(arguments, cursor)
        stepped_state, stepped = self._env.step(key, state.env_state, env_action)

        env_state = jax.tree.map(select, stepped_state, state.env_state)
        tokens = jnp.asarray(self._to_tokens(stepped.obs), jnp.int32)
        obs = select(tokens, jnp.full(self._observation_shape, self._pad, jnp.int32))
        reward = select(stepped.reward, jnp.zeros_like(stepped.reward))
        terminated = select(stepped.terminated, jnp.zeros_like(stepped.terminated))
        truncated = select(stepped.truncated, jnp.zeros_like(stepped.truncated))
        info = jax.tree.map(
            lambda leaf: select(leaf, jnp.zeros_like(leaf)), stepped.info
        )
        info = {**info, "fired": fire.astype(jnp.float32)}

        state = MCPState(
            env_state=env_state,
            active=jnp.where(fire, jnp.bool_(False), active),
            arguments=arguments,
            cursor=jnp.where(fire, jnp.int32(0), cursor),
        )
        timestep = Timestep(
            obs=obs,
            action=token,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        return state, timestep

    def observation_space(self) -> Space:
        return Space(
            shape=self._observation_shape,
            dtype=jnp.int32,
            low=0,
            high=self._vocab_size - 1,
        )

    def action_space(self) -> Space:
        return Space(
            shape=self._action_shape, dtype=jnp.int32, low=0, high=self._vocab_size - 1
        )

    def action_mask(self, state: MCPState) -> Array:
        return jnp.ones((*self._action_shape, self._vocab_size), dtype=bool)

    def observe(self, state: MCPState) -> Array:
        raise NotImplementedError

    def time_limit(self) -> int:
        return int(self._env.time_limit()) * self._tokens_per_call
