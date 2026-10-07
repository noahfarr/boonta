import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from boonta.utils import Array, Key, Timestep

from .wrapper import Wrapper, WrapperState


@struct.dataclass
class PromptState(WrapperState):
    cursor: Array


class Prompt(Wrapper):
    def __init__(self, env, prompt: Array, pad: int):
        super().__init__(env)
        self._length, *_ = env.observation_space().shape
        prompt = np.asarray(prompt, np.int32)
        self._prompt_length = len(prompt)
        self._chunks = max(int(np.ceil(len(prompt) / self._length)), 1)
        self._prompt = jnp.asarray(
            np.pad(
                prompt,
                (0, self._chunks * self._length - len(prompt)),
                constant_values=pad,
            )
        )

    def init(self, key: Key) -> tuple[PromptState, Timestep]:
        env_state, timestep = self._env.init(key)
        state = PromptState(env_state=env_state, cursor=jnp.int32(self._length))
        return state, timestep.replace(obs=self._prompt[: self._length])

    def step(
        self,
        key: Key,
        state: PromptState,
        action: Array,
    ) -> tuple[PromptState, Timestep]:
        reading = state.cursor < self._prompt_length
        chunk = jax.lax.dynamic_slice_in_dim(self._prompt, state.cursor, self._length)

        env_state, timestep = self._env.step(key, state.env_state, action)

        env_state = jax.tree.map(
            lambda frozen, played: jnp.where(reading, frozen, played),
            state.env_state,
            env_state,
        )
        cursor = state.cursor + jnp.where(reading, self._length, 0)
        timestep = timestep.replace(
            obs=jnp.where(reading, chunk, timestep.obs),
            reward=jnp.where(reading, jnp.zeros_like(timestep.reward), timestep.reward),
            terminated=jnp.where(
                reading, jnp.zeros_like(timestep.terminated), timestep.terminated
            ),
            info=jax.tree.map(
                lambda leaf: jnp.where(reading, jnp.zeros_like(leaf), leaf),
                timestep.info,
            ),
        )
        return PromptState(env_state=env_state, cursor=cursor), timestep

    def time_limit(self) -> int:
        return int(self._env.time_limit()) + self._chunks
