import atexit
import threading
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.experimental import io_callback

from boonta.utils import Array, Key, Timestep, canonicalize_dtype

from .environment import Environment
from .spaces import Space
from .wrappers import Batched


@struct.dataclass
class GymnasiumState:
    handle: Array = struct.field(metadata={"axis": None})


def convert(space) -> Space:
    from gymnasium import spaces

    if isinstance(space, spaces.Box):
        return Space(shape=space.shape, dtype=space.dtype, low=space.low, high=space.high)
    if isinstance(space, spaces.Discrete):
        start = int(space.start)
        return Space(shape=(), dtype=jnp.int32, low=start, high=start + int(space.n) - 1)
    if isinstance(space, spaces.MultiDiscrete):
        return Space(
            shape=space.shape,
            dtype=jnp.int32,
            low=space.start,
            high=space.start + space.nvec - 1,
        )
    raise ValueError(
        f"boonta has no space for {space}; expose a Box, Discrete or MultiDiscrete space"
    )


class Gymnasium(Environment):
    def __init__(self, factory: Callable[[], Any]):
        from gymnasium.vector import AutoresetMode

        self.factory = factory
        probe = factory()
        mode = probe.metadata.get("autoreset_mode")
        assert mode == AutoresetMode.SAME_STEP, (
            f"the vector environment resets in {mode} mode; boonta needs "
            f"autoreset_mode=AutoresetMode.SAME_STEP, so the terminal step reports "
            f"the episode's end with the next episode's first observation"
        )
        self.num_envs = probe.num_envs
        self._observation_space = convert(probe.single_observation_space)
        self._action_space = convert(probe.single_action_space)
        self.lock = threading.Lock()
        self.free = [probe]
        self.busy = {}
        self.opened = 0
        atexit.register(self.shutdown)

    def observation_space(self) -> Space:
        return self._observation_space

    def action_space(self) -> Space:
        return self._action_space

    def batched(self, space: Space, dtype=None) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.num_envs, *space.shape), dtype or space.dtype)

    def observe(self, obs) -> np.ndarray:
        return np.asarray(obs, canonicalize_dtype(self._observation_space.dtype))

    def open(self, seed):
        with self.lock:
            environment = self.free.pop() if self.free else self.factory()
            handle = self.opened
            self.opened += 1
            self.busy[handle] = environment
        obs, _ = environment.reset(seed=int(seed))
        return np.int32(handle), self.observe(obs)

    def advance(self, handle, action):
        with self.lock:
            environment = self.busy[int(handle)]
        obs, reward, terminated, truncated, _ = environment.step(np.asarray(action))
        return (
            np.int32(handle),
            self.observe(obs),
            np.asarray(reward, np.float32),
            np.asarray(terminated, np.bool_),
            np.asarray(truncated, np.bool_),
        )

    def release(self, handle):
        with self.lock:
            environment = self.busy.pop(int(handle), None)
            if environment is not None:
                self.free.append(environment)
        return np.int32(0)

    def shutdown(self):
        with self.lock:
            for environment in [*self.free, *self.busy.values()]:
                environment.close()
            self.free, self.busy = [], {}

    def init(self, key: Key) -> tuple[GymnasiumState, Timestep]:
        seed = jax.random.randint(key, (), 0, jnp.iinfo(jnp.int32).max)
        handle, obs = io_callback(
            self.open,
            (jax.ShapeDtypeStruct((), jnp.int32), self.batched(self._observation_space)),
            seed,
        )
        flags = jnp.zeros((self.num_envs,), bool)
        return GymnasiumState(handle), Timestep(
            obs=obs,
            action=jnp.zeros(
                (self.num_envs, *self._action_space.shape), self._action_space.dtype
            ),
            reward=jnp.zeros((self.num_envs,), jnp.float32),
            terminated=~flags,
            truncated=flags,
            info={},
        )

    def step(
        self, key: Key, state: GymnasiumState, action: Array
    ) -> tuple[GymnasiumState, Timestep]:
        flags = jax.ShapeDtypeStruct((self.num_envs,), jnp.bool_)
        handle, obs, reward, terminated, truncated = io_callback(
            self.advance,
            (
                jax.ShapeDtypeStruct((), jnp.int32),
                self.batched(self._observation_space),
                jax.ShapeDtypeStruct((self.num_envs,), jnp.float32),
                flags,
                flags,
            ),
            state.handle,
            action,
        )
        return GymnasiumState(handle), Timestep(
            obs=obs,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={},
        )

    def close(self, state: GymnasiumState) -> None:
        io_callback(self.release, jax.ShapeDtypeStruct((), jnp.int32), state.handle)


def make(
    env_id: str,
    num_envs: int = 1,
    vectorization_mode: str = "async",
    **kwargs,
) -> Batched:
    import gymnasium
    from gymnasium.vector import AutoresetMode

    vector_kwargs = {"autoreset_mode": AutoresetMode.SAME_STEP}
    if vectorization_mode == "async":
        vector_kwargs |= {"context": "forkserver"}

    def factory():
        return gymnasium.make_vec(
            env_id,
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
            vector_kwargs=vector_kwargs,
            **kwargs,
        )

    return Batched(Gymnasium(factory), num_envs=num_envs)
