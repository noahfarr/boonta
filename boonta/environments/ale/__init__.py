import os
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, Timestep

from .. import build, load, register_targets
from ..environment import Environment
from ..spaces import Space

RAM_SIZE = 128
DIRECTORY = Path(os.path.dirname(__file__)) / "ffi"

registered = False


def register():
    global registered
    if registered:
        return
    build(DIRECTORY, "libale_vec.so", ["vectorize.cpp", "ffi.cc"])
    lib = load(DIRECTORY, "libale_vec.so")
    register_targets(
        lib,
        {
            "ale_init": "ale_ffi_init",
            "ale_step": "ale_ffi_step",
            "ale_close": "ale_ffi_close",
        },
    )
    registered = True


def rom_path(game: str) -> str:
    import ale_py.roms as roms

    return os.path.join(os.path.dirname(roms.__file__), f"{game}.bin")


@struct.dataclass
class ALEState:
    handle: Array


class ALE(Environment):
    def __init__(
        self,
        game: str = "montezuma_revenge",
        num_envs: int = 128,
        frame_skip: int = 4,
        num_threads: int = 28,
        fraction: float = 0.0,
        capacity: int = 512,
        depth: int = 8,
        warmup_episodes: int = 64,
        novelty: float = 0.0,
        snapshot_every: int = 0,
        snapshot: str = "",
    ):
        register()
        self.num_envs = num_envs
        self.num_actions = 18
        self._attrs = dict(
            num_envs=num_envs,
            rom=rom_path(game),
            frame_skip=frame_skip,
            threads=num_threads,
            fraction_ppm=int(fraction * 1_000_000),
            capacity=capacity,
            depth=depth,
            warmup_episodes=warmup_episodes,
            novelty_ppm=int(novelty * 1_000_000),
            snapshot_every=snapshot_every,
            snapshot=snapshot,
        )

    def init(self, key: Key) -> tuple[ALEState, Timestep]:
        handle, obs = jax.ffi.ffi_call(
            "ale_init",
            (
                jax.ShapeDtypeStruct((1,), jnp.int32),
                jax.ShapeDtypeStruct((self.num_envs, RAM_SIZE), jnp.uint8),
            ),
            has_side_effect=True,
        )(jax.random.randint(key, (), 0, 2**30, jnp.int32), **self._attrs)
        action = jnp.zeros((self.num_envs,), jnp.int32)
        return ALEState(handle=handle), Timestep(
            obs=obs,
            action=action,
            reward=jnp.zeros(self.num_envs, jnp.float32),
            terminated=jnp.ones(self.num_envs, bool),
            truncated=jnp.zeros(self.num_envs, bool),
            info={"warm": jnp.zeros(self.num_envs, bool)},
        )

    def step(
        self, key: Key, state: ALEState, action: Array
    ) -> tuple[ALEState, Timestep]:
        handle, obs, reward, done, warm = jax.ffi.ffi_call(
            "ale_step",
            (
                jax.ShapeDtypeStruct((1,), jnp.int32),
                jax.ShapeDtypeStruct((self.num_envs, RAM_SIZE), jnp.uint8),
                jax.ShapeDtypeStruct((self.num_envs,), jnp.float32),
                jax.ShapeDtypeStruct((self.num_envs,), jnp.uint8),
                jax.ShapeDtypeStruct((self.num_envs,), jnp.uint8),
            ),
            has_side_effect=True,
        )(state.handle, jnp.asarray(action, jnp.int32))
        done = done.astype(bool)
        return ALEState(handle=handle), Timestep(
            obs=obs,
            action=action,
            reward=reward,
            terminated=done,
            truncated=jnp.zeros_like(done),
            info={"warm": warm.astype(bool)},
        )

    def close(self, state: ALEState) -> None:
        jax.ffi.ffi_call(
            "ale_close", jax.ShapeDtypeStruct((1,), jnp.int32), has_side_effect=True
        )(state.handle)

    def observation_space(self) -> Space:
        return Space(shape=(RAM_SIZE,), dtype=jnp.uint8, low=0, high=255)

    def action_space(self) -> Space:
        return Space(shape=(), dtype=jnp.int32, low=0, high=self.num_actions - 1)


def make(env_id, **kwargs):
    return ALE(game=env_id, **kwargs)
