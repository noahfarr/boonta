from pathlib import Path

import jax.numpy as jnp
from flax import struct

from boonta.utils import Array, Key, Timestep

from ..environment import Environment
from ..spaces import Space

KEYS = ("level", "cleared", "distance", "score")
RAM_SIZE = 2048


@struct.dataclass
class SuperMarioBrosState:
    emulator: Array
    frames: Array
    ram: Array
    cleared: Array


def ending(done: Array, value: Array) -> Array:
    return jnp.where(done, value.astype(jnp.float32), jnp.nan)


class SuperMarioBros(Environment):
    def __init__(
        self,
        pool,
        snapshot,
        movement,
        episode: str = "game",
        reward: str = "progress",
        pixels: bool = True,
        stack: int = 4,
        time_limit: int = 6000,
    ):
        import pynes.xla

        if episode not in ("game", "stage"):
            raise ValueError(f"episode must be game or stage, got {episode}")
        if reward not in ("progress", "score"):
            raise ValueError(f"reward must be progress or score, got {reward}")
        self.pool = pool
        self.movement = jnp.asarray(movement, jnp.uint8)
        self.episode = episode
        self.reward = reward
        self.pixels = pixels
        self.stack = stack
        self._time_limit = time_limit
        self._advance = pynes.xla.advancer(pool)
        self._render = pynes.xla.renderer(pool)
        emulator, frame, ram = self._advance(jnp.asarray(snapshot), jnp.uint8(0))
        self.start = SuperMarioBrosState(
            emulator=emulator,
            frames=jnp.repeat(frame[None], stack, axis=0),
            ram=ram,
            cleared=jnp.zeros((), jnp.int32),
        )

    def observe(self, state: SuperMarioBrosState) -> Array:
        return state.frames if self.pixels else state.ram

    def init(self, key: Key) -> tuple[SuperMarioBrosState, Timestep]:
        return self.start, Timestep(
            obs=self.observe(self.start),
            action=jnp.zeros((), jnp.int32),
            reward=jnp.zeros((), jnp.float32),
            terminated=jnp.ones((), bool),
            truncated=jnp.zeros((), bool),
            info={name: jnp.float32(jnp.nan) for name in KEYS},
        )

    def step(
        self, key: Key, state: SuperMarioBrosState, action: Array
    ) -> tuple[SuperMarioBrosState, Timestep]:
        from pynes import smb

        emulator, frame, ram = self._advance(state.emulator, self.movement[action])
        if self.reward == "score":
            reward = (smb.score(ram) - smb.score(state.ram)).astype(jnp.float32)
        else:
            reward = smb.reward(state.ram, ram)
        advanced = smb.level(ram) != smb.level(state.ram)
        if self.episode == "game":
            done = smb.game_over(ram)
        else:
            done = smb.died(state.ram, ram) | advanced
        cleared = state.cleared + advanced.astype(jnp.int32)
        info = {
            "level": ending(done, smb.level(state.ram)),
            "cleared": ending(done, cleared),
            "distance": ending(done, smb.x_position(state.ram)),
            "score": ending(done, smb.score(ram)),
        }
        frames = jnp.concatenate([state.frames[1:], frame[None]], axis=0)
        state = SuperMarioBrosState(emulator=emulator, frames=frames, ram=ram, cleared=cleared)
        return state, Timestep(
            obs=self.observe(state),
            action=action,
            reward=reward,
            terminated=done,
            truncated=jnp.zeros((), bool),
            info=info,
        )

    def observation_space(self) -> Space:
        if self.pixels:
            return Space(
                shape=(self.stack, self.pool.obs_size, self.pool.obs_size),
                dtype=jnp.uint8,
                low=0,
                high=255,
            )
        return Space(shape=(RAM_SIZE,), dtype=jnp.uint8, low=0, high=255)

    def action_space(self) -> Space:
        return Space(shape=(), dtype=jnp.int32, low=0, high=len(self.movement) - 1)

    def time_limit(self) -> int:
        return self._time_limit

    def render(self, state: SuperMarioBrosState) -> Array:
        return self._render(state.emulator)

    def __del__(self) -> None:
        if getattr(self, "pool", None) is not None:
            self.pool.close()


def make(
    num_envs: int,
    rom: str | None = None,
    movement: str = "complex",
    frame_skip: int = 4,
    stage: tuple[int, int] | None = None,
    num_threads: int | None = None,
    **kwargs,
):
    import pynes
    from pynes import smb

    if rom is None or not Path(rom).exists():
        raise FileNotFoundError(
            f"pynes: no ROM at {rom}; boonta ships no game ROMs, set environment.kwargs.rom "
            "or BOONTA_SMB_ROM to your own Super Mario Bros. ROM"
        )
    moves = {
        "right_only": pynes.RIGHT_ONLY,
        "simple": pynes.SIMPLE_MOVEMENT,
        "complex": pynes.COMPLEX_MOVEMENT,
    }[movement]
    pool = pynes.Pool(
        rom,
        num_envs=num_envs,
        frame_skip=frame_skip,
        num_threads=num_threads,
        game="smb",
    )
    snapshot = smb.opening(pool, tuple(stage) if stage is not None else None)
    return SuperMarioBros(pool, snapshot, moves, **kwargs)
