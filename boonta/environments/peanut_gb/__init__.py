import ctypes
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from boonta.utils import Array, Key, Timestep

from .. import build, load as load_library, register_targets
from ..environment import Environment
from ..spaces import Space

RAM_SIZE = 8192
RAM_BASE = 0xC000
BURST = 16
SCREEN_HEIGHT, SCREEN_WIDTH = 144, 160
ACTIONS = ("noop", "up", "down", "left", "right", "a", "b", "start")
DIRECTORY = Path(os.path.dirname(__file__)) / "ffi"
ROMS = Path(os.path.dirname(__file__)) / "roms"
START_STATES = Path(os.path.dirname(__file__)) / "start_states"

library = None


def load():
    global library
    if library is None:
        build(
            DIRECTORY,
            "libpokemon.so",
            ["pool.cpp", "ffi.cc", "peanut.c", "profile.c", "vendor/peanut_gb.h"],
        )
        held = load_library(DIRECTORY, "libpokemon.so")
        held.pokemon_create.restype = ctypes.c_int32
        held.pokemon_create.argtypes = [ctypes.c_char_p] + [ctypes.c_int32] * 5
        held.pokemon_state_size.restype = ctypes.c_int64
        held.pokemon_state_size.argtypes = [ctypes.c_int32]
        for name in ("num_actions", "obs_height", "obs_width"):
            getattr(held, f"pokemon_{name}").restype = ctypes.c_int32
            getattr(held, f"pokemon_{name}").argtypes = [ctypes.c_int32]
        for name in ("reset", "errors"):
            getattr(held, f"pokemon_{name}").restype = ctypes.c_int32
            getattr(held, f"pokemon_{name}").argtypes = [ctypes.c_int32] * 2
        held.pokemon_press.restype = ctypes.c_int32
        held.pokemon_press.argtypes = [ctypes.c_int32] * 5
        held.pokemon_read.restype = ctypes.c_int32
        held.pokemon_read.argtypes = [ctypes.c_int32] * 3
        for name in ("screen", "save", "load"):
            getattr(held, f"pokemon_{name}").restype = ctypes.c_int32
            getattr(held, f"pokemon_{name}").argtypes = [
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.c_void_p,
            ]
        held.pokemon_observe.restype = ctypes.c_int32
        held.pokemon_observe.argtypes = [ctypes.c_int32] * 2 + [ctypes.c_void_p] * 2
        held.pokemon_destroy.restype = ctypes.c_int32
        held.pokemon_destroy.argtypes = [ctypes.c_int32]
        register_targets(
            held,
            {"pokemon_advance": "pokemon_ffi_advance", "pokemon_render": "pokemon_ffi_render"},
        )
        library = held
    return library


class Pool:
    def __init__(
        self,
        rom,
        num_envs: int = 1,
        frame_skip: int = 24,
        hold: int = 8,
        num_threads: int | None = None,
        seed: int = 0,
    ):
        self.lib = load()
        self.num_envs = num_envs
        self.frame_skip = frame_skip
        self.hold = hold
        self.handle = self.lib.pokemon_create(
            str(rom).encode(),
            num_envs,
            frame_skip,
            hold,
            num_threads or len(os.sched_getaffinity(0)),
            seed,
        )
        if self.handle < 0:
            raise RuntimeError(f"peanut_gb: could not load {rom}")
        self.state_size = int(self.lib.pokemon_state_size(self.handle))
        self.num_actions = int(self.lib.pokemon_num_actions(self.handle))
        self.obs_height = int(self.lib.pokemon_obs_height(self.handle))
        self.obs_width = int(self.lib.pokemon_obs_width(self.handle))

    def press(self, action: int, hold: int | None = None, frames: int | None = None, slot: int = 0):
        self.lib.pokemon_press(
            self.handle,
            slot,
            int(action),
            self.hold if hold is None else hold,
            self.frame_skip if frames is None else frames,
        )

    def read(self, address: int, slot: int = 0) -> int:
        return int(self.lib.pokemon_read(self.handle, slot, int(address)))

    def screen(self, slot: int = 0) -> np.ndarray:
        rgb = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), np.uint8)
        self.lib.pokemon_screen(self.handle, slot, rgb.ctypes.data)
        return rgb

    def observe(self, slot: int = 0):
        frame = np.zeros((self.obs_height, self.obs_width), np.uint8)
        ram = np.zeros(RAM_SIZE, np.uint8)
        self.lib.pokemon_observe(self.handle, slot, frame.ctypes.data, ram.ctypes.data)
        return frame, ram

    def save(self, slot: int = 0) -> np.ndarray:
        state = np.zeros(self.state_size, np.uint8)
        if self.lib.pokemon_save(self.handle, slot, state.ctypes.data) != 0:
            raise RuntimeError("peanut_gb: state serialization failed")
        return state

    def restore(self, state, slot: int = 0) -> None:
        held = np.ascontiguousarray(np.asarray(state, np.uint8))
        if self.lib.pokemon_load(self.handle, slot, held.ctypes.data) != 0:
            raise RuntimeError("peanut_gb: state restore failed")

    def reset(self, slot: int = 0) -> None:
        self.lib.pokemon_reset(self.handle, slot)

    def errors(self, slot: int = 0) -> int:
        return int(self.lib.pokemon_errors(self.handle, slot))

    def close(self) -> None:
        if getattr(self, "handle", None) is not None and self.handle >= 0:
            self.lib.pokemon_destroy(self.handle)
            self.handle = None


def advancer(pool: Pool):
    shapes = (
        jax.ShapeDtypeStruct((pool.state_size,), jnp.uint8),
        jax.ShapeDtypeStruct((pool.obs_height, pool.obs_width), jnp.uint8),
        jax.ShapeDtypeStruct((RAM_SIZE,), jnp.uint8),
    )
    call = jax.ffi.ffi_call(
        "pokemon_advance", shapes, vmap_method="broadcast_all", input_output_aliases={0: 0}
    )

    def advance(state, action):
        return call(state, jnp.asarray(action, jnp.int32), handle=np.int32(pool.handle))

    return advance


def renderer(pool: Pool):
    call = jax.ffi.ffi_call(
        "pokemon_render",
        jax.ShapeDtypeStruct((SCREEN_HEIGHT, SCREEN_WIDTH, 3), jnp.uint8),
        vmap_method="broadcast_all",
    )

    def render(state):
        return call(state, handle=np.int32(pool.handle))

    return render


@struct.dataclass
class GameBoyState:
    emulator: Array
    frames: Array
    ram: Array
    visited: Array
    walked: Array
    charted: Array
    flags: Array
    downed: Array
    climbed: Array
    trained: Array
    tallied: Array


def visit(visited: Array, index: Array) -> tuple[Array, Array]:
    word, bit = index // 32, jnp.uint32(1) << (index % 32).astype(jnp.uint32)
    seen = (visited[word] & bit) > 0
    return visited.at[word].set(visited[word] | bit), seen


def forget(visited: Array, stale: Array) -> Array:
    return jnp.where(stale, jnp.zeros_like(visited), visited)


def visits(visited: Array) -> Array:
    return jnp.sum(jax.lax.population_count(visited))


def sift(held: Array, fresh: Array) -> tuple[Array, Array]:
    gained = (visits(fresh) - visits(held)).astype(jnp.float32)
    sane = gained <= BURST
    return jnp.where(sane, fresh, held), jnp.where(sane, gained, 0.0)


def byte(ram, address: int):
    return ram[..., address - RAM_BASE].astype(jnp.int32)


def flag(ram, base: int, bit: int):
    return (byte(ram, base + bit // 8) >> (bit % 8)) & 1


def core(state) -> GameBoyState:
    held = state
    while not isinstance(held, GameBoyState):
        held = held.env_state
    return held


class GameBoy(Environment):
    def __init__(
        self,
        pool: Pool,
        snapshot,
        game,
        stack: int = 4,
        horizon: int = 16384,
        flag_reward: float = 0.0,
        level_reward: float = 0.0,
        experience_reward: float = 0.0,
        heal_reward: float = 0.0,
        faint_penalty: float = 0.0,
        catch_reward: float = 0.0,
        map_reward: float = 0.0,
        tile_reward: float = 0.0,
        menu_penalty: float = 0.0,
    ):
        self.pool = pool
        self.game = game
        self.stack = stack
        self._horizon = horizon
        self.flag_reward = flag_reward
        self.level_reward = level_reward
        self.experience_reward = experience_reward
        self.heal_reward = heal_reward
        self.faint_penalty = faint_penalty
        self.catch_reward = catch_reward
        self.map_reward = map_reward
        self.tile_reward = tile_reward
        self.menu_penalty = menu_penalty
        self._advance = advancer(pool)
        self._render = renderer(pool)
        emulator, frame, ram = snapshot
        ram = jnp.asarray(ram)
        visited, _ = visit(jnp.zeros((game.NUM_MAPS + 31) // 32, jnp.uint32), game.place(ram))
        walked, _ = visit(jnp.zeros((game.NUM_CELLS + 31) // 32, jnp.uint32), game.walk(ram))
        self.start = GameBoyState(
            emulator=jnp.asarray(emulator),
            frames=jnp.repeat(jnp.asarray(frame)[None], stack, axis=0),
            ram=ram,
            visited=visited,
            walked=walked,
            charted=walked,
            flags=game.lit(ram),
            downed=jnp.zeros((), bool),
            climbed=game.leveling(ram),
            trained=game.earned(ram),
            tallied=game.caught(ram),
        )

    def observe(self, state: GameBoyState) -> dict:
        height, width = state.frames.shape[-2:]
        seen = self.game.crop(state.charted, state.ram, height, width)
        return {
            "image": state.frames,
            "tiles": seen[None],
            "flags": self.game.bits(state.ram),
            "vitals": self.game.vitals(state.ram),
        }

    def tally(self, state: GameBoyState, fell: Array, mending: Array) -> dict:
        info = jax.tree.map(lambda value: value.astype(jnp.float32), self.game.annotate(state.ram))
        info["maps"] = visits(state.visited).astype(jnp.float32)
        info["tiles"] = visits(state.walked).astype(jnp.float32)
        info["deaths"] = fell.astype(jnp.float32) * self._horizon
        info["healed"] = mending * self._horizon
        info["experience"] = state.trained
        info["caught"] = state.tallied.astype(jnp.float32)
        return info

    def init(self, key: Key) -> tuple[GameBoyState, Timestep]:
        return self.start, Timestep(
            obs=self.observe(self.start),
            action=jnp.zeros((), jnp.int32),
            reward=jnp.zeros((), jnp.float32),
            terminated=jnp.ones((), bool),
            truncated=jnp.zeros((), bool),
            info=self.tally(self.start, jnp.zeros((), bool), jnp.zeros((), jnp.float32)),
        )

    def step(self, key: Key, state: GameBoyState, action: Array) -> tuple[GameBoyState, Timestep]:
        game = self.game
        emulator, frame, ram = self._advance(state.emulator, action)

        flags, gained = sift(state.flags, state.flags | game.lit(ram))
        reward = self.flag_reward * gained

        climbed = jnp.maximum(state.climbed, game.leveling(ram))
        reward += self.level_reward * (climbed - state.climbed).astype(jnp.float32)
        trained = jnp.maximum(state.trained, game.earned(ram))
        reward += self.experience_reward * (trained - state.trained)

        health, ailing = game.vitality(ram), game.vitality(state.ram)
        steady = (game.party_size(ram) == game.party_size(state.ram)) & (
            game.capacity(ram) == game.capacity(state.ram)
        )
        sunk = game.fainted(ram)
        mended, downed = game.tending(health, ailing, steady, sunk, state.downed)
        mending = jnp.where(mended, health - ailing, 0.0)
        reward += self.heal_reward * mending
        fell = sunk & (ailing > 0)
        reward -= self.faint_penalty * fell.astype(jnp.float32)

        tallied = jnp.maximum(state.tallied, game.caught(ram))
        reward += self.catch_reward * (tallied - state.tallied).astype(jnp.float32)
        reward -= self.menu_penalty * game.menu(ram).astype(jnp.float32)

        stale = gained > 0
        visited, seen = visit(forget(state.visited, stale), game.place(ram))
        entered = (1.0 - seen.astype(jnp.float32)) * (1.0 - stale.astype(jnp.float32))
        reward += self.map_reward * game.extent(ram) * entered
        walked, trod = visit(forget(state.walked, stale), game.walk(ram))
        charted, _ = visit(state.charted, game.walk(ram))
        reward += self.tile_reward * (1.0 - trod.astype(jnp.float32))

        state = GameBoyState(
            emulator=emulator,
            frames=jnp.concatenate([state.frames[1:], frame[None]], axis=0),
            ram=ram,
            visited=visited,
            walked=walked,
            charted=charted,
            flags=flags,
            downed=downed,
            climbed=climbed,
            trained=trained,
            tallied=tallied,
        )
        return state, Timestep(
            obs=self.observe(state),
            action=action,
            reward=reward,
            terminated=jnp.zeros((), bool),
            truncated=jnp.zeros((), bool),
            info=self.tally(state, fell, mending),
        )

    def observation_space(self) -> dict[str, Space]:
        shape = (self.stack, self.pool.obs_height, self.pool.obs_width)
        blank = self.start.ram
        return {
            "image": Space(shape=shape, dtype=jnp.uint8, low=0, high=255),
            "tiles": Space(shape=(1, *shape[1:]), dtype=jnp.uint8, low=0, high=255),
            "flags": Space(shape=self.game.bits(blank).shape, dtype=jnp.uint8, low=0, high=1),
            "vitals": Space(
                shape=self.game.vitals(blank).shape, dtype=jnp.float32, low=0.0, high=1.0
            ),
        }

    def action_space(self) -> Space:
        return Space(shape=(), dtype=jnp.int32, low=0, high=len(ACTIONS) - 1)

    def time_limit(self) -> int:
        return self._horizon

    def render(self, state: GameBoyState, scale: int = 2) -> Array:
        screen = self._render(state.emulator)
        return jnp.repeat(jnp.repeat(screen, scale, axis=-3), scale, axis=-2)

    def __del__(self) -> None:
        if getattr(self, "pool", None) is not None:
            self.pool.close()


def start(pool: Pool, path):
    state = np.fromfile(path, np.uint8)
    if state.size != pool.state_size:
        raise ValueError(
            f"{path} holds {state.size} bytes but the emulator state is {pool.state_size}; "
            "regenerate it with hangar/scripts/opening_pokemon_red.py"
        )
    pool.restore(state)
    frame, ram = pool.observe()
    return state, frame, ram


from . import pokemon_red  # noqa: E402

games = {"pokemon_red": pokemon_red}


def make(
    env_id,
    num_envs: int,
    frame_skip: int = 24,
    hold: int = 8,
    num_threads: int | None = None,
    seed: int = 0,
    **kwargs,
):
    rom = ROMS / f"{env_id}.gb"
    if not rom.exists():
        raise FileNotFoundError(
            f"peanut_gb: no ROM at {rom}; boonta ships no game ROMs, supply your own"
        )
    pool = Pool(
        rom,
        num_envs=num_envs,
        frame_skip=frame_skip,
        hold=hold,
        num_threads=num_threads,
        seed=seed,
    )
    return GameBoy(pool, start(pool, START_STATES / f"{env_id}.bin"), games[env_id], **kwargs)
