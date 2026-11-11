import ctypes
import functools
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import lox

from boonta.utils import Array, Key, Timestep

from ..environment import Environment
from .. import build, capsule, load
from ..spaces import Space

FFI_DIR = Path(__file__).parent / "ffi"

ACHIEVEMENTS = (
    "collect_wood", "place_table", "eat_cow", "collect_sapling",
    "collect_drink", "make_wood_pickaxe", "make_wood_sword", "place_plant",
    "defeat_zombie", "collect_stone", "place_stone", "eat_plant",
    "defeat_skeleton", "make_stone_pickaxe", "make_stone_sword", "wake_up",
    "place_furnace", "collect_coal", "collect_iron", "collect_diamond",
    "make_iron_pickaxe", "make_iron_sword", "make_arrow", "make_torch",
    "place_torch", "make_diamond_sword", "make_iron_armour",
    "make_diamond_armour", "enter_gnomish_mines", "enter_dungeon",
    "enter_sewers", "enter_vault", "enter_troll_mines", "enter_fire_realm",
    "enter_ice_realm", "enter_graveyard", "defeat_gnome_warrior",
    "defeat_gnome_archer", "defeat_orc_soldier", "defeat_orc_mage",
    "defeat_lizard", "defeat_kobold", "defeat_troll", "defeat_deep_thing",
    "defeat_pigman", "defeat_fire_elemental", "defeat_frost_troll",
    "defeat_ice_elemental", "damage_necromancer", "defeat_necromancer",
    "eat_bat", "eat_snail", "find_bow", "fire_bow", "collect_sapphire",
    "learn_fireball", "cast_fireball", "learn_iceball", "cast_iceball",
    "collect_ruby", "make_diamond_pickaxe", "open_chest", "drink_potion",
    "enchant_sword", "enchant_armour", "defeat_knight", "defeat_archer",
)


@functools.cache
def library():
    build(FFI_DIR, "libcraftaxcu.so", ["binding.cu", "ffi.cc", "vendor/craftax.cu"])
    lib = load(FFI_DIR, "libcraftaxcu.so")
    lib.craftaxcu_create.restype = ctypes.c_void_p
    lib.craftaxcu_create.argtypes = [ctypes.c_int32, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint32]
    lib.craftaxcu_close.argtypes = [ctypes.c_void_p]
    lib.craftaxcu_archive.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
    lib.craftaxcu_archive.restype = ctypes.c_int32
    lib.craftaxcu_describe.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.craftaxcu_describe.restype = ctypes.c_int32
    lib.craftaxcu_map.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    lib.craftaxcu_map.restype = ctypes.c_int32
    lib.craftaxcu_soa_bytes.argtypes = [ctypes.c_int32]
    lib.craftaxcu_soa_bytes.restype = ctypes.c_int32
    lib.craftaxcu_tail_bytes.argtypes = []
    lib.craftaxcu_tail_bytes.restype = ctypes.c_int32
    lib.craftaxcu_tail_key.argtypes = []
    lib.craftaxcu_tail_key.restype = ctypes.c_int32
    lib.craftaxcu_soa_row_bytes.argtypes = []
    lib.craftaxcu_soa_row_bytes.restype = ctypes.c_int32
    for target, getter in (
        ("craftaxcu_step", "craftaxcu_ffi_step_cuda"),
        ("craftaxcu_step_full", "craftaxcu_ffi_step_full_cuda"),
        ("craftaxcu_reset_full", "craftaxcu_ffi_reset_full_cuda"),
        ("craftaxcu_observe_full", "craftaxcu_ffi_observe_full_cuda"),
        ("craftaxcu_reset", "craftaxcu_ffi_reset_cuda"),
        ("craftaxcu_observe", "craftaxcu_ffi_observe_cuda"),
        ("craftaxcu_bank", "craftaxcu_ffi_bank_cuda"),
        ("craftaxcu_revive", "craftaxcu_ffi_revive_cuda"),
        ("craftaxcu_save", "craftaxcu_ffi_save_cuda"),
        ("craftaxcu_restore", "craftaxcu_ffi_restore_cuda"),
        ("craftaxcu_cells", "craftaxcu_ffi_cells_cuda"),
        ("craftaxcu_portray", "craftaxcu_ffi_portray_cuda"),
        ("craftaxcu_worlds", "craftaxcu_ffi_worlds_cuda"),
        ("craftaxcu_laurels", "craftaxcu_ffi_laurels_cuda"),
        ("craftaxcu_action_mask", "craftaxcu_ffi_action_mask_cuda"),
        ("craftaxcu_groups", "craftaxcu_ffi_groups_cuda"),
        ("craftaxcu_owners", "craftaxcu_ffi_owners_cuda"),
        ("craftaxcu_step_archive", "craftaxcu_ffi_step_archive_cuda"),
    ):
        handler = getattr(lib, getter)
        handler.restype = ctypes.c_void_p
        jax.ffi.register_ffi_target(target, capsule(handler()), platform="CUDA")
    return lib


class CraftaxCu(Environment):
    def __init__(
        self,
        num_envs: int,
        seed: int = 0,
        fixed: int | None = None,
        coarse: int = 0,
        laurels: bool = False,
    ):
        if num_envs < 1:
            raise ValueError('num_envs must be positive')
        lib = library()
        dims = (ctypes.c_int32 * 5)()
        lib.craftaxcu_dims(dims)
        self.obs_size, self.num_actions, self.num_achievements, self.blob_bytes = (
            int(dims[0]),
            int(dims[1]),
            int(dims[3]),
            int(dims[4]),
        )
        self.state_bytes = int(dims[2])
        self.soa_bytes = int(lib.craftaxcu_soa_bytes(num_envs))
        self.tail_bytes = int(lib.craftaxcu_tail_bytes())
        self.tail_key = int(lib.craftaxcu_tail_key())
        self.soa_row_bytes = int(lib.craftaxcu_soa_row_bytes())
        self.num_envs = num_envs
        self.seed = seed
        self.fixed = fixed
        self.coarse = coarse
        self._lib = lib
        self._pool = int(
            lib.craftaxcu_create(
                num_envs, seed, 0 if fixed is None else int(fixed) + 1, int(coarse)
            )
        )
        shapes = (
            jax.ShapeDtypeStruct((num_envs, self.obs_size), jnp.float32),
            jax.ShapeDtypeStruct((num_envs,), jnp.float32),
            jax.ShapeDtypeStruct((num_envs,), jnp.float32),
        )
        self._observe = jax.ffi.ffi_call("craftaxcu_observe", shapes, has_side_effect=True)
        self._step = jax.ffi.ffi_call("craftaxcu_step", shapes, has_side_effect=True)
        self._reset = jax.ffi.ffi_call("craftaxcu_reset", shapes, has_side_effect=True)
        # A per-env row lets a JAX archive gather one env with a row index.
        self.soa_shape = (
            (num_envs, self.soa_row_bytes)
            if self.soa_row_bytes
            else (self.soa_bytes,)
        )
        held = (
            jax.ShapeDtypeStruct((num_envs, self.state_bytes), jnp.uint8),
            jax.ShapeDtypeStruct(self.soa_shape, jnp.uint8),
            jax.ShapeDtypeStruct((num_envs, self.tail_bytes), jnp.uint8),
        )
        self._reset_full = jax.ffi.ffi_call(
            "craftaxcu_reset_full", (*held, *shapes), has_side_effect=True
        )
        self._observe_full = jax.ffi.ffi_call(
            "craftaxcu_observe_full", shapes, has_side_effect=True
        )
        self._step_full = jax.ffi.ffi_call(
            "craftaxcu_step_full",
            (
                *held,
                *shapes,
                jax.ShapeDtypeStruct((num_envs, 2 * self.num_achievements + 1), jnp.float32),
            ),
            has_side_effect=True,
            input_output_aliases={0: 0, 1: 1, 2: 2},
        )
        self._bank = jax.ffi.ffi_call(
            "craftaxcu_bank",
            held,
            has_side_effect=True,
            input_output_aliases={0: 0, 1: 1, 2: 2},
        )
        self._revive = jax.ffi.ffi_call(
            "craftaxcu_revive",
            held,
            has_side_effect=True,
            input_output_aliases={0: 0, 1: 1, 2: 2},
        )
        blob = jax.ShapeDtypeStruct((num_envs, self.blob_bytes), jnp.uint8)
        self._save = jax.ffi.ffi_call("craftaxcu_save", blob, has_side_effect=True)
        self._restore = jax.ffi.ffi_call(
            "craftaxcu_restore",
            jax.ShapeDtypeStruct((1,), jnp.int32),
            has_side_effect=True,
        )
        self._portray = jax.ffi.ffi_call(
            "craftaxcu_portray",
            jax.ShapeDtypeStruct((num_envs, 9), jnp.int32),
            has_side_effect=True,
        )
        self._cells = jax.ffi.ffi_call(
            "craftaxcu_cells",
            jax.ShapeDtypeStruct((num_envs,), jnp.uint32),
            has_side_effect=True,
        )
        self._worlds = jax.ffi.ffi_call(
            "craftaxcu_worlds",
            jax.ShapeDtypeStruct((num_envs,), jnp.uint32),
            has_side_effect=True,
        )
        self._laurels = (
            jax.ffi.ffi_call(
                "craftaxcu_laurels",
                jax.ShapeDtypeStruct((num_envs, self.num_achievements), jnp.float32),
                has_side_effect=True,
            )
            if laurels
            else None
        )
        self._action_mask = jax.ffi.ffi_call(
            "craftaxcu_action_mask",
            jax.ShapeDtypeStruct((num_envs, self.num_actions), jnp.uint8),
            has_side_effect=True,
        )

    def observe(self, state: Any, key: Key | None = None) -> dict[str, Array]:
        obs, _, _ = self._observe_full(*state, pool=self._pool)
        return self.dress(state, obs)

    def bank(self, state: Any, slots: Array) -> Any:
        return tuple(self._bank(*state, slots.astype(jnp.int32), pool=self._pool))

    def revive(self, state: Any, slots: Array) -> Any:
        return tuple(self._revive(*state, slots.astype(jnp.int32), pool=self._pool))

    def reseed(self, state: Any, key: Key) -> Any:
        states, soa, tail = state
        words = jax.random.bits(key, (self.num_envs, 2), jnp.uint32)
        fresh = jax.lax.bitcast_convert_type(words, jnp.uint8).reshape(self.num_envs, 8)
        return states, soa, tail.at[:, self.tail_key : self.tail_key + 8].set(fresh)

    def legal(self, state: Any) -> Array:
        states, soa, _ = state
        return self._action_mask(states, soa, pool=self._pool) > 0

    def action_mask(self, state: Any) -> Array:
        return self.legal(state)

    def dress(self, state: Any, obs: Array) -> dict[str, Array]:
        return {"view": obs, "mask": self.legal(state)}

    def timestep(self, state, obs, action, reward, terminated) -> Timestep:
        states, soa, _ = state
        return Timestep(
            obs=self.dress(state, obs),
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=jnp.zeros((self.num_envs,), jnp.bool_),
            info=(
                {}
                if self._laurels is None
                else {"laurels": self._laurels(states, soa, pool=self._pool)}
            ),
        )

    def init(self, key: Key, group_size: int = 1) -> tuple[Any, Timestep]:
        *state, obs, reward, terminated = self._reset_full(pool=self._pool)
        state = tuple(state)
        return state, self.timestep(
            state,
            obs,
            jnp.zeros((self.num_envs,), jnp.int32),
            jnp.zeros((self.num_envs,), jnp.float32),
            jnp.ones((self.num_envs,), jnp.bool_),
        )

    def step(self, key: Key, state: Any, action: Array) -> tuple[Any, Timestep]:
        *state, obs, reward, terminated, tally = self._step_full(
            *state, action.astype(jnp.int32), pool=self._pool
        )
        self.tally(tally)
        state = tuple(state)
        return state, self.timestep(state, obs, action, reward, terminated > 0.5)

    def tally(self, tally: Array) -> None:
        count = len(ACHIEVEMENTS)
        episodes, held, unlocked = tally[:, 0], tally[:, 1 : count + 1], tally[:, count + 1 :]
        ended = episodes > 0.0
        rate = held / jnp.maximum(episodes, 1.0)[:, None]
        when = jnp.where(held > 0.0, unlocked / jnp.maximum(held, 1.0), jnp.nan)
        lox.log(
            {
                f"achievements/{name}": jnp.where(ended, rate[:, index], jnp.nan)
                for index, name in enumerate(ACHIEVEMENTS)
            }
            | {
                f"unlocked/{name}": jnp.where(ended, when[:, index], jnp.nan)
                for index, name in enumerate(ACHIEVEMENTS)
            }
            | {"achievements/count": jnp.where(ended, jnp.sum(rate, axis=-1), jnp.nan)}
        )

    def save(self, index: Array | None = None) -> Array:
        if index is None:
            index = jnp.arange(self.num_envs, dtype=jnp.int32)
        call = jax.ffi.ffi_call(
            "craftaxcu_save",
            jax.ShapeDtypeStruct((index.shape[0], self.blob_bytes), jnp.uint8),
            has_side_effect=True,
        )
        return call(index.astype(jnp.int32), pool=self._pool)

    def restore(self, index: Array, blob: Array) -> Array:
        return self._restore(
            index.astype(jnp.int32), blob.astype(jnp.uint8), pool=self._pool
        )

    def archive(self, capacity: int, reservoir: int = 1) -> None:
        code = self._lib.craftaxcu_archive(
            ctypes.c_void_p(self._pool), ctypes.c_int32(capacity), ctypes.c_int32(reservoir)
        )
        if code != 0:
            need = capacity * self.blob_bytes / 1e9
            raise MemoryError(
                f"craftax.cu archive of {capacity} cells needs {need:.1f} GB "
                "and could not be allocated"
            )
        self.capacity = capacity
        self.cells_held = capacity // max(int(reservoir), 1)
        self._step_archive = jax.ffi.ffi_call(
            "craftaxcu_step_archive",
            (
                jax.ShapeDtypeStruct((self.num_envs, self.state_bytes), jnp.uint8),
                jax.ShapeDtypeStruct(self.soa_shape, jnp.uint8),
                jax.ShapeDtypeStruct((self.num_envs, self.tail_bytes), jnp.uint8),
                jax.ShapeDtypeStruct((self.num_envs, self.obs_size), jnp.float32),
                jax.ShapeDtypeStruct((self.num_envs,), jnp.float32),
                jax.ShapeDtypeStruct((self.num_envs,), jnp.float32),
                jax.ShapeDtypeStruct((self.num_envs,), jnp.int32),
                jax.ShapeDtypeStruct((self.num_envs,), jnp.int32),
                jax.ShapeDtypeStruct((self.num_envs,), jnp.int32),
                jax.ShapeDtypeStruct(
                    (self.num_envs, self.num_achievements), jnp.float32
                ),
                jax.ShapeDtypeStruct((self.num_envs,), jnp.int32),
                jax.ShapeDtypeStruct((self.num_envs,), jnp.int32),
            ),
            has_side_effect=True,
            input_output_aliases={0: 0, 1: 1, 2: 2},
        )

    def step_archive(
        self,
        state: Any,
        action: Array,
        weights: Array,
        share: float,
        floor: float = 0.0,
        leash: float = 0.0,
        refresh: Array | float = 0.0,
        recent: Array | float = 0.0,
    ):
        weights = weights.astype(jnp.float32)
        cdf = jnp.cumsum(jnp.maximum(weights, 0.0))
        held = self._step_archive(
            *state,
            action.astype(jnp.int32),
            weights,
            jnp.concatenate([cdf, cdf[-1:]]),
            jnp.reshape(jnp.asarray(share, jnp.float32), (1,)),
            jnp.reshape(jnp.asarray(floor, jnp.float32), (1,)),
            jnp.reshape(jnp.asarray(leash, jnp.float32), (1,)),
            jnp.reshape(jnp.asarray(refresh, jnp.float32), (1,)),
            jnp.reshape(jnp.asarray(recent, jnp.float32), (1,)),
            pool=self._pool,
        )
        return (tuple(held[:3]), *held[3:])

    def portray(self, rows: Any) -> Array:
        states, soa, _ = rows
        count = states.shape[0]
        if count == self.num_envs:
            return self._portray(states, soa, pool=self._pool)
        sheets = []
        for start in range(0, count, self.num_envs):
            taken = jnp.arange(start, start + self.num_envs) % count
            sheets.append(self._portray(states[taken], soa[taken], pool=self._pool))
        return jnp.concatenate(sheets)[:count]

    def cells(self, state: Any) -> Array:
        states, soa, _ = state
        return self._cells(states, soa, pool=self._pool)

    def worlds(self, state: Any) -> Array:
        states, soa, _ = state
        return self._worlds(states, soa, pool=self._pool)

    def groups(self, after: Array | None = None) -> Array:
        token = jnp.zeros((1,), jnp.float32) if after is None else after
        call = jax.ffi.ffi_call(
            "craftaxcu_groups",
            jax.ShapeDtypeStruct((self.cells_held,), jnp.int32),
            has_side_effect=True,
        )
        return call(token, pool=self._pool)

    def owners(self, after: Array | None = None) -> Array:
        token = jnp.zeros((1,), jnp.float32) if after is None else after
        call = jax.ffi.ffi_call(
            "craftaxcu_owners",
            jax.ShapeDtypeStruct((self.cells_held,), jnp.int32),
            has_side_effect=True,
        )
        return call(token, pool=self._pool)

    def describe(self):
        import numpy as np

        sheet = np.empty((self.cells_held, 9), np.int32)
        code = self._lib.craftaxcu_describe(
            ctypes.c_void_p(self._pool), sheet.ctypes.data_as(ctypes.c_void_p)
        )
        if code != 0:
            raise RuntimeError(f"craftax.cu describe failed with code {code}")
        return sheet

    def terrain(self, env: int = 0):
        import numpy as np

        levels = np.empty((9, 48, 48), np.uint8)
        code = self._lib.craftaxcu_map(
            ctypes.c_void_p(self._pool), ctypes.c_int32(env),
            levels.ctypes.data_as(ctypes.c_void_p),
        )
        if code != 0:
            raise RuntimeError(f"craftax.cu map failed with code {code}")
        return levels

    def understudy(self, num_envs: int) -> "CraftaxCu":
        return CraftaxCu(
            num_envs,
            seed=self.seed + 7919,
            fixed=self.fixed,
            coarse=self.coarse,
        )

    def close(self, state: Any) -> None:
        return None

    def __del__(self) -> None:
        if getattr(self, "_pool", 0):
            self._lib.craftaxcu_close(ctypes.c_void_p(self._pool))
            self._pool = 0

    def observation_space(self) -> dict[str, Space]:
        return {
            "view": Space((self.obs_size,), jnp.float32),
            "mask": Space(
                shape=(self.num_actions,),
                dtype=jnp.bool_,
                low=0,
                high=1,
            ),
        }

    def action_space(self) -> Space:
        return Space(
            shape=(),
            dtype=jnp.int32,
            low=0,
            high=self.num_actions - 1,
        )


def make(env_id, **kwargs):
    return CraftaxCu(**kwargs)
