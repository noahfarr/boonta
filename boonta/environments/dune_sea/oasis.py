import functools

import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .dune_sea import DuneSea
from .generators import Generator, cave_generator
from .state import MAX_ENERGY, DuneSeaState

OASIS_RESERVE = 1.5
DRINK_RATE = 0.05
POND_CELLS = 10
MIN_RADIUS = 1.7
MAX_RADIUS = 3.1
EDGE_JITTER = 0.9
MIN_NEIGHBOURS = 3
CLEARANCE = 5
POND_SPACING = 12
BUSH_CLEARANCE = 2
ENERGY_COST = 0.004


def dilate(mask: Array, reach: int, outside: float = 0.0) -> Array:
    height, width = mask.shape
    padded = jnp.pad(mask.astype(jnp.float32), reach, constant_values=outside)
    rows = functools.reduce(
        jnp.maximum, [padded[dy : dy + height] for dy in range(2 * reach + 1)]
    )
    return functools.reduce(
        jnp.maximum, [rows[:, dx : dx + width] for dx in range(2 * reach + 1)]
    )


def spawn_pond(key: Key, wall: Array, oasis: Array, reserve: float) -> Array:
    height, width = wall.shape
    center_key, radius_key, edge_key = jax.random.split(key, 3)
    open_ground = wall == 0
    room = sum(
        (dilate(wall > 0, reach, outside=1.0) == 0).astype(jnp.int32) for reach in range(1, CLEARANCE + 1)
    )
    far = dilate(oasis > 0, POND_SPACING) == 0
    room = jnp.where(far & open_ground, room, -1)
    free = ((room == room.max()) & open_ground).reshape(-1).astype(jnp.float32)
    center = jax.random.choice(center_key, height * width, p=free / free.sum())
    centre_row, centre_column = center // width, center % width
    room_at_centre = room.reshape(-1)[center]
    radius = jax.random.uniform(radius_key, minval=MIN_RADIUS, maxval=MAX_RADIUS)
    radius = jnp.minimum(radius, jnp.maximum(1.4, room_at_centre - 0.6))
    ragged = jax.random.uniform(edge_key, (height, width), minval=-EDGE_JITTER, maxval=EDGE_JITTER)
    rows, columns = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    distance = jnp.hypot(rows - centre_row, columns - centre_column)
    centre = (rows == centre_row) & (columns == centre_column)
    inside = ((distance + ragged < radius) & (wall == 0)) | centre
    padded = jnp.pad(inside.astype(jnp.float32), 1)
    around = sum(
        padded[dy : dy + height, dx : dx + width]
        for dy in range(3)
        for dx in range(3)
        if (dy, dx) != (1, 1)
    )
    inside = inside & ((around >= MIN_NEIGHBOURS) | centre)
    return jnp.where(inside & (wall == 0) & (oasis == 0), reserve, oasis)


class Oasis(DuneSea):

    def __init__(
        self,
        height: int = 64,
        width: int = 64,
        num_agents: int = 16,
        view_size: int = 7,
        time_limit: int = 800,
        generator: Generator = cave_generator,
        num_oases: int = 3,
        reserve: float = OASIS_RESERVE,
        drink_rate: float = DRINK_RATE,
        beacon_slots: int = 3,
        energy_cost: float = ENERGY_COST,
        min_live_cells: int | None = None,
    ):
        base_generator = generator

        def generate(key, height, width, num_agents):
            cells, agent = base_generator(key, height, width, num_agents)
            oasis = cells.oasis
            for pond_key in jax.random.split(jax.random.fold_in(key, 11), num_oases):
                oasis = spawn_pond(pond_key, cells.wall, oasis, reserve)
            bush = jnp.where(dilate(oasis > 0, BUSH_CLEARANCE) > 0, 0.0, cells.bush)
            return cells.replace(energy=jnp.zeros_like(cells.energy), oasis=oasis, bush=bush), agent

        super().__init__(
            height,
            width,
            num_agents,
            view_size,
            time_limit,
            generate,
            energy_cost=energy_cost,
            observe_oasis=True,
            beacon_slots=beacon_slots,
        )
        self.reserve = reserve
        self.drink_rate = drink_rate
        self.min_live_cells = POND_CELLS * (num_oases - 1) + 1 if min_live_cells is None else min_live_cells

    def step(
        self, key: Key, state: DuneSeaState, action: Array
    ) -> tuple[DuneSeaState, Timestep]:
        thirsty = state.agent.energy > 0
        state, timestep = super().step(key, state, action)
        cells = state.cells
        y, x = state.agent.position[:, 0], state.agent.position[:, 1]

        offered = jnp.where(thirsty, jnp.minimum(self.drink_rate, cells.oasis[y, x]), 0.0)
        energy = jnp.clip(state.agent.energy + offered, 0.0, MAX_ENERGY)
        drunk = energy - state.agent.energy
        oasis = cells.oasis.at[y, x].add(-drunk)
        oasis = jnp.where(oasis < 1e-4, 0.0, oasis)

        dried = (oasis > 0).sum() < self.min_live_cells
        replacement = spawn_pond(jax.random.fold_in(key, 7), cells.wall, oasis, self.reserve)
        oasis = jnp.where(dried, replacement, oasis)

        bush = jnp.where(dilate(oasis > 0, BUSH_CLEARANCE) > 0, 0.0, cells.bush)
        state = state.replace(
            cells=cells.replace(oasis=oasis, bush=bush), agent=state.agent.replace(energy=energy)
        )
        timestep = timestep.replace(
            obs=self.observe(state),
            reward=timestep.reward + drunk,
            terminated=energy <= 0.0,
        )
        return state, timestep
