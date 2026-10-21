import functools

import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .dune_sea import DuneSea
from .generators import Generator, cave_generator, place_agents
from .state import MAX_ENERGY, DuneSeaState

OASIS_RESERVE = 0.45
DRINK_RATE = 0.05
EVAPORATION = 0.0
MIRAGE_START = 200
MIRAGE_END = 600
SWEEP = 0.0
MIN_RADIUS = 2.4
MAX_RADIUS = 4.4
EDGE_JITTER = 0.9
MIN_NEIGHBOURS = 3
CLEARANCE = 5
POND_SPACING = 12
BUSH_CLEARANCE = 2
ENERGY_COST = 0.004
CAVE_FILL = 0.39


def dilate(mask: Array, reach: int, outside: float = 0.0) -> Array:
    height, width = mask.shape
    padded = jnp.pad(mask.astype(jnp.float32), reach, constant_values=outside)
    rows = functools.reduce(
        jnp.maximum, [padded[dy : dy + height] for dy in range(2 * reach + 1)]
    )
    return functools.reduce(
        jnp.maximum, [rows[:, dx : dx + width] for dx in range(2 * reach + 1)]
    )


def smooth_cells(mask: Array, rounds: int = 2) -> Array:
    for _ in range(rounds):
        padded = jnp.pad(mask, 1).astype(jnp.int32)
        count = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        mask = jnp.where(mask, count >= 2, count >= 3)
    return mask


def spawn_pond(key: Key, wall: Array, oasis: Array, reserve: float) -> Array:
    height, width = wall.shape
    center_key, radius_key, edge_key = jax.random.split(key, 3)
    open_ground = wall == 0
    crowded = (wall > 0).astype(jnp.float32)
    room = jnp.zeros((height, width), jnp.int32)
    for _ in range(CLEARANCE):
        crowded = dilate(crowded, 1, outside=1.0)
        room = room + (crowded == 0).astype(jnp.int32)
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
        generator: Generator = functools.partial(cave_generator, fill=CAVE_FILL),
        num_oases: int = 2,
        num_mirages: int = 1,
        mirage_start: int = MIRAGE_START,
        mirage_end: int = MIRAGE_END,
        sweep: float = SWEEP,
        reserve: float = OASIS_RESERVE,
        drink_rate: float = DRINK_RATE,
        evaporation: float = EVAPORATION,
        beacon_slots: int = 3,
        energy_cost: float = ENERGY_COST,
    ):
        assert num_mirages <= num_oases
        base_generator = generator

        def generate(key, height, width, num_agents):
            cells, agent = base_generator(key, height, width, num_agents)
            ponds = jnp.zeros_like(cells.oasis)
            keys = jax.random.split(jax.random.fold_in(key, 11), num_oases + num_mirages)
            for number, pond_key in enumerate(keys, start=1):
                grown = spawn_pond(pond_key, cells.wall, jnp.maximum(ponds, cells.oasis), 1.0)
                added = smooth_cells((grown > 0) & (ponds == 0) & (cells.oasis == 0)) & (cells.wall == 0) & (ponds == 0)
                ponds = jnp.where(added, float(number), ponds)
            oasis = jnp.where((ponds >= 1) & (ponds <= num_oases), reserve, cells.oasis)
            columns = jnp.broadcast_to(jnp.arange(width), (height, width))
            starts = jnp.sort(jax.random.randint(jax.random.fold_in(key, 17), (num_mirages,), mirage_start, mirage_end + 1))
            mirage = cells.mirage
            for number in range(1, num_mirages + 1):
                for pond, sign in ((number, 1.0), (num_oases + number, -1.0)):
                    pond_cells = ponds == pond
                    edge = jnp.where(pond_cells, columns, width).min()
                    when = starts[number - 1].astype(jnp.float32) + (columns - edge) * sweep
                    mirage = jnp.where(pond_cells, sign * when, mirage)
            bush = jnp.where(dilate(ponds > 0, BUSH_CLEARANCE) > 0, 0.0, cells.bush)
            cells = cells.replace(energy=jnp.zeros_like(cells.energy), oasis=oasis, bush=bush, mirage=mirage)
            row, column = agent.position[:, 0], agent.position[:, 1]
            standing = oasis[row, column] > 0
            crowded = jnp.zeros_like(oasis).at[row, column].set(1.0)
            moved = place_agents(jax.random.fold_in(key, 23), (cells.wall == 0) & (oasis == 0) & (crowded == 0), num_agents)
            agent = agent.replace(
                position=jnp.where(standing[:, None], moved.position, agent.position),
                direction=jnp.where(standing, moved.direction, agent.direction),
            )
            return cells, agent

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
        self.evaporation = evaporation

    def step(
        self, key: Key, state: DuneSeaState, action: Array
    ) -> tuple[DuneSeaState, Timestep]:
        thirsty = state.agent.energy > 0
        state, timestep = super().step(key, state, action)
        cells = state.cells
        radius = self.view_size // 2
        offsets = jnp.arange(-radius, radius + 1)
        dy, dx = jnp.meshgrid(offsets, offsets, indexing="ij")
        distance = jnp.maximum(jnp.abs(dy), jnp.abs(dx)).reshape(-1)
        padded_wall = jnp.pad(cells.wall, radius, constant_values=1.0)
        padded_water = jnp.pad((cells.oasis > 0).astype(jnp.float32), radius)

        def nearest_water(position):
            visible = self.visibility(padded_wall, position)
            window = jax.lax.dynamic_slice(padded_water, (position[0], position[1]), (self.view_size, self.view_size))
            reach = jnp.where((visible & (window > 0)).reshape(-1), distance, jnp.inf)
            index = jnp.argmin(reach)
            cell = position + jnp.stack([dy.reshape(-1)[index], dx.reshape(-1)[index]])
            return cell, jnp.isfinite(reach[index])

        cell, in_view = jax.vmap(nearest_water)(state.agent.position)
        y = jnp.clip(cell[:, 0], 0, self.height - 1)
        x = jnp.clip(cell[:, 1], 0, self.width - 1)

        drinking = thirsty & in_view
        wanted = jnp.where(drinking, jnp.minimum(self.drink_rate, MAX_ENERGY - state.agent.energy), 0.0)
        demand = jnp.zeros_like(cells.oasis).at[y, x].add(wanted)
        share = jnp.minimum(1.0, cells.oasis / jnp.maximum(demand, 1e-9))
        drunk = wanted * share[y, x]
        energy = jnp.clip(state.agent.energy + drunk, 0.0, MAX_ENERGY)
        oasis = cells.oasis - jnp.zeros_like(cells.oasis).at[y, x].add(drunk)
        oasis = jnp.where(oasis > self.evaporation, oasis - self.evaporation, 0.0)

        buried = (cells.mirage > 0) & (cells.mirage <= state.time)
        uncovered = (cells.mirage < 0) & (-cells.mirage == state.time) & (cells.wall == 0)
        oasis = jnp.where(uncovered, self.reserve, jnp.where(buried, 0.0, oasis))

        state = state.replace(cells=cells.replace(oasis=oasis), agent=state.agent.replace(energy=energy))
        timestep = timestep.replace(
            obs=self.observe(state),
            reward=timestep.reward + drunk,
            terminated=energy <= 0.0,
        )
        return state, timestep
