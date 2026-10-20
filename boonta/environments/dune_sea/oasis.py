import functools

import jax
import jax.numpy as jnp

from boonta.utils import Array, Key, Timestep

from .dune_sea import DIRECTIONS, INTERACT, DuneSea
from .generators import Generator, cave_generator, place_agents
from .state import MAX_ENERGY, MIRAGE_STEPS, DuneSeaState

OASIS_RESERVE = 0.8
DRINK_RATE = 0.05
MIRAGE_START = 200
MIRAGE_END = 600
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
        generator: Generator = cave_generator,
        num_oases: int = 2,
        num_mirages: int = 1,
        mirage_start: int = MIRAGE_START,
        mirage_end: int = MIRAGE_END,
        reserve: float = OASIS_RESERVE,
        drink_rate: float = DRINK_RATE,
        beacon_slots: int = 3,
        energy_cost: float = ENERGY_COST,
    ):
        base_generator = generator

        def generate(key, height, width, num_agents):
            cells, agent = base_generator(key, height, width, num_agents)
            oasis, ponds = cells.oasis, jnp.zeros_like(cells.oasis)
            for number, pond_key in enumerate(jax.random.split(jax.random.fold_in(key, 11), num_oases), start=1):
                grown = spawn_pond(pond_key, cells.wall, oasis, reserve)
                ponds = jnp.where((grown > 0) & (oasis == 0), float(number), ponds)
                oasis = grown
            taken, spare = oasis, cells.spare
            for number, pond_key in enumerate(jax.random.split(jax.random.fold_in(key, 13), num_mirages), start=1):
                pond = spawn_pond(pond_key, cells.wall, taken, 1.0)
                fresh = (pond > 0) & (taken == 0)
                spare = jnp.where(fresh, float(number), spare)
                taken = jnp.where(fresh, 1.0, taken)
            fades = jnp.sort(jax.random.randint(jax.random.fold_in(key, 17), (num_mirages,), mirage_start, mirage_end + 1))
            mirage = cells.mirage
            for number in range(1, num_mirages + 1):
                mirage = jnp.where(ponds == number, fades[number - 1].astype(jnp.float32), mirage)
            bush = jnp.where(dilate(oasis > 0, BUSH_CLEARANCE) > 0, 0.0, cells.bush)
            cells = cells.replace(energy=jnp.zeros_like(cells.energy), oasis=oasis, bush=bush, spare=spare, mirage=mirage)
            row, column = agent.position[:, 0], agent.position[:, 1]
            standing = oasis[row, column] > 0
            crowded = jnp.zeros_like(oasis).at[row, column].set(1.0)
            moved = place_agents(jax.random.fold_in(key, 19), (cells.wall == 0) & (oasis == 0) & (crowded == 0), num_agents)
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

    def step(
        self, key: Key, state: DuneSeaState, action: Array
    ) -> tuple[DuneSeaState, Timestep]:
        thirsty = state.agent.energy > 0
        state, timestep = super().step(key, state, action)
        cells = state.cells
        facing = state.agent.position + DIRECTIONS[state.agent.direction]
        inside = ((facing >= 0) & (facing < jnp.array([self.height, self.width]))).all(-1)
        y = jnp.clip(facing[:, 0], 0, self.height - 1)
        x = jnp.clip(facing[:, 1], 0, self.width - 1)

        drinking = thirsty & inside & (timestep.action == INTERACT)
        wanted = jnp.where(drinking, jnp.minimum(self.drink_rate, MAX_ENERGY - state.agent.energy), 0.0)
        demand = jnp.zeros_like(cells.oasis).at[y, x].add(wanted)
        share = jnp.minimum(1.0, cells.oasis / jnp.maximum(demand, 1e-9))
        drunk = wanted * share[y, x]
        energy = jnp.clip(state.agent.energy + drunk, 0.0, MAX_ENERGY)
        oasis = cells.oasis - jnp.zeros_like(cells.oasis).at[y, x].add(drunk)
        oasis = jnp.where(oasis < 1e-4, 0.0, oasis)

        fading = (cells.mirage > 0) & (cells.mirage <= state.time)
        expiring = (cells.mirage < 0) & (state.time + cells.mirage >= MIRAGE_STEPS)
        oasis = jnp.where(fading, 0.0, oasis)
        mirage = jnp.where(fading, -state.time.astype(jnp.float32), jnp.where(expiring, 0.0, cells.mirage))

        waiting = cells.spare > 0
        next_pond = jnp.where(waiting, cells.spare, jnp.inf).min()
        rising = waiting & (cells.spare == next_pond) & fading.any()
        oasis = jnp.where(rising & (cells.wall == 0) & (oasis == 0), self.reserve, oasis)
        spare = jnp.where(rising, -state.time.astype(jnp.float32), cells.spare)

        looks_wet = (oasis > 0) | (mirage < 0)
        bush = jnp.where(dilate(looks_wet, BUSH_CLEARANCE) > 0, 0.0, cells.bush)
        state = state.replace(
            cells=cells.replace(oasis=oasis, bush=bush, spare=spare, mirage=mirage),
            agent=state.agent.replace(energy=energy),
        )
        timestep = timestep.replace(
            obs=self.observe(state),
            reward=timestep.reward + drunk,
            terminated=energy <= 0.0,
        )
        return state, timestep
