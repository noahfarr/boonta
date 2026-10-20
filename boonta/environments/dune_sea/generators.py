from collections.abc import Callable

import jax
import jax.numpy as jnp

from boonta.utils import Array, Key

from .state import (ENERGY_GAIN, MAX_ENERGY, MAX_PLANT_ENERGY, SEED_ENERGY,
                    Agent, Cell)

WALL_DENSITY = 0.2
ENERGY_DENSITY = 0.1
CAVE_FILL = 0.46
CAVE_ROUNDS = 4
CAVE_ENERGY_DENSITY = 0.05

Generator = Callable[[Key, int, int, int], tuple[Cell, Agent]]


def place_agents(key: Key, free: Array, num_agents: int) -> Agent:
    position_key, direction_key = jax.random.split(key)
    height, width = free.shape
    weights = free.reshape(-1).astype(jnp.float32)
    cells = jax.random.choice(
        position_key,
        height * width,
        (num_agents,),
        replace=False,
        p=weights / weights.sum(),
    )
    return Agent(
        position=jnp.stack([cells // width, cells % width], axis=-1).astype(jnp.int32),
        direction=jax.random.randint(direction_key, (num_agents,), 0, 4),
        energy=jnp.full(num_agents, MAX_ENERGY, jnp.float32),
        broadcast=jnp.zeros(num_agents, bool),
        inventory=jnp.zeros(num_agents, jnp.int32),
    )


def occupied_cells(agent: Agent, height: int, width: int) -> Array:
    return (
        jnp.zeros((height, width), bool)
        .at[agent.position[:, 0], agent.position[:, 1]]
        .set(True)
    )


def neighbour_walls(wall: Array) -> Array:
    height, width = wall.shape
    padded = jnp.pad(wall.astype(jnp.int32), 1)
    return sum(
        padded[dy : dy + height, dx : dx + width]
        for dy in range(3)
        for dx in range(3)
        if (dy, dx) != (1, 1)
    )


def smooth_cave(wall: Array, rounds: int) -> Array:
    def smooth(wall, _):
        neighbours = neighbour_walls(wall)
        return (neighbours > 4) | (wall & (neighbours == 4)), None

    wall, _ = jax.lax.scan(smooth, wall, None, length=rounds)
    return wall


def uniform_generator(
    key: Key, height: int, width: int, num_agents: int
) -> tuple[Cell, Agent]:
    wall_key, agent_key, energy_key = jax.random.split(key, 3)
    goal = jnp.zeros((height, width), bool).at[-1, -1].set(True)
    wall = jax.random.bernoulli(wall_key, WALL_DENSITY, (height, width)) & ~goal
    agent = place_agents(agent_key, ~wall & ~goal, num_agents)
    energy = jax.random.bernoulli(energy_key, ENERGY_DENSITY, (height, width))
    energy = energy & ~wall & ~goal & ~occupied_cells(agent, height, width)
    cells = Cell(
        wall=wall.astype(jnp.float32),
        goal=goal.astype(jnp.float32),
        energy=energy.astype(jnp.float32) * ENERGY_GAIN,
    )
    return cells, agent


def cave_generator(
    key: Key, height: int, width: int, num_agents: int
) -> tuple[Cell, Agent]:
    cave_key, agent_key, energy_key, stage_key = jax.random.split(key, 4)
    noise = jax.random.bernoulli(cave_key, CAVE_FILL, (height, width))
    wall = smooth_cave(noise, CAVE_ROUNDS)
    agent = place_agents(agent_key, ~wall, num_agents)
    energy = jax.random.bernoulli(energy_key, CAVE_ENERGY_DENSITY, (height, width))
    energy = energy & ~wall & ~occupied_cells(agent, height, width)
    stage = jax.random.uniform(
        stage_key, (height, width), minval=SEED_ENERGY, maxval=MAX_PLANT_ENERGY
    )
    cells = Cell(
        wall=wall.astype(jnp.float32),
        goal=jnp.zeros((height, width), jnp.float32),
        energy=jnp.where(energy, stage, 0.0),
    )
    return cells, agent
