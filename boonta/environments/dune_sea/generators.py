from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from boonta.utils import Array, Key

from .state import (ENERGY_GAIN, MAX_ENERGY, MAX_PLANT_ENERGY, SEED_ENERGY,
                    Agent, Cell)

CAVE_FILL = 0.46
CAVE_ROUNDS = 4
CAVE_ENERGY_DENSITY = 0.05
BUSH_FILL = 0.42
BUSH_ROUNDS = 3
MAZE_BRAID = 0.1
MAZE_PRIZE_DISTANCE = 12

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


def cave_generator(
    key: Key, height: int, width: int, num_agents: int, fill: float = CAVE_FILL
) -> tuple[Cell, Agent]:
    cave_key, agent_key, energy_key, stage_key, bush_key = jax.random.split(key, 5)
    noise = jax.random.bernoulli(cave_key, fill, (height, width))
    wall = smooth_cave(noise, CAVE_ROUNDS)
    bush = smooth_cave(jax.random.bernoulli(bush_key, BUSH_FILL, (height, width)), BUSH_ROUNDS) & ~wall
    agent = place_agents(agent_key, ~wall, num_agents)
    energy = jax.random.bernoulli(energy_key, CAVE_ENERGY_DENSITY, (height, width))
    energy = energy & ~wall & ~occupied_cells(agent, height, width)
    stage = jax.random.uniform(
        stage_key, (height, width), minval=SEED_ENERGY, maxval=MAX_PLANT_ENERGY
    )
    cells = Cell(
        wall=wall.astype(jnp.float32),
        energy=jnp.where(energy, stage, 0.0),
        bush=bush.astype(jnp.float32),
    )
    return cells, agent


def _neighbours(rows: int, columns: int) -> Array:
    ids = np.arange(rows * columns).reshape(rows, columns)
    table = -np.ones((rows * columns, 4), np.int32)
    table[ids[1:].ravel(), 0] = ids[:-1].ravel()
    table[ids[:-1].ravel(), 1] = ids[1:].ravel()
    table[ids[:, 1:].ravel(), 2] = ids[:, :-1].ravel()
    table[ids[:, :-1].ravel(), 3] = ids[:, 1:].ravel()
    return jnp.asarray(table)


def _partitions(rows: int, columns: int) -> Array:
    across = np.stack(
        [
            np.repeat(2 * np.arange(rows) + 1, columns - 1),
            np.tile(2 * np.arange(columns - 1) + 2, rows),
        ],
        axis=-1,
    )
    down = np.stack(
        [
            np.repeat(2 * np.arange(rows - 1) + 2, columns),
            np.tile(2 * np.arange(columns) + 1, rows - 1),
        ],
        axis=-1,
    )
    return jnp.asarray(np.concatenate([across, down]), jnp.int32)


def _wilson(key: Key, rows: int, columns: int) -> Array:
    table = _neighbours(rows, columns)
    count = rows * columns
    root_key, order_key, walk_key = jax.random.split(key, 3)
    root = jax.random.randint(root_key, (), 0, count)
    order = jax.random.permutation(order_key, count)

    def grow(carry, start):
        in_tree, successor, key = carry

        def hop(state):
            node, successor, key = state
            key, choice_key = jax.random.split(key)
            options = table[node]
            choice = options[
                jax.random.categorical(
                    choice_key, jnp.where(options >= 0, 0.0, -jnp.inf)
                )
            ]
            return choice, successor.at[node].set(choice), key

        _, successor, key = jax.lax.while_loop(
            lambda state: ~in_tree[state[0]], hop, (start, successor, key)
        )
        _, in_tree = jax.lax.while_loop(
            lambda state: ~state[1][state[0]],
            lambda state: (successor[state[0]], state[1].at[state[0]].set(True)),
            (start, in_tree),
        )
        return (in_tree, successor, key), None

    (_, successor, _), _ = jax.lax.scan(
        grow,
        (jnp.zeros(count, bool).at[root].set(True), jnp.arange(count), walk_key),
        order,
    )
    return successor


def _carve(key: Key, height: int, width: int, braid: float) -> Array:
    rows, columns = (height - 1) // 2, (width - 1) // 2
    tree_key, braid_key = jax.random.split(key)
    successor = _wilson(tree_key, rows, columns)
    node = jnp.arange(rows * columns)
    cell = lambda index: 2 * jnp.stack([index // columns, index % columns], -1) + 1
    between = (cell(node) + cell(successor)) // 2
    wall = jnp.ones((height, width), jnp.float32).at[1::2, 1::2].set(0.0)
    wall = wall.at[between[:, 0], between[:, 1]].set(0.0)
    at = _partitions(rows, columns)
    knocked = jax.random.bernoulli(braid_key, braid, (at.shape[0],))
    return wall.at[at[:, 0], at[:, 1]].multiply(1.0 - knocked)


def _distances(free: Array, sources: Array, rounds: int) -> Array:
    unreached = free.size

    def relax(distance, _):
        padded = jnp.pad(distance, 1, constant_values=unreached)
        nearest = jnp.minimum(
            jnp.minimum(padded[:-2, 1:-1], padded[2:, 1:-1]),
            jnp.minimum(padded[1:-1, :-2], padded[1:-1, 2:]),
        )
        return jnp.where(free, jnp.minimum(distance, nearest + 1), unreached), None

    distance, _ = jax.lax.scan(
        relax, jnp.where(sources, 0, unreached), None, length=rounds
    )
    return distance


def maze_generator(
    key: Key,
    height: int,
    width: int,
    num_agents: int,
    braid: float = MAZE_BRAID,
    prize_distance: int = MAZE_PRIZE_DISTANCE,
) -> tuple[Cell, Agent]:
    maze_key, agent_key, prize_key = jax.random.split(key, 3)
    wall = _carve(maze_key, height, width, braid)
    agent = place_agents(agent_key, wall == 0, num_agents)
    rounds = (height - 1) * (width - 1) // 2
    distance = _distances(wall == 0, occupied_cells(agent, height, width), rounds)
    distance = jnp.where(wall == 0, distance, -1)
    eligible = distance >= jnp.minimum(prize_distance, distance.max())
    prize = jax.random.choice(
        prize_key,
        height * width,
        p=eligible.reshape(-1) / eligible.sum(),
    )
    energy = (
        jnp.zeros(height * width, jnp.float32).at[prize].set(MAX_PLANT_ENERGY)
    ).reshape(height, width)
    cells = Cell(wall=wall, energy=energy, bush=jnp.zeros_like(wall))
    return cells, agent
