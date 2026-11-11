from collections import deque

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boonta.environments import dune_sea
from boonta.environments.dune_sea import Maze
from boonta.environments.dune_sea.maze import MOVES, cell


def steps_to_goal(env):
    successor = np.asarray(env.successor)
    steps = np.full(env.num_cells, -1)
    steps[env.goal] = 0
    frontier = deque([env.goal])
    while frontier:
        landing = frontier.popleft()
        for origin in np.flatnonzero((successor == landing).any(axis=-1)):
            if steps[origin] < 0:
                steps[origin] = steps[landing] + 1
                frontier.append(origin)
    return steps


def walk(env, policy, num_steps):
    state, _ = env.init(jax.random.key(0))
    timesteps = []
    for _ in range(num_steps):
        state, timestep = env.step(jax.random.key(0), state, jnp.int32(policy(int(state.position))))
        timesteps.append(timestep)
        if bool(timestep.terminated | timestep.truncated):
            break
    return state, timesteps


@pytest.mark.parametrize("size", [2, 5, 8])
@pytest.mark.parametrize("seed", [0, 1])
def test_every_cell_leads_to_the_goal_along_exactly_one_route(size, seed):
    env = Maze(size=size, seed=seed)
    successor = np.asarray(env.successor)
    assert env.num_cells == 2 * size * size - 1
    assert (steps_to_goal(env) >= 0).all()
    passages = np.sum(successor != np.arange(env.num_cells)[:, None]) // 2
    assert passages == env.num_cells - 1


def test_the_start_and_the_goal_sit_in_opposite_corners():
    env = Maze(size=5)
    assert (int(env.rows[env.start]), int(env.columns[env.start])) == (1, 1)
    assert (int(env.rows[env.goal]), int(env.columns[env.goal])) == (9, 9)


def test_the_seed_picks_the_maze():
    same, other = Maze(size=6, seed=3), Maze(size=6, seed=4)
    np.testing.assert_array_equal(Maze(size=6, seed=3).cell_grid, same.cell_grid)
    assert not np.array_equal(same.cell_grid, other.cell_grid)


@pytest.mark.parametrize("size, seed", [(4, 0), (8, 0), (8, 5)])
def test_the_reward_is_minus_the_straight_line_distance_to_the_goal_over_the_farthest(size, seed):
    env = Maze(size=size, seed=seed)
    rows, columns = np.asarray(env.rows), np.asarray(env.columns)
    distances = np.sqrt((rows - rows[env.goal]) ** 2 + (columns - columns[env.goal]) ** 2)
    np.testing.assert_allclose(env.reward, -distances / distances.max(), rtol=1e-6)
    assert float(env.reward[env.goal]) == 0.0
    assert float(env.reward[env.start]) == -1.0
    assert float(jnp.max(jnp.where(jnp.arange(env.num_cells) == env.goal, -1.0, env.reward))) < 0.0


def test_a_step_pays_for_the_cell_it_lands_on_and_a_wall_holds_the_agent():
    env = Maze(size=5)
    successor = np.asarray(env.successor)
    state, _ = env.init(jax.random.key(0))
    blocked = int(np.flatnonzero(successor[env.start] == env.start)[0])
    open_ = int(np.flatnonzero(successor[env.start] != env.start)[0])
    held, timestep = env.step(jax.random.key(0), state, jnp.int32(blocked))
    assert int(held.position) == env.start
    assert float(timestep.reward) == -1.0
    moved, timestep = env.step(jax.random.key(0), state, jnp.int32(open_))
    row, column = np.array([env.rows[env.start], env.columns[env.start]]) + MOVES[open_]
    assert (int(env.rows[moved.position]), int(env.columns[moved.position])) == (row, column)
    assert float(timestep.reward) == float(env.reward[moved.position])
    assert -1.0 < float(timestep.reward) < 0.0


def test_the_shortest_route_ends_the_episode_at_the_goal_for_free():
    env = Maze()
    steps = steps_to_goal(env)
    successor = np.asarray(env.successor)
    state, timesteps = walk(env, lambda position: steps[successor[position]].argmin(), env.horizon)
    assert int(state.position) == env.goal
    assert len(timesteps) == steps[env.start]
    assert bool(timesteps[-1].terminated) and not bool(timesteps[-1].truncated)
    assert float(timesteps[-1].reward) == 0.0
    assert not any(bool(timestep.terminated | timestep.truncated) for timestep in timesteps[:-1])


def test_chasing_the_reward_strands_the_agent_until_the_horizon_cuts_it_off():
    env = Maze(horizon=40)
    successor, reward = np.asarray(env.successor), np.asarray(env.reward)
    state, timesteps = walk(env, lambda position: reward[successor[position]].argmax(), 2 * env.horizon)
    assert int(state.position) != env.goal
    assert len(timesteps) == env.horizon == env.time_limit()
    assert bool(timesteps[-1].truncated) and not bool(timesteps[-1].terminated)
    assert not any(bool(timestep.truncated) for timestep in timesteps[:-1])


@pytest.mark.parametrize("seed", range(4))
def test_the_best_policy_still_takes_the_shortest_route(seed):
    env = Maze(seed=seed)
    successor, reward = np.asarray(env.successor), np.asarray(env.reward)
    steps = steps_to_goal(env)
    value = np.zeros(env.num_cells)
    for _ in range(env.horizon):
        value = (reward[successor] + value[successor]).max(axis=-1)
        value[env.goal] = 0.0
    position, route = env.start, 0.0
    for _ in range(steps[env.start]):
        position = successor[position, (reward[successor[position]] + value[successor[position]]).argmax()]
        route += reward[position]
    assert position == env.goal
    assert abs(value[env.start] - route) < 1e-4


def test_the_cell_is_the_position_and_place_moves_the_agent_there():
    env = dune_sea.make("maze", size=4)
    state, _ = env.init(jax.random.key(0))
    assert int(cell(env)(state)) == env.start
    state, _ = env.step(jax.random.key(0), state, jnp.int32(0))
    placed = env.place(state, env.goal - 1)
    assert int(cell(env)(placed)) == env.goal - 1
    assert int(placed.clock) == 0
    np.testing.assert_array_equal(env.observe(placed), jax.nn.one_hot(env.goal - 1, env.num_cells))


def test_the_maze_renders_the_agent_red_the_goal_green_and_the_walls_grey():
    env = Maze(size=5)
    state, _ = env.init(jax.random.key(0))
    tile = 8
    image = np.asarray(env.render(state, tile=tile))
    cell_grid = np.asarray(env.cell_grid)
    num_rows, num_columns = cell_grid.shape
    assert image.shape == (num_rows * tile, num_columns * tile, 3)
    assert image.dtype == np.uint8

    def centre(row, column, across=0.5):
        return image[int((row + 0.5) * tile), int((column + across) * tile)]

    np.testing.assert_array_equal(centre(1, 1, across=0.4), [255, 0, 0])
    np.testing.assert_array_equal(centre(num_rows - 2, num_columns - 2), [0, 255, 0])
    for row, column in np.argwhere(cell_grid < 0):
        np.testing.assert_array_equal(centre(row, column), [100, 100, 100])
    for row, column in np.argwhere((cell_grid >= 0) & (cell_grid != env.start) & (cell_grid != env.goal)):
        np.testing.assert_array_equal(centre(row, column), [0, 0, 0])
