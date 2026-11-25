import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boonta import environments
from boonta.environments.dune_sea.maze import (FORWARD, GREEN, GREY, LEFT, LIMIT_QUANTILE, NOOP, TILE,
                                               TIME_PENALTY, TURN_RIGHT, Level, Maze,
                                               agent_tile, distances, kruskal_generator)
from boonta.environments.wrappers import UED, SameStepAutoReset

KEY = jax.random.key(0)


def open_passages(wall):
    wall = np.asarray(wall)
    across = wall[1::2, 2:-1:2] == 0
    down = wall[2:-1:2, 1::2] == 0
    return int(across.sum() + down.sum())


def corridor(length=4):
    wall = jnp.ones((3, 2 * length + 1)).at[1, 1:-1].set(0.0)
    return Level(wall=wall, start=jnp.array([1, 1]), direction=jnp.int32(0), goal=jnp.array([1, 2 * length - 1]))


@pytest.mark.parametrize("height, width", [(3, 3), (6, 9), (13, 13)])
def test_kruskal_carves_a_perfect_maze_that_reaches_every_cell(height, width):
    for seed in range(4):
        level = jax.jit(kruskal_generator(height, width))(jax.random.key(seed))
        distance = np.asarray(distances(level.wall, level.start))[1::2, 1::2]

        assert level.wall.shape == (2 * height + 1, 2 * width + 1)
        assert open_passages(level.wall) == height * width - 1
        assert (distance < level.wall.size).all()


def test_the_border_is_always_rock():
    wall = np.asarray(kruskal_generator(7, 7, braid=1.0)(KEY).wall)
    assert wall[0].all() and wall[-1].all() and wall[:, 0].all() and wall[:, -1].all()


def test_braiding_opens_loops():
    perfect = kruskal_generator(9, 9)(KEY)
    braided = kruskal_generator(9, 9, braid=0.3)(KEY)
    assert open_passages(braided.wall) > open_passages(perfect.wall) == 80


def test_the_goal_is_the_cell_farthest_from_the_start():
    for seed in range(4):
        level = kruskal_generator(8, 8)(jax.random.key(seed))
        distance = np.asarray(distances(level.wall, level.start))
        assert distance[tuple(np.asarray(level.goal))] == distance[1::2, 1::2].max() > 0


def test_every_seed_carves_a_different_maze():
    walls = {np.asarray(kruskal_generator(6, 6)(jax.random.key(seed)).wall).tobytes() for seed in range(8)}
    assert len(walls) == 8


def test_the_default_time_limit_is_the_random_walk_bound_of_a_hard_maze():
    env = Maze(height=4, width=4)
    levels = jax.vmap(env.generator)(jax.random.split(jax.random.key(0), 256))
    paths = jax.vmap(lambda wall, start, goal: distances(wall, start)[goal[0], goal[1]])(levels.wall, levels.start, levels.goal)
    bounds = 7 * (np.asarray(levels.wall) == 0).sum((1, 2)) * np.asarray(paths)

    assert env.time_limit() == int(np.ceil(np.quantile(bounds, LIMIT_QUANTILE)))
    assert Maze(height=4, width=4, time_limit=50).time_limit() == 50
    assert Maze(height=4, width=4, braid=0.3).time_limit() != env.time_limit()


def test_a_random_policy_usually_reaches_the_goal_within_the_default_limit():
    env = Maze(height=3, width=3)
    count = 512
    states, _ = jax.vmap(env.init)(jax.random.split(KEY, count))

    def body(carry, key):
        states, reached = carry
        actions = jax.random.randint(key, (count,), 0, 7)
        states, timestep = jax.vmap(env.step)(jax.random.split(key, count), states, actions)
        return (states, reached | timestep.terminated), None

    (_, reached), _ = jax.jit(lambda states: jax.lax.scan(
        body, (states, jnp.zeros(count, bool)), jax.random.split(KEY, env.time_limit())
    ))(states)
    assert float(reached.mean()) > 0.8


def test_walls_stop_the_agent():
    env = Maze()
    state = env.start(corridor())

    state, _ = env.step(KEY, state, jnp.int32(LEFT))
    assert state.position.tolist() == [1, 1]
    state, _ = env.step(KEY, state, jnp.int32(FORWARD))
    assert state.position.tolist() == [1, 2]


def test_turning_changes_where_forward_goes():
    env = Maze()
    state = env.start(corridor())

    state, _ = env.step(KEY, state, jnp.int32(TURN_RIGHT))
    state, _ = env.step(KEY, state, jnp.int32(FORWARD))
    assert state.position.tolist() == [1, 1] and int(state.direction) == 1


def test_reaching_the_goal_ends_the_episode_with_a_reward_that_shrinks_with_time():
    env = Maze(time_limit=100)
    state = env.start(corridor(length=3))
    rewards = []
    for _ in range(4):
        state, timestep = env.step(KEY, state, jnp.int32(FORWARD))
        rewards.append(float(timestep.reward))

    assert bool(timestep.terminated) and not bool(timestep.truncated)
    assert rewards == pytest.approx([0.0, 0.0, 0.0, 1.0 - TIME_PENALTY * 4 / 100])


def test_the_episode_is_cut_at_the_time_limit():
    env = Maze(time_limit=3)
    state = env.start(corridor())
    for _ in range(3):
        state, timestep = env.step(KEY, state, jnp.int32(NOOP))

    assert bool(timestep.truncated) and not bool(timestep.terminated)


def test_the_view_shows_rock_and_the_goal_but_not_what_rock_hides():
    env = Maze(view_size=5)
    state = env.start(corridor(length=2))
    image = np.asarray(env.observe(state)["image"])

    assert image[2, 2, 0] == 0 and image[2, 2, 2] == 1
    assert image[2, 1, 0] == 1
    assert image[0, 2, 1] == 1
    assert image[2, 4, 2] == 0


def test_walls_stop_vision_but_are_seen_themselves():
    wall = jnp.zeros((7, 7)).at[2].set(1.0).at[4].set(1.0)
    level = Level(wall=wall, start=jnp.array([3, 3]), direction=jnp.int32(0), goal=jnp.array([3, 6]))
    env = Maze(view_size=5)
    seen = np.asarray(env.field_of_view(env.start(level)))

    assert seen[2:5, 1:6].all()
    assert not seen[1].any() and not seen[5].any()
    assert int(seen.sum()) == 15


def test_every_wall_touching_visible_floor_is_seen():
    for seed in range(8):
        env = Maze(view_size=5)
        state, _ = env.init(jax.random.key(seed))
        wall = np.asarray(state.level.wall) > 0
        seen = np.asarray(env.field_of_view(state))
        floor = np.pad(seen & ~wall, 1)
        touching = np.zeros_like(seen)
        for row in range(3):
            for column in range(3):
                touching |= floor[row : row + wall.shape[0], column : column + wall.shape[1]]
        window = np.zeros_like(seen)
        (r, c) = np.asarray(state.position)
        window[max(r - 2, 0) : r + 3, max(c - 2, 0) : c + 3] = True
        assert not (touching & wall & window & ~seen).any()


@pytest.mark.parametrize("direction", range(4))
def test_the_frame_highlights_exactly_the_cells_the_agent_sees(direction):
    env = Maze(view_size=5)
    for seed in range(4):
        state, _ = env.init(jax.random.key(seed))
        state = state.replace(direction=jnp.int32(direction))
        visible = np.asarray(env.observe(state)["image"][..., 2]) > 0
        cells = np.asarray(state.position + env.offsets[(direction + 1) % 4])
        rows, columns = np.asarray(state.level.wall).shape
        inside = (cells[..., 0] >= 0) & (cells[..., 0] < rows) & (cells[..., 1] >= 0) & (cells[..., 1] < columns)
        observed = np.zeros((rows, columns), bool)
        observed[cells[visible & inside, 0], cells[visible & inside, 1]] = True
        np.testing.assert_array_equal(observed, np.asarray(env.field_of_view(state)))


def test_an_update_with_a_level_starts_it_and_anything_else_is_ignored():
    env = Maze()
    state, _ = env.init(KEY)
    level = kruskal_generator(13, 13)(jax.random.key(7))

    assert env.update(state, KEY, setting=1.0) is state
    started = env.update(state, KEY, theta=level)
    assert (started.level.wall == level.wall).all()
    assert started.position.tolist() == level.start.tolist() and int(started.time) == 0


def test_a_level_reaches_the_game_through_ued():
    env = UED(SameStepAutoReset(Maze(height=5, width=5), 4))
    state, _ = env.init(KEY)
    levels = jax.vmap(kruskal_generator(5, 5))(jax.random.split(KEY, 3))
    state = env.update(state, KEY, theta=levels, weights=jnp.zeros(3))
    state = env.update(state, KEY, assign=jnp.array([0, 1, 2, 0]))
    idle = jnp.zeros(4, jnp.int32)
    state, _ = env.step(KEY, state, idle)
    state, timestep = env.step(KEY, state, idle)

    np.testing.assert_array_equal(timestep.info["theta"], [0, 1, 2, 0])
    np.testing.assert_array_equal(state.env_state.level.wall, np.asarray(levels.wall)[[0, 1, 2, 0]])


def test_make_builds_a_maze():
    assert isinstance(environments.make("dune_sea", "maze", kwargs={"height": 5, "width": 5}), Maze)


def test_a_frame_draws_walls_goal_and_agent_in_minigrid_colours():
    env = Maze(view_size=3)
    state = env.start(corridor())
    frame = env.render(state)

    def tile(row, column):
        return frame[row * TILE : (row + 1) * TILE, column * TILE : (column + 1) * TILE].reshape(-1, 3)

    assert frame.shape == (3 * TILE, 9 * TILE, 3) and frame.dtype == np.uint8
    assert (tile(0, 8) == GREY).all()
    assert (tile(1, 7) == GREEN).all(-1).mean() > 0.9
    agent = tile(1, 1).astype(int)
    assert ((agent[:, 0] - agent[:, 1]) > 100).any()
    moved, _ = env.step(KEY, state, jnp.int32(FORWARD))
    assert (frame != env.render(moved)).any()


def test_the_cells_in_view_are_lightened():
    env = Maze(view_size=3)
    frame = env.render(env.start(corridor())).astype(int)
    seen, unseen = frame[TILE + TILE // 2, 2 * TILE + TILE // 2], frame[TILE + TILE // 2, 5 * TILE + TILE // 2]
    assert (unseen == 0).all() and (seen > 0).all()


@pytest.mark.parametrize(
    "direction, tip, behind",
    [(0, (16, 24), (16, 1)), (1, (24, 15), (1, 15)), (2, (15, 7), (15, 30)), (3, (7, 16), (30, 16))],
)
def test_the_agent_triangle_points_where_the_agent_faces(direction, tip, behind):
    coverage = agent_tile(32, direction)
    assert coverage[tip] > 0.5 and coverage[behind] == 0
