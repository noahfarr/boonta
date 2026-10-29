from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boonta.environments.dune_sea import DeadReckoning
from boonta.environments.dune_sea.dead_reckoning import (COMPASS, FORWARD, LEFT, RIGHT,
                                                          STOP)


def decode(env, obs):
    width = 2 * env.radius + 1
    return np.asarray(obs)[: 2 * width].reshape(2, width).argmax(-1) - env.radius


def executed(env, obs):
    return np.asarray(obs)[2 * (2 * env.radius + 1) :]


def allocentric(offset):
    dx, dy = offset
    return [1] * max(dy, 0) + [3] * max(-dy, 0) + [2] * max(dx, 0) + [4] * max(-dx, 0) + [STOP]


def egocentric(offset):
    dx, dy = offset
    plan = [FORWARD] * max(dy, 0)
    if dx:
        plan += [RIGHT if dx > 0 else LEFT] + [FORWARD] * abs(dx)
        if dy < 0:
            plan += [RIGHT if dx > 0 else LEFT] + [FORWARD] * -dy
    elif dy < 0:
        plan += [RIGHT, RIGHT] + [FORWARD] * -dy
    return plan + [STOP]


def play(env, key, plan):
    state, timestep = env.init(key)
    actions = plan(decode(env, timestep.obs))
    total, steps = 0.0, 0
    for action in actions:
        key, step = jax.random.split(key)
        state, timestep = env.step(step, state, jnp.int32(action))
        total += float(timestep.reward)
        steps += 1
        if timestep.terminated or timestep.truncated:
            break
    return total, steps, timestep


@pytest.mark.parametrize("radius", [1, 3])
@pytest.mark.parametrize("ego", [False, True], ids=["allocentric", "egocentric"])
def test_integrating_your_own_moves_always_reaches_the_target(radius, ego):
    env = DeadReckoning(radius=radius, egocentric=ego)
    for seed in range(32):
        total, steps, timestep = play(env, jax.random.key(seed), egocentric if ego else allocentric)
        assert total == 1.0
        assert bool(timestep.terminated)
        assert steps <= env.time_limit()


def test_starts_and_targets_are_distinct_and_cover_every_pair_of_cells():
    env = DeadReckoning(radius=1)
    pairs = set()
    for seed in range(2000):
        state, _ = env.init(jax.random.key(seed))
        pairs.add((*np.asarray(state.position), *np.asarray(state.target)))
    assert all(pair[:2] != pair[2:] for pair in pairs)
    assert len(pairs) == 9 * 8


def test_only_the_first_observation_shows_the_offset_without_glimpses():
    env = DeadReckoning(radius=2)
    state, timestep = env.init(jax.random.key(0))
    assert float(timestep.obs.sum()) == 2.0
    assert not executed(env, timestep.obs).any()
    for seed in range(5):
        state, timestep = env.step(jax.random.key(seed), state, jnp.int32(1))
        assert not timestep.obs[: 2 * 5].any()
        np.testing.assert_array_equal(executed(env, timestep.obs), np.eye(5)[1])


def test_a_glimpse_shows_the_remaining_offset():
    env = DeadReckoning(radius=2, glimpse=1.0)
    state, _ = env.init(jax.random.key(3))
    state, timestep = env.step(jax.random.key(0), state, jnp.int32(2))
    np.testing.assert_array_equal(
        decode(env, timestep.obs), env.wrap(state.target - state.position)
    )


def test_stopping_anywhere_else_ends_the_episode_without_reward():
    env = DeadReckoning(radius=2)
    state, _ = env.init(jax.random.key(0))
    _, timestep = env.step(jax.random.key(0), state, jnp.int32(STOP))
    assert bool(timestep.terminated)
    assert float(timestep.reward) == 0.0


def test_leaving_one_side_enters_the_opposite_side():
    env = DeadReckoning(radius=2)
    state, _ = env.init(jax.random.key(0))
    state = state.replace(position=jnp.array([2, -2]))
    state, _ = env.step(jax.random.key(0), state, jnp.int32(2))
    np.testing.assert_array_equal(state.position, [-2, -2])
    state, _ = env.step(jax.random.key(0), state, jnp.int32(3))
    np.testing.assert_array_equal(state.position, [-2, 2])


def test_going_once_around_returns_to_the_start_and_wandering_runs_out_of_time():
    env = DeadReckoning(radius=1)
    state, _ = env.init(jax.random.key(0))
    start = np.asarray(state.position)
    for step in range(env.time_limit()):
        state, timestep = env.step(jax.random.key(0), state, jnp.int32(1))
        if step == 2:
            np.testing.assert_array_equal(state.position, start)
    assert bool(timestep.truncated) and not bool(timestep.terminated)


@pytest.mark.parametrize("ego", [False, True], ids=["allocentric", "egocentric"])
def test_a_slip_makes_a_random_move_which_the_agent_observes_but_never_reports_as_its_action(ego):
    env = DeadReckoning(radius=3, egocentric=ego, slip_probability=1.0)
    state, _ = env.init(jax.random.key(0))
    moves = set()
    for seed in range(64):
        before = state
        state, timestep = env.step(jax.random.key(seed), state, jnp.int32(1))
        assert int(timestep.action) == 1
        move = int(np.argmax(executed(env, timestep.obs)))
        assert move != STOP and move == int(state.executed)
        if not ego:
            np.testing.assert_array_equal(
                env.wrap(state.position - before.position), np.asarray(COMPASS[move - 1])
            )
        moves.add(move)
    assert moves == set(range(1, env.num_actions))


def test_stopping_never_slips():
    env = DeadReckoning(radius=2, slip_probability=1.0)
    for seed in range(16):
        state, _ = env.init(jax.random.key(seed))
        _, timestep = env.step(jax.random.key(seed), state, jnp.int32(STOP))
        assert bool(timestep.terminated)


def test_integrating_the_observed_moves_still_wins_when_moves_slip():
    env = DeadReckoning(radius=3, slip_probability=0.3)
    wins = 0
    for seed in range(32):
        key = jax.random.key(seed)
        state, timestep = env.init(key)
        remaining = decode(env, timestep.obs)
        for _ in range(env.time_limit()):
            if not remaining.any():
                action = STOP
            else:
                action = allocentric(remaining)[0]
            key, step = jax.random.split(key)
            state, timestep = env.step(step, state, jnp.int32(action))
            if timestep.terminated:
                break
            move = int(np.argmax(executed(env, timestep.obs)))
            remaining = env.wrap(remaining - np.asarray(COMPASS[move - 1]))
        wins += float(timestep.reward)
        assert float(timestep.reward) == float(timestep.terminated)
    assert wins >= 28


def test_a_random_policy_rarely_wins():
    env = DeadReckoning(radius=3)
    wins = 0
    for seed in range(300):
        key = jax.random.key(seed)
        state, _ = env.init(key)
        for _ in range(env.time_limit()):
            key, choice, step = jax.random.split(key, 3)
            action = jax.random.randint(choice, (), 0, 5)
            state, timestep = env.step(step, state, action)
            wins += float(timestep.reward)
            if timestep.terminated or timestep.truncated:
                break
    assert wins < 30


@pytest.mark.parametrize(
    "heading, inside, outside",
    [
        (0, (0.3, 0.75), (0.3, 0.25)),
        (1, (0.25, 0.3), (0.75, 0.3)),
        (2, (0.3, 0.25), (0.3, 0.75)),
        (3, (0.75, 0.3), (0.25, 0.3)),
    ],
    ids=["north", "east", "south", "west"],
)
def test_the_render_draws_a_minigrid_with_the_goal_and_a_pointing_agent(
    heading, inside, outside
):
    env = DeadReckoning(radius=2, egocentric=True)
    state, _ = env.init(jax.random.key(1))
    state = state.replace(
        target=jnp.array([-2, 2]), position=jnp.array([1, -1]), heading=jnp.int32(heading)
    )
    cell = 64
    image = np.asarray(jax.jit(partial(env.render, cell=cell))(state))
    assert image.shape == (5 * cell, 5 * cell, 3) and image.dtype == np.uint8

    def pixel(column, row, u=0.5, v=0.5):
        return image[int((row + v) * cell), int((column + u) * cell)]

    np.testing.assert_array_equal(pixel(0, 0), [0, 255, 0])
    np.testing.assert_array_equal(pixel(1, 1, 0.01, 0.5), [100, 100, 100])
    np.testing.assert_array_equal(pixel(2, 2), [0, 0, 0])
    np.testing.assert_array_equal(pixel(3, 3, *inside), [255, 0, 0])
    np.testing.assert_array_equal(pixel(3, 3, *outside), [0, 0, 0])
