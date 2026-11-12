from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boonta.environments import dune_sea
from boonta.environments.dune_sea import Chain, CombinationLock
from boonta.environments.dune_sea.chain import DIGITS, Scorecard, cell, shackle_rows


def play(env, num_episodes, policy):
    def episode(key):
        state, _ = env.init(key)

        def body(carry, key):
            state, ended, deepest, payout = carry
            policy_key, step_key = jax.random.split(key)
            deepest = jnp.where(ended, deepest, state.position)
            state, timestep = env.step(step_key, state, policy(policy_key, state))
            payout = payout + jnp.where(ended, 0.0, timestep.reward)
            return (state, ended | timestep.terminated | timestep.truncated, deepest, payout), None

        initial = (state, jnp.bool_(False), jnp.int32(0), jnp.float32(0.0))
        (_, ended, deepest, payout), _ = jax.lax.scan(body, initial, jax.random.split(key, env.time_limit()))
        return ended, deepest, payout

    return jax.jit(jax.vmap(episode))(jax.random.split(jax.random.key(0), num_episodes))


def informed(env):
    return lambda key, state: env.secrets[state.position]


@pytest.mark.parametrize("length, num_actions", [(1, 2), (6, 4), (12, 3)])
def test_without_hazard_every_cell_pays_its_own_secret_and_nothing_else(length, num_actions):
    env = Chain(length=length, num_actions=num_actions, hazard=0.0)
    secrets = np.asarray(env.secrets)
    for offset in range(num_actions):
        state, _ = env.init(jax.random.key(0))
        rewards, ends = [], []
        for position in range(length):
            action = jnp.int32((secrets[position] + offset * (position % 2)) % num_actions)
            state, timestep = env.step(jax.random.key(position), state, action)
            rewards.append(float(timestep.reward))
            ends.append(bool(timestep.terminated))
            assert not bool(timestep.truncated)
        expected = [1.0 if offset == 0 or position % 2 == 0 else 0.0 for position in range(length)]
        assert rewards == expected
        assert ends == [False] * (length - 1) + [True]
    assert env.time_limit() == length


def test_the_hazard_decides_how_deep_an_episode_gets_whatever_the_agent_does():
    env = Chain(length=10, num_actions=4, hazard=0.3)
    for policy in (informed(env), lambda key, state: jax.random.randint(key, (), 0, 4)):
        ended, deepest, _ = play(env, 40_000, policy)
        assert bool(ended.all())
        reached = np.array([np.mean(np.asarray(deepest) >= position) for position in range(env.num_cells)])
        np.testing.assert_allclose(reached, env.reach, atol=0.01)
    np.testing.assert_allclose(env.reach, 0.7 ** np.arange(10), rtol=1e-6)


def test_the_seed_picks_the_secrets():
    same, other = Chain(length=64, seed=3), Chain(length=64, seed=4)
    np.testing.assert_array_equal(Chain(length=64, seed=3).secrets, same.secrets)
    assert not np.array_equal(same.secrets, other.secrets)
    assert set(np.asarray(same.secrets).tolist()) == set(range(same.num_actions))


def test_a_restart_answers_the_cell_it_is_placed_on_at_once():
    env = dune_sea.make("chain", length=10, hazard=0.9)
    state, _ = env.init(jax.random.key(0))
    assert int(cell(env)(state)) == env.start == 0
    for position in range(env.num_cells):
        placed = env.place(state, position)
        assert int(cell(env)(placed)) == position
        np.testing.assert_array_equal(env.observe(placed), jax.nn.one_hot(position, env.num_cells))
        _, right = env.step(jax.random.key(0), placed, env.secrets[position])
        _, wrong = env.step(jax.random.key(0), placed, (env.secrets[position] + 1) % env.num_actions)
        assert (float(right.reward), float(wrong.reward)) == (1.0, 0.0)


def test_the_tables_describe_the_game_the_agent_plays():
    env = Chain(length=8, num_actions=3, hazard=0.25)
    policy = jax.nn.softmax(jax.random.normal(jax.random.key(0), (env.num_cells, 3)))
    correct = np.asarray(policy)[np.arange(8), np.asarray(env.secrets)]
    np.testing.assert_allclose(env.correct(policy), correct, rtol=1e-6)
    np.testing.assert_allclose(env.expected_return(policy), np.sum(0.75 ** np.arange(8) * correct), rtol=1e-6)

    def sampled(key, state):
        return jax.random.categorical(key, jnp.log(policy[state.position]))

    _, _, payout = play(env, 40_000, sampled)
    assert abs(float(np.mean(payout)) - float(env.expected_return(policy))) < 0.02
    _, _, payout = play(env, 40_000, informed(env))
    assert abs(float(np.mean(payout)) - float(env.reach.sum())) < 0.03

    advantages = np.asarray(env.advantages(policy))
    np.testing.assert_allclose(advantages, np.eye(3)[np.asarray(env.secrets)] - correct[:, None], atol=1e-6)
    np.testing.assert_allclose(np.sum(policy * advantages, axis=-1), 0.0, atol=1e-6)

    transitions = np.asarray(env.transitions(policy))
    expected = 0.75 * np.eye(8, k=1)
    expected[:, env.start] += [0.25] * 7 + [1.0]
    np.testing.assert_allclose(transitions, expected, rtol=1e-6)
    np.testing.assert_allclose(transitions.sum(axis=-1), 1.0, rtol=1e-6)


def test_a_lock_ends_the_episode_on_the_first_wrong_answer():
    env = dune_sea.make("combination_lock", length=6, num_actions=3)
    assert isinstance(env, CombinationLock) and env.hazard == 0.0
    state, _ = env.init(jax.random.key(0))
    for position in range(3):
        state, timestep = env.step(jax.random.key(position), state, env.secrets[position])
        assert (float(timestep.reward), bool(timestep.terminated)) == (1.0, False)
    _, timestep = env.step(jax.random.key(9), state, (env.secrets[3] + 1) % env.num_actions)
    assert (float(timestep.reward), bool(timestep.terminated)) == (0.0, True)


def test_behind_a_lock_how_deep_an_episode_gets_depends_on_the_policy():
    env = CombinationLock(length=8, num_actions=3, hazard=0.1)
    policy = jax.nn.softmax(2.0 * jax.random.normal(jax.random.key(1), (env.num_cells, 3)))

    def sampled(key, state):
        return jax.random.categorical(key, jnp.log(policy[state.position]))

    ended, deepest, payout = play(env, 60_000, sampled)
    assert bool(ended.all())
    reached = np.array([np.mean(np.asarray(deepest) >= position) for position in range(env.num_cells)])
    np.testing.assert_allclose(reached, env.reached(policy), atol=0.01)
    assert abs(float(np.mean(payout)) - float(env.expected_return(policy))) < 0.03
    assert float(env.reached(jax.nn.one_hot(env.secrets, 3))[-1]) == pytest.approx(0.9**7, rel=1e-5)
    transitions = np.asarray(env.transitions(policy))
    onward = 0.9 * np.asarray(env.correct(policy))[:-1]
    np.testing.assert_allclose(np.diag(transitions, k=1), onward, rtol=1e-5)
    np.testing.assert_allclose(transitions.sum(axis=-1), 1.0, rtol=1e-5)


@pytest.mark.parametrize("game", [Chain, CombinationLock])
def test_the_advantages_match_returns_sampled_from_every_cell_and_action(game):
    env = game(length=5, num_actions=3, hazard=0.2)
    policy = jax.nn.softmax(jax.random.normal(jax.random.key(2), (env.num_cells, 3)))
    advantages = np.asarray(env.advantages(policy))

    def returns(position, first, key):
        def body(carry, key):
            state, ended, total, step = carry
            policy_key, step_key = jax.random.split(key)
            action = jnp.where(step == 0, first, jax.random.categorical(policy_key, jnp.log(policy[state.position])))
            state, timestep = env.step(step_key, state, action)
            return (state, ended | timestep.terminated, total + jnp.where(ended, 0.0, timestep.reward), step + 1), None

        state, _ = env.init(key)
        initial = (env.place(state, position), jnp.bool_(False), jnp.float32(0.0), jnp.int32(0))
        (_, _, total, _), _ = jax.lax.scan(body, initial, jax.random.split(key, env.num_cells))
        return total

    sample = jax.jit(jax.vmap(returns, in_axes=(None, None, 0)))
    for position in range(env.num_cells):
        q = np.array([float(np.mean(sample(position, action, jax.random.split(jax.random.key(position * 3 + action), 40_000))))
                      for action in range(3)])
        np.testing.assert_allclose(advantages[position], q - np.sum(np.asarray(policy[position]) * q), atol=0.03)


class Oracle:
    def __init__(self, env, confidence, known):
        table = np.zeros((env.num_cells, env.num_actions), np.float32)
        table[np.arange(known), np.asarray(env.secrets)[:known]] = confidence
        self.table = jnp.asarray(table)

    def step(self, state, carry, key, timestep, temperature):
        return state, carry, None, {"logits": timestep.obs @ self.table}


@pytest.mark.parametrize("known", [0, 3, 8])
def test_the_scorecard_tells_the_return_from_how_many_cells_are_answered(known):
    env = Chain(length=8, num_actions=4, hazard=0.5)
    state = type("State", (), {"algorithm_state": None})()
    oracle = Oracle(env, 50.0, known)
    data = Scorecard().craft(jax.random.key(0), SimpleNamespace(algorithm=oracle, environment=env), state, {}).data
    reach = 0.5 ** np.arange(8)
    correct = np.where(np.arange(8) < known, 1.0, 0.25)
    assert abs(float(data["chain/best_return"]) - reach.sum()) < 1e-5
    assert abs(float(data["chain/return"]) - np.sum(reach * correct)) < 1e-5
    assert abs(float(data["chain/correct"]) - correct.mean()) < 1e-5
    np.testing.assert_allclose([data[f"chain/correct_{index}"] for index in range(8)], correct, atol=1e-5)


@pytest.mark.parametrize("length, body", [(1, (1, 1)), (6, (2, 3)), (7, (3, 3)), (64, (8, 8))])
def test_the_wheels_sit_on_a_near_square_grid_in_reading_order_below_the_shackle(length, body):
    grid = np.asarray(Chain(length=length).cell_grid)
    top = shackle_rows(body[1])
    assert grid.shape == (top + body[0], body[1])
    assert (grid[:top] == -1).all()
    wheels = grid[top:].ravel()
    np.testing.assert_array_equal(wheels[:length], np.arange(length))
    assert (wheels[length:] == -1).all()


def test_the_padlock_shows_the_digits_of_opened_wheels_and_frames_the_current_one():
    env = CombinationLock(length=7, num_actions=4)
    state, _ = env.init(jax.random.key(0))
    tile = 60
    grid = np.asarray(env.cell_grid)
    image = np.asarray(env.render(env.place(state, 4), tile=tile)).astype(int)
    assert image.shape == (grid.shape[0] * tile, grid.shape[1] * tile, 3)
    ink = np.array([25, 25, 25])

    def at(index, across=0.0, down=0.0):
        row, column = np.argwhere(grid == index)[0]
        return image[int((row + 0.5 + down) * tile), int((column + 0.5 + across) * tile)]

    for index in range(4):
        glyph = DIGITS[int(env.secrets[index])]
        for glyph_y in range(5):
            for glyph_x in range(3):
                pixel = at(index, -0.15 + (glyph_x + 0.5) * 0.1, -0.24 + (glyph_y + 0.5) * 0.096)
                assert np.array_equal(pixel, ink) == bool(glyph[glyph_y, glyph_x])
    np.testing.assert_array_equal(at(4, across=0.3), [230, 20, 20])
    assert not np.array_equal(at(4), ink)
    for index in (5, 6):
        np.testing.assert_allclose(at(index), [92, 92, 98], atol=2)
        np.testing.assert_array_equal(at(index, across=0.3), [45, 38, 30])
    np.testing.assert_array_equal(at(0, across=0.45), [210, 165, 60])
    columns = grid.shape[1]
    outer = 0.35 * columns
    crown_top = 0.2 + 0.28 * outer / 2
    np.testing.assert_array_equal(image[int(crown_top * tile), int(columns / 2 * tile)], [180, 184, 192])
    np.testing.assert_array_equal(image[2, 2], [245, 245, 245])
