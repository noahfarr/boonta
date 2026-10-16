import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boonta.environments.dune_sea import ChainOfForks, Hotspots
from boonta.environments.dune_sea.hotspots import Scorecard, reach


def optimum(env, survival):
    successor, reward, terminal = map(np.asarray, (env.successor, env.reward, env.terminal))
    value = np.zeros(env.num_cells)
    for _ in range(20000):
        backed = reward[successor] + survival * np.where(terminal[successor], 0.0, value[successor])
        backed = np.where(terminal, 0.0, backed.max(axis=-1))
        if np.abs(backed - value).max() < 1e-12:
            break
        value = backed
    return value


@pytest.mark.parametrize(
    "depth, num_cells, best",
    [(6, 77, 5.42), (12, 149, 10.08), (24, 293, 17.54), (48, 581, 27.16)],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_the_chain_of_forks_has_the_size_and_best_return_of_the_paper(depth, num_cells, best, seed):
    env = ChainOfForks(depth=depth, seed=seed)
    assert env.num_cells == num_cells
    assert round(float(optimum(env, 1.0 - env.hazard)[env.start]), 2) == best


def test_the_best_policy_without_hazard_collects_one_per_fork_and_the_goal():
    env = ChainOfForks(depth=6, hazard=0.0)
    value = optimum(env, 1.0)
    successor, reward = np.asarray(env.successor), np.asarray(env.reward)
    best = (reward[successor] + value[successor]).argmax(axis=-1)
    state, _ = env.init(jax.random.key(0))
    total, done = 0.0, False
    while not done:
        state, timestep = env.step(jax.random.key(0), state, jnp.int32(best[state.cell]))
        total += float(timestep.reward)
        done = bool(timestep.terminated)
    assert total == 6.0


def test_only_forks_and_dead_ends_offer_a_choice():
    env = ChainOfForks(depth=6)
    successor = np.asarray(env.successor)
    choices = np.sum(successor[:, 0] != successor[:, 1])
    assert choices == 2 * 6


def play(env, num_episodes, policy):
    def episode(key):
        state, _ = env.init(key)

        def body(carry, key):
            state, ended, landing, payout = carry
            policy_key, step_key = jax.random.split(key)
            state, timestep = env.step(step_key, state, policy(policy_key, state))
            done = timestep.terminated | timestep.truncated
            landing = jnp.where(ended, landing, jnp.where(timestep.terminated, state.position, -1))
            payout = payout + jnp.where(ended, 0.0, timestep.reward)
            return (state, ended | done, landing, payout), None

        initial = (state, jnp.bool_(False), jnp.int32(-1), jnp.float32(0.0))
        (_, _, landing, payout), _ = jax.lax.scan(body, initial, jax.random.split(key, env.horizon))
        return landing, payout

    return jax.jit(jax.vmap(episode))(jax.random.split(jax.random.key(0), num_episodes))


def wander(key, state):
    return jax.random.randint(key, (), 0, 4)


def test_a_wandering_agent_reaches_each_hotspot_as_often_as_reach_says():
    env = Hotspots(width=9, num_hotspots=4, nearest=2, farthest=7, horizon=60)
    landing, _ = play(env, 40_000, wander)
    landing = np.asarray(landing)
    landed = np.array([np.mean(landing == hotspot) for hotspot in np.asarray(env.hotspots)])
    np.testing.assert_allclose(landed, reach(env), atol=0.01)


def test_only_the_secret_action_at_a_hotspot_pays():
    env = Hotspots(width=9, num_hotspots=4, nearest=2, farthest=7, horizon=60)

    def informed(key, state):
        hotspot = env.hotspot_index[state.position]
        return jnp.where(hotspot >= 0, env.secrets[jnp.maximum(hotspot, 0)], wander(key, state))

    def mistaken(key, state):
        return (informed(key, state) + 1) % 4

    _, informed_return = play(env, 40_000, informed)
    _, mistaken_return = play(env, 40_000, mistaken)
    assert abs(float(np.mean(informed_return)) - reach(env).sum()) < 0.01
    assert float(np.max(mistaken_return)) == 0.0


def test_hotspots_sit_at_the_requested_distances_from_the_start():
    env = Hotspots(width=15, num_hotspots=6, nearest=3, farthest=13, seed=4)
    rows, columns = np.divmod(np.asarray(env.hotspots), env.width)
    start_row, start_column = divmod(env.start, env.width)
    distances = np.abs(rows - start_row) + np.abs(columns - start_column)
    np.testing.assert_array_equal(distances, [3, 5, 7, 9, 11, 13])


def test_hotspots_render_the_agent_red_and_the_hotspots_green():
    env = Hotspots(width=9, num_hotspots=4, nearest=2, farthest=7)
    state, _ = env.init(jax.random.key(0))
    tile = 8
    image = np.asarray(env.render(state, tile=tile))
    assert image.shape == ((env.width + 2) * tile, (env.width + 2) * tile, 3)
    assert image.dtype == np.uint8

    def centre(cell, across=0.5):
        row, column = divmod(int(cell), env.width)
        return image[int((row + 1.5) * tile), int((column + 1 + across) * tile)]

    np.testing.assert_array_equal(centre(env.start, across=0.4), [255, 0, 0])
    for hotspot in np.asarray(env.hotspots):
        np.testing.assert_array_equal(centre(hotspot), [0, 255, 0])


class Oracle:
    def __init__(self, env, confidence):
        table = np.zeros((env.num_cells, 4), np.float32)
        table[np.asarray(env.hotspots), np.asarray(env.secrets)] = confidence
        self.table = jnp.asarray(table)

    def step(self, state, key, timestep, temperature):
        return state, None, {"logits": timestep.obs @ self.table}


@pytest.mark.parametrize("confidence, share", [(50.0, 1.0), (0.0, 0.25)])
def test_the_scorecard_weighs_each_hotspot_by_how_often_rho_reaches_it(confidence, share):
    env = Hotspots(width=9, num_hotspots=4, nearest=2, farthest=7, horizon=60)
    state = type("State", (), {"algorithm_state": None})()
    data = Scorecard().craft(jax.random.key(0), Oracle(env, confidence), env, state, {}).data
    best = reach(env).sum()
    assert abs(float(data["hotspots/best_return"]) - best) < 1e-5
    assert abs(float(data["hotspots/return"]) - share * best) < 1e-5


def test_hotspots_fill_the_nearest_free_distance_once_the_requested_one_is_full():
    env = Hotspots(width=9, num_hotspots=80, nearest=1, farthest=8)
    hotspots = np.asarray(env.hotspots)
    assert len(set(hotspots.tolist())) == 80
    assert set(hotspots.tolist()) == set(range(81)) - {env.start}
