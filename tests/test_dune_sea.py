import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boonta.environments.dune_sea import ChainOfForks


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
