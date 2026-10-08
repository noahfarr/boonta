import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boonta.planners import CEM, MPC, MPPI, RandomShooting

NUM_ENVS, HORIZON, ACTION_DIM = 4, 3, 2
SHAPE = (NUM_ENVS, HORIZON, ACTION_DIM)
GOAL = jnp.linspace(-0.5, 0.5, NUM_ENVS * ACTION_DIM).reshape(NUM_ENVS, 1, ACTION_DIM)

PLANNERS = [
    pytest.param(
        CEM(SHAPE, population_size=256, num_elites=32, num_iterations=6, min_std=0.05, max_std=2.0),
        id="cem",
    ),
    pytest.param(
        MPPI(
            SHAPE,
            population_size=256,
            num_elites=32,
            num_iterations=6,
            elite_temperature=0.5,
            min_std=0.05,
            max_std=2.0,
        ),
        id="mppi",
    ),
    pytest.param(RandomShooting(SHAPE, population_size=256), id="random_shooting"),
]


def sample_fn(key, mean, std, num_proposals):
    noise = jax.random.normal(key, (NUM_ENVS, num_proposals, HORIZON, ACTION_DIM))
    return jnp.clip(mean[:, None] + std[:, None] * noise, -1.0, 1.0)


def rollout_fn(key, actions):
    return -jnp.sum((actions - GOAL[:, None]) ** 2, axis=(-2, -1))


def miss(action):
    return float(jnp.sum((action - GOAL[:, 0]) ** 2))


@pytest.mark.parametrize("planner", PLANNERS)
def test_every_planner_acts_far_better_than_chance(planner):
    controller = MPC(planner=planner)
    state = controller.init(jax.random.key(0))
    done = jnp.ones(NUM_ENVS, bool)

    _, action = jax.jit(controller.step, static_argnums=(2, 3))(
        jax.random.key(1), state, rollout_fn, sample_fn, done, 0.0
    )
    proposals = sample_fn(jax.random.key(2), jnp.zeros(SHAPE), jnp.ones(SHAPE), 256)
    chance = np.mean([miss(proposals[:, index, 0]) for index in range(256)])

    assert action.shape == (NUM_ENVS, ACTION_DIM)
    assert miss(action) < 0.25 * chance
