import jax
import jax.numpy as jnp
import pytest

pytest.importorskip("mapox")

from boonta import environments

AGENTS, LENGTH = 2, 32


@pytest.fixture(scope="module")
def env():
    return environments.make("mapox", "find_return", kwargs={"num_agents": AGENTS, "length": LENGTH})


def test_observations_rewards_and_masks_follow_the_declared_spaces(env):
    spaces, action_space = env.observation_space(), env.action_space()
    state, timestep = env.init(jax.random.key(0))
    step = jax.jit(env.step)
    assert timestep.terminated.all() and not timestep.truncated.any()
    assert action_space.shape == (AGENTS,) and action_space.num_actions > 1

    for index in range(5):
        for name, space in spaces.items():
            assert timestep.obs[name].shape == space.shape
            assert timestep.obs[name].dtype == space.dtype
        assert timestep.obs["mask"].any(axis=-1).all()
        assert timestep.reward.shape == (AGENTS,) and timestep.reward.dtype == jnp.float32
        key = jax.random.key(index)
        state, timestep = step(key, state, action_space.sample(key))


def test_an_episode_ends_exactly_at_the_declared_horizon_and_a_new_one_starts(env):
    state, timestep = env.init(jax.random.key(0))
    step = jax.jit(env.step)
    ends = []

    for index in range(env.time_limit() + 1):
        state, timestep = step(jax.random.key(index), state, jnp.zeros((AGENTS,), jnp.int32))
        assert not timestep.truncated.any()
        ends.append(bool(timestep.terminated.all()))

    assert ends == [False] * (env.time_limit() - 1) + [True, False]
