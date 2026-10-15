import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import zoo
from boonta.algorithms.ppo import PPO, PPOConfig
from boonta.curricula import sgs
from boonta.curricula.sgs import SGS, TaskPool
from boonta.environments.wrappers import RecordEpisodeStatistics, SameStepAutoReset
from boonta.networks import ActorCritic, Network
from boonta.podracers import anakin
from boonta.utils import Timestep, Transition, mesh
from dummies import Dial


def pit(pool_size=3, num_envs=4, history_length=4, success_threshold=0.0):
    return SGS(
        None,
        pool_size=pool_size,
        num_envs=num_envs,
        history_length=history_length,
        target_success=0.5,
        concentration=1.0,
        epsilon=1e-4,
        temperature=2.0,
        success_threshold=success_threshold,
        generate_level=lambda key: jax.random.uniform(key),
    )


def pool(outcomes, num_outcomes, cursor=None):
    outcomes = jnp.array(outcomes, jnp.float32)
    pool_size, _ = outcomes.shape
    return TaskPool(
        levels=jnp.arange(pool_size, dtype=jnp.float32),
        outcomes=outcomes,
        num_outcomes=jnp.array(num_outcomes, jnp.int32),
        cursor=jnp.zeros(pool_size, jnp.int32) if cursor is None else jnp.array(cursor, jnp.int32),
    )


def test_the_kernel_prefers_tasks_at_the_target_success_rate_but_keeps_the_extremes():
    tasks = pool([[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]], [4, 4, 4])
    weights = np.asarray(pit().sampling_weights(tasks))
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-6)
    assert weights[1] > weights[0] and weights[1] > weights[2]
    assert weights[0] > 0.0 and weights[2] > 0.0


def test_the_success_rate_is_the_mean_of_the_recorded_outcomes():
    tasks = pool([[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]], [2, 4, 0])
    np.testing.assert_allclose(pit().success_rates(tasks), [0.5, 0.75, 0.0])


def test_the_history_keeps_only_the_latest_outcomes():
    tasks = pool([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [0, 0, 0], cursor=[0, 3, 0])
    recorded = pit().record(tasks, jnp.array([2, 2, 9]), jnp.array([1, 2, 9]))
    np.testing.assert_array_equal(recorded.outcomes[0], [1, 0, 0, 0])
    np.testing.assert_array_equal(recorded.outcomes[1], [1, 0, 0, 1])
    np.testing.assert_array_equal(recorded.outcomes[2], [1, 1, 1, 1])
    np.testing.assert_array_equal(recorded.num_outcomes, [2, 2, 4])
    np.testing.assert_array_equal(recorded.cursor, [2, 1, 0])


def played(theta, reward, terminated, truncated):
    steps, envs = reward.shape
    timestep = Timestep(
        obs=jnp.zeros((steps, envs, 3)),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info={"theta": jnp.array(theta, jnp.int32)},
    )
    return Transition(first=timestep, second=timestep, aux={})


def test_an_episode_succeeds_only_if_it_terminates_with_the_success_reward():
    reward = jnp.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    terminated = jnp.array([[False, False, False, False], [True, False, True, False]])
    truncated = jnp.array([[False, False, False, False], [False, True, False, False]])
    theta = jnp.array([[0, 1, 2, 2], [0, 1, 2, -1]])
    episodes, successes = pit().count_outcomes(played(theta, reward, terminated, truncated))
    np.testing.assert_array_equal(episodes, [1, 1, 1])
    np.testing.assert_array_equal(successes, [1, 0, 0])


def test_an_untagged_episode_is_never_counted():
    done = jnp.ones((1, 3), bool)
    transitions = played(-jnp.ones((1, 3)), jnp.ones((1, 3)), done, ~done)
    episodes, successes = pit().count_outcomes(transitions)
    np.testing.assert_array_equal(episodes, 0)
    np.testing.assert_array_equal(successes, 0)


def ppo():
    environment = Dial()
    return PPO(
        cfg=PPOConfig(
            num_minibatches=2,
            update_epochs=1,
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            gamma=0.99,
            gae_lambda=0.95,
        ),
        network=Network(
            feature_extractor=zoo.encoder(),
            head=ActorCritic(actor=zoo.policy(environment), critic=nn.Dense(1)),
        ),
        optimizer=zoo.adam(),
    )


class Blink(Dial):
    def step(self, key, state, action):
        state, timestep = super().step(key, state, action)
        return state, timestep.replace(terminated=state.clock >= 2.0)


def test_sgs_samples_its_pool_beneath_the_episode_statistics():
    algorithm, environment, pit, lap = sgs.make(
        ppo(),
        SameStepAutoReset(Blink(), 8),
        pool_size=16,
        generate_level=lambda key: jax.random.uniform(key, minval=1.0, maxval=2.0),
    )
    environment = RecordEpisodeStatistics(environment)
    config = anakin.AnakinConfig(num_envs=8, num_steps=8, mesh=mesh(1))
    podracer = anakin.make(config, algorithm, environment, pit, lap)
    state = podracer.init(jax.random.key(0))
    state, logs = podracer.train(state, jax.random.key(1), 6)
    assert np.asarray(logs["pool/num_tried"]).ravel()[-1] > 0
    assert int(jnp.sum(state.pit_state.num_outcomes)) > 0
    assert lap is None

    playing = np.asarray(state.environment_state.env_state.playing)
    assert np.all((playing >= 0) & (playing < 16))

    evaluated, _ = podracer.evaluate(state, jax.random.key(2), 4)
    for kept, held in zip(jax.tree.leaves(state.pit_state), jax.tree.leaves(evaluated.pit_state)):
        np.testing.assert_array_equal(kept, held)


def test_sgs_keeps_running_episodes_across_updates():
    algorithm, environment, pit, lap = sgs.make(
        ppo(),
        SameStepAutoReset(Dial(), 8),
        pool_size=4,
        generate_level=lambda key: jax.random.uniform(key, minval=1.0, maxval=2.0),
    )
    config = anakin.AnakinConfig(num_envs=8, num_steps=4, mesh=mesh(1))
    podracer = anakin.make(config, algorithm, environment, pit, lap)
    state = podracer.init(jax.random.key(0))
    state, _ = podracer.train(state, jax.random.key(1), 1)
    first = np.asarray(state.environment_state.playing)
    state, _ = podracer.train(state, jax.random.key(2), 3)
    np.testing.assert_array_equal(state.environment_state.playing, first)
