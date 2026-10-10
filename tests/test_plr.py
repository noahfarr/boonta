import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

import zoo
from boonta.algorithms.ppo import PPO, PPOConfig
from boonta.curricula import plr
from boonta.curricula.plr import (Graded, LevelBufferState, admit_levels,
                                  draw_levels, maximum_monte_carlo,
                                  positive_value_loss, rank_weights,
                                  replay_weights, staleness_weights,
                                  tally_episodes, update_scores)
from boonta.environments.wrappers import (RecordEpisodeStatistics,
                                          SameStepAutoReset)
from boonta.networks import ActorCritic, Network
from boonta.podracers import anakin
from boonta.utils import Timestep, Transition, mesh
from dummies import Dial


def test_rank_weights_are_inverse_ranks_sharpened_by_the_temperature():
    scores = jnp.array([3.0, 1.0, 2.0, 7.0])
    np.testing.assert_allclose(rank_weights(scores, 3, 1.0), np.array([6, 2, 3, 0]) / 11, rtol=1e-6)
    np.testing.assert_allclose(rank_weights(scores, 3, 0.5), np.array([36, 4, 9, 0]) / 49, rtol=1e-6)


def test_staleness_weights_follow_episodes_since_each_level_was_played():
    timestamps = jnp.array([2, 5, 0, 9])
    np.testing.assert_allclose(staleness_weights(timestamps, 3, 6), np.array([4, 1, 6, 0]) / 11, rtol=1e-6)


def test_staleness_is_uniform_over_the_buffer_when_nothing_is_staleness_weights():
    np.testing.assert_allclose(staleness_weights(jnp.array([6, 6, 6, 0]), 3, 6), [1 / 3, 1 / 3, 1 / 3, 0])


def test_priority_mixes_rank_and_staleness():
    weights = replay_weights(jnp.array([3.0, 1.0, 2.0, 7.0]), jnp.array([2, 5, 0, 9]), 3, 6, 1.0, 0.3)
    np.testing.assert_allclose(weights, np.array([5.4, 1.7, 3.9, 0.0]) / 11, rtol=1e-6)


def test_an_empty_buffer_has_no_weight():
    np.testing.assert_array_equal(replay_weights(jnp.zeros(3), jnp.zeros(3, jnp.int32), 0, 0, 1.0, 0.3), 0.0)


def rollout():
    theta = jnp.array([[2, 2, -1], [2, 2, 0], [5, 2, 0], [5, 2, 0]])
    done = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]], bool)
    reward = jnp.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [3.0, 5.0, 0.0]])
    advantages = jnp.array([[1.0, 0.5, 9.0], [-1.0, 0.5, 0.0], [2.0, -3.0, 0.0], [4.0, 9.0, 0.0]])
    values = jnp.array([[1.0, 3.0, 9.0], [3.0, 3.0, 0.0], [0.0, 3.0, 0.0], [2.0, 9.0, 0.0]])
    second = Timestep(
        obs=jnp.zeros((4, 3)),
        reward=reward,
        terminated=done,
        truncated=jnp.zeros_like(done),
        info={"theta": theta},
    )
    return Transition(first=second, second=second, aux={}), values, advantages


def test_a_tally_scores_each_finished_episode_on_the_level_it_played():
    transitions, values, advantages = rollout()
    counted = jax.tree.map(np.asarray, tally_episodes(transitions, values, advantages, slots=6))
    np.testing.assert_array_equal(counted.episodes, [0, 0, 2, 0, 0, 1])
    np.testing.assert_allclose(counted.regret[[2, 5]], [0.5 + 1 / 3, 3.0], rtol=1e-6)
    np.testing.assert_allclose(counted.value[[2, 5]], [5.0, 1.0], rtol=1e-6)
    np.testing.assert_array_equal(counted.returns[[2, 5]], [3.0, 3.0])
    assert np.isneginf(counted.returns[0])


def test_scores_are_the_positive_value_loss_or_the_maximum_monte_carlo_regret():
    transitions, values, advantages = rollout()
    counted = tally_episodes(transitions, values, advantages, slots=6)
    returns = jnp.full(6, -jnp.inf).at[2].set(4.0)
    regret = np.asarray(positive_value_loss(counted, returns))
    monte_carlo = np.asarray(maximum_monte_carlo(counted, returns))
    np.testing.assert_allclose(regret[[2, 5]], [5 / 12, 3.0], rtol=1e-6)
    np.testing.assert_allclose(monte_carlo[[2, 5]], [1.5, 2.0], rtol=1e-6)


@struct.dataclass
class Inner:
    playing: jax.Array


def buffer(scores, timestamps, size, episodes):
    capacity = len(scores)
    return LevelBufferState(
        Inner(playing=jnp.array([-1])),
        scores=jnp.array(scores, jnp.float32),
        timestamps=jnp.array(timestamps, jnp.int32),
        returns=jnp.full(capacity, -jnp.inf),
        size=jnp.int32(size),
        episodes=jnp.int32(episodes),
        exploring=jnp.bool_(False),
        key=jax.random.key(0),
    )


def fresh(scores):
    scored = jnp.array(scores, jnp.float32)
    counted = tally_episodes(*rollout(), slots=len(scores)).replace(returns=jnp.zeros(len(scores)))
    return scored, counted


def test_a_fresh_level_replaces_the_lowest_weighted_level_only_if_it_scores_higher():
    levels = jnp.arange(6.0) * 10
    scored, counted = fresh([0, 0, 0, 2.0, 0.5, -jnp.inf])
    state, levels, admitted = admit_levels(
        buffer([1.0, 5.0, 3.0], [10, 10, 10], 3, 10),
        levels, counted, scored, jnp.bool_(True), 1.0, 0.3,
    )
    np.testing.assert_array_equal(state.scores, [2.0, 5.0, 3.0])
    np.testing.assert_array_equal(levels[:3], [30.0, 10.0, 20.0])
    np.testing.assert_array_equal(state.timestamps, [11, 10, 10])
    assert int(state.episodes) == 13 and int(admitted) == 1


def test_any_scored_fresh_level_enters_a_buffer_that_is_not_full():
    levels = jnp.arange(6.0) * 10
    scored, counted = fresh([0, 0, 0, -5.0, -jnp.inf, -jnp.inf])
    state, levels, admitted = admit_levels(
        buffer([7.0, -jnp.inf, -jnp.inf], [0, 0, 0], 1, 3),
        levels, counted, scored, jnp.bool_(True), 1.0, 0.3,
    )
    assert int(state.size) == 2 and int(admitted) == 1
    np.testing.assert_array_equal(state.scores[:2], [7.0, -5.0])
    np.testing.assert_array_equal(levels[1], 30.0)


def test_nothing_is_admitted_after_a_replay_rollout():
    levels = jnp.arange(6.0) * 10
    scored, counted = fresh([0, 0, 0, 9.0, 9.0, 9.0])
    state, levels, admitted = admit_levels(
        buffer([1.0, -jnp.inf, -jnp.inf], [0, 0, 0], 1, 3),
        levels, counted, scored, jnp.bool_(False), 1.0, 0.3,
    )
    assert int(state.size) == 1 and int(admitted) == 0 and int(state.episodes) == 3


def test_a_replayed_level_takes_its_new_score_and_best_return():
    counted = tally_episodes(*rollout(), slots=6).replace(
        episodes=jnp.array([0, 2, 0, 0, 0, 0], jnp.float32),
        returns=jnp.array([0, 4.0, 0, 0, 0, 0]),
    )
    scored = jnp.array([0, 9.0, 0, 0, 0, 0])
    state = update_scores(buffer([1.0, 5.0, 3.0], [1, 2, 3], 3, 10), counted, scored)
    np.testing.assert_array_equal(state.scores, [1.0, 9.0, 3.0])
    np.testing.assert_array_equal(state.returns[1], 4.0)
    np.testing.assert_array_equal(state.timestamps, [1, 2, 3])


def test_each_replay_draw_counts_an_episode_and_marks_its_level_as_just_played():
    state, indices = draw_levels(buffer([1.0, 5.0, 3.0], [10, 10, 10], 3, 10), jax.random.key(0), 4, 1.0, 0.3)
    assert int(state.episodes) == 14
    indices = np.asarray(indices)
    assert set(indices) <= {0, 1, 2}
    assert int(state.timestamps[indices[-1]]) == 14


def test_replay_draws_only_from_the_filled_part_of_the_buffer():
    state, indices = draw_levels(buffer([1.0, 5.0, 3.0], [0, 0, 0], 1, 0), jax.random.key(0), 8, 1.0, 0.3)
    np.testing.assert_array_equal(indices, 0)
    assert int(state.timestamps[0]) == 8


def graded(robust):
    environment = Dial()
    algorithm = PPO(
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
    return Graded(algorithm, slots=3, capacity=1, gamma=0.99, gae_lambda=0.95, robust=robust)


def played(theta, cut=1):
    terminated = jnp.zeros((4, 4), bool).at[-1].set(True).at[0].set(True)
    timestep = Timestep(
        obs=jax.random.normal(jax.random.key(0), (4, 4, 3)),
        action=jnp.zeros((4, 4), jnp.int32),
        reward=jnp.ones((4, 4)),
        terminated=terminated,
        truncated=jnp.zeros((4, 4), bool),
        info={"theta": jnp.full((4, 4), theta, jnp.int32).at[0].set(cut)},
    )
    aux = {"log_prob": jnp.full((4, 4), -0.7), "value": jnp.zeros((4, 4))}
    return Transition(first=timestep, second=timestep, aux=aux)


def moved(robust, theta):
    algorithm = graded(robust)
    transitions = played(theta)
    state = algorithm.init(jax.random.key(1), jax.tree.map(lambda leaf: leaf[0], transitions.first))
    updated = algorithm.update(state, jax.random.key(2), transitions)
    leaves = zip(jax.tree.leaves(state.params), jax.tree.leaves(updated.params))
    return updated, any(not np.array_equal(old, new) for old, new in leaves)


def test_robust_plr_skips_the_update_after_a_fresh_rollout():
    updated, changed = moved(robust=True, theta=2)
    assert not changed


def test_robust_plr_trains_after_a_replay_rollout():
    _, changed = moved(robust=True, theta=0)
    assert changed


def test_exploratory_plr_trains_after_a_fresh_rollout_too():
    _, changed = moved(robust=False, theta=2)
    assert changed


def test_the_cut_that_opens_a_rollout_is_never_scored():
    updated, _ = moved(robust=True, theta=2)
    np.testing.assert_array_equal(updated.tally.episodes, [0, 0, 4])


class Blink(Dial):
    def step(self, key, state, action):
        state, timestep = super().step(key, state, action)
        return state, timestep.replace(terminated=state.clock >= 2.0)


def test_plr_fills_its_buffer_beneath_the_episode_statistics():
    algorithm = graded(robust=True).algorithm
    environment = SameStepAutoReset(Blink(), 8)
    algorithm, environment, pit, lap = plr(
        algorithm,
        environment,
        capacity=4,
        replay_probability=0.5,
        staleness=0.3,
        temperature=1.0,
        minimum_fill=0.5,
        robust=True,
        gamma=0.99,
        gae_lambda=0.95,
        sample=lambda key: jax.random.uniform(key, minval=1.0, maxval=2.0),
    )
    environment = RecordEpisodeStatistics(environment)
    config = anakin.AnakinConfig(num_envs=8, num_steps=8, mesh=mesh(1))
    podracer = anakin.make(config, algorithm, environment, pit, lap)
    state = podracer.init(jax.random.key(0))
    state, logs = podracer.train(state, jax.random.key(1), 6)
    sizes = np.asarray(logs["plr/levels/size"])
    assert sizes.ravel()[-1] == 4
    assert np.asarray(logs["plr/levels/replaying"]).any()
    played = np.asarray(state.environment_state.env_state.env_state.playing)
    assert np.all(played >= 0)
