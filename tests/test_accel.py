import jax
import numpy as np

from boonta.curricula import accel
from boonta.curricula.plr import LevelBufferState
from boonta.environments.wrappers import (RecordEpisodeStatistics,
                                          SameStepAutoReset, Vectorize)
from boonta.podracers import anakin
from boonta.utils import mesh
from test_plr import Blink, graded


def shift(key, level, num_edits):
    return level + 10.0 * num_edits


def raced():
    algorithm, environment, pit, lap = accel(
        graded(robust=False).algorithm,
        Vectorize(SameStepAutoReset(Blink()), 8),
        num_edits=1,
        mutate=shift,
        capacity=16,
        replay_probability=0.5,
        staleness=0.3,
        temperature=1.0,
        minimum_fill=0.5,
        robust=False,
        gamma=0.99,
        gae_lambda=0.95,
        sample=lambda key: jax.random.uniform(key, minval=1.0, maxval=2.0),
    )
    environment = RecordEpisodeStatistics(environment)
    config = anakin.AnakinConfig(num_envs=8, num_steps=8, mesh=mesh(1))
    return anakin.make(config, algorithm, environment, pit, lap)


def trained(num_updates):
    podracer = raced()
    state = podracer.init(jax.random.key(0))
    state, logs = podracer.train(state, jax.random.key(1), num_updates)
    return state, jax.tree.map(lambda leaf: np.asarray(leaf).ravel(), logs)


def schedule(logs):
    mutating = logs["accel/levels/mutating"] > 0
    replaying = (logs["plr/levels/replaying"] > 0) & ~mutating
    return replaying, mutating


def test_mutation_follows_every_replay_and_nothing_else():
    _, logs = trained(24)
    replaying, mutating = schedule(logs)
    assert replaying.any() and mutating.any()
    assert not mutating[0]
    np.testing.assert_array_equal(mutating[1:], replaying[:-1])


def test_mutated_levels_enter_the_buffer():
    state, logs = trained(24)
    _, mutating = schedule(logs)
    after_mutation = np.concatenate([[False], mutating[:-1]])
    assert (logs["plr/levels/admitted"][after_mutation] > 0).any()
    buffer = state.environment_state.env_state
    assert isinstance(buffer, LevelBufferState)
    levels = np.asarray(buffer.theta)[: int(buffer.size)]
    assert (levels >= 10.0).any()


def test_a_mutation_rollout_plays_only_children_and_then_plr_decides_again():
    _, logs = trained(24)
    replaying, mutating = schedule(logs)
    after_mutation = np.concatenate([[False], mutating[:-1]])
    np.testing.assert_array_equal(logs["plr/levels/replayed"][after_mutation], 0.0)
    assert not mutating[after_mutation].any()
    assert replaying[after_mutation].any() and (~replaying[after_mutation]).any()


def test_a_mutation_leaves_the_staleness_clock_where_the_replay_left_it():
    podracer = raced()
    state = podracer.init(jax.random.key(0))
    mutations = 0
    for update in range(24):
        before = int(state.environment_state.env_state.episodes)
        state, logs = podracer.train(state, jax.random.key(update), 1)
        if np.asarray(logs["accel/levels/mutating"]).any():
            mutations += 1
            assert int(state.environment_state.env_state.episodes) == before
    assert mutations > 0
