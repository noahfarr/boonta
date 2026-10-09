import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

import zoo
from boonta.algorithms.wrappers.pbt import (PBT, attach_multiplier,
                                            choose_sources, finish_episodes,
                                            read_multiplier, record_episodes,
                                            write_multiplier)
from dummies import match, reach


def pbt(members=4, seeds=2, interval=4, threshold=1.0, low=0.5, high=1.0, factors=(0.5, 2.0)):
    def build(algorithm, environment, num_envs, num_steps):
        algorithm = PBT(
            algorithm,
            members=members,
            seeds=seeds,
            interval=interval,
            fraction=0.25,
            threshold=threshold,
            window=128,
            low=low,
            high=high,
            factors=factors,
        )
        return zoo.online(algorithm, environment, num_envs, num_steps)

    return build


def test_equal_members_never_copy():
    fitness = jnp.full((4, 2), 0.5)
    sources, copied = choose_sources(fitness, jnp.ones(4, bool), jax.random.key(0), 0.25, 0.0)
    np.testing.assert_array_equal(copied, False)
    np.testing.assert_array_equal(sources, np.arange(4))


def test_a_gap_inside_the_noise_between_seeds_never_copies():
    fitness = jnp.array([[0.0, 1.0], [0.1, 0.9], [0.2, 1.0], [0.3, 1.1]])
    for seed in range(8):
        _, copied = choose_sources(fitness, jnp.ones(4, bool), jax.random.key(seed), 0.25, 2.0)
        np.testing.assert_array_equal(copied, False)


def test_the_worst_member_copies_the_best_when_the_gap_is_real():
    fitness = jnp.array([[0.0, 0.1], [0.5, 0.6], [0.4, 0.5], [1.0, 1.1]])
    sources, copied = choose_sources(fitness, jnp.ones(4, bool), jax.random.key(0), 0.25, 2.0)
    np.testing.assert_array_equal(copied, [True, False, False, False])
    np.testing.assert_array_equal(sources, [3, 1, 2, 3])


def test_members_without_a_full_window_neither_copy_nor_are_copied():
    fitness = jnp.array([[0.0, 0.1], [0.9, 1.0], [0.4, 0.5], [1.0, 1.1]])
    ready = jnp.array([False, True, True, False])
    sources, copied = choose_sources(fitness, ready, jax.random.key(0), 0.25, 2.0)
    np.testing.assert_array_equal(copied, [False, False, True, False])
    np.testing.assert_array_equal(sources, [0, 1, 1, 3])


def test_episodes_carry_their_return_across_updates():
    reward = jnp.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    done = jnp.array([[False, True], [True, False], [False, False]])
    running, totals = finish_episodes(jnp.array([5.0, 0.0]), reward, done)
    np.testing.assert_allclose(running, [1.0, 4.0])
    np.testing.assert_allclose(jnp.where(done, totals, 0.0), [[0.0, 2.0], [7.0, 0.0], [0.0, 0.0]])


def test_the_window_keeps_the_latest_episodes():
    window, filled = record_episodes(
        jnp.zeros(3), jnp.int32(2), jnp.arange(1.0, 6.0), jnp.array([True, False, True, True, True])
    )
    np.testing.assert_allclose(window, [3.0, 4.0, 5.0])
    assert int(filled) == 6


def test_the_multiplier_scales_the_update():
    podracer = zoo.ppo(match(), num_envs=8, podracer=pbt())
    optimizer = podracer.algorithm.algorithm.algorithm.optimizer
    algorithm_state, *_ = podracer.init(jax.random.key(0)).algorithm_state.algorithm_states
    params = algorithm_state.params["params"]
    grads = jax.tree.map(lambda leaf: jnp.full_like(leaf, 0.1), params)

    single = write_multiplier(algorithm_state, 1.0)
    tripled = write_multiplier(algorithm_state, 3.0)
    assert float(read_multiplier(tripled)) == 3.0
    plain, _ = optimizer.update(grads, single.optimizer_state, params)
    scaled, _ = optimizer.update(grads, tripled.optimizer_state, params)
    for leaf, reference in zip(jax.tree.leaves(scaled), jax.tree.leaves(plain)):
        np.testing.assert_allclose(leaf, 3.0 * reference, rtol=1e-6)
    assert any(np.abs(np.asarray(leaf)).max() > 0 for leaf in jax.tree.leaves(plain))


def test_algorithms_with_several_optimizers_are_refused():
    podracer = zoo.sac(reach(), num_envs=8)
    with pytest.raises(ValueError, match="single-optimizer"):
        attach_multiplier(podracer.algorithm)


def test_exploit_copies_whole_seed_copies_and_explore_perturbs_only_the_copier():
    podracer = zoo.ppo(match(), num_envs=8, podracer=pbt())
    algorithm = podracer.algorithm
    state = podracer.init(jax.random.key(0)).algorithm_state
    window = jnp.array([0.0, 0.1, 0.5, 0.6, 0.4, 0.5, 1.0, 1.1]).reshape(8, 1)
    state = state.replace(
        window=jnp.broadcast_to(window, (8, 128)), filled=jnp.full(8, 128, jnp.int32)
    )
    before = algorithm.multipliers(state)

    after = algorithm.exploit(state, jax.random.key(1))

    copies = after.algorithm_state.algorithm_states
    originals = state.algorithm_state.algorithm_states
    for seed in range(2):
        for leaf, source in zip(
            jax.tree.leaves(copies[seed].params), jax.tree.leaves(originals[6 + seed].params)
        ):
            np.testing.assert_array_equal(leaf, source)
        moments, _ = copies[seed].optimizer_state
        source, _ = originals[6 + seed].optimizer_state
        for leaf, origin in zip(jax.tree.leaves(moments), jax.tree.leaves(source)):
            np.testing.assert_array_equal(leaf, origin)
    for index in range(2, 8):
        for leaf, source in zip(jax.tree.leaves(copies[index]), jax.tree.leaves(originals[index])):
            np.testing.assert_array_equal(leaf, source)

    multipliers = algorithm.multipliers(after)
    ratio = float(multipliers[0, 0] / before[3, 0])
    assert ratio == pytest.approx(0.5) or ratio == pytest.approx(2.0)
    np.testing.assert_array_equal(multipliers[0, 0], multipliers[0, 1])
    np.testing.assert_array_equal(multipliers[1:], before[1:])
    np.testing.assert_array_equal(after.filled, [0, 0, 128, 128, 128, 128, 128, 128])
    np.testing.assert_array_equal(after.replaced, [1, 0, 0, 0])


def test_initial_multipliers_are_shared_within_a_member():
    podracer = zoo.ppo(match(), num_envs=8, podracer=pbt(low=0.1, high=10.0))
    state = podracer.init(jax.random.key(0)).algorithm_state
    multipliers = np.asarray(podracer.algorithm.multipliers(state))
    np.testing.assert_array_equal(multipliers[:, 0], multipliers[:, 1])
    assert np.all((multipliers >= 0.1) & (multipliers <= 10.0))
    assert len(np.unique(multipliers[:, 0])) == 4


def evaluation_return(podracer, num_updates):
    state = podracer.init(jax.random.key(0))
    state, _ = podracer.train(state, jax.random.key(1), num_updates)
    _, logs = podracer.evaluate(state, jax.random.key(2), 24)
    return state, np.nanmean(np.asarray(logs["episode_statistics/episode_return"]))


def test_pbt_raises_a_learning_rate_too_small_to_solve_the_task():
    environment = reach()
    optimizer = zoo.adam(3e-5)

    fixed = zoo.ppo(environment, num_envs=64, optimizer=optimizer, podracer=pbt(interval=1000))
    _, fixed_return = evaluation_return(fixed, 150)
    assert fixed_return < environment.solved

    trained = zoo.ppo(environment, num_envs=64, optimizer=optimizer, podracer=pbt())
    state, trained_return = evaluation_return(trained, 150)
    assert trained_return >= environment.solved
    assert float(trained.algorithm.multipliers(state.algorithm_state).min()) > 1.0


def test_running_returns_shard_with_the_environments():
    def build(algorithm, environment, num_envs, num_steps):
        algorithm = PBT(
            algorithm, members=4, seeds=2, interval=1, fraction=0.25, threshold=0.0,
            window=4, low=0.5, high=1.0, factors=(0.8, 1.25),
        )
        return zoo.online(algorithm, environment, num_envs, num_steps, devices=4)

    podracer = zoo.ppo(match(), num_envs=16, podracer=build)
    state = podracer.init(jax.random.key(0))
    state, logs = podracer.train(state, jax.random.key(1), 3)
    assert state.algorithm_state.running.sharding.spec == P("data")
    assert np.asarray(logs["pbt/member_0/multiplier"]).shape[0] == 3
