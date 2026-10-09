from dataclasses import fields, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

import zoo
from boonta.algorithms.wrappers.injected import Injected
from boonta.artisans import Leaderboard
from boonta.curricula import pbt as curriculum
from boonta.curricula.pbt import (MULTIPLIER, attach_multiplier,
                                  choose_sources, read_hyperparameters,
                                  read_multiplier, record_episodes,
                                  write_hyperparameters,
                                  write_multiplier)
from boonta.podracers import anakin
from boonta.utils import mesh
from dummies import corridor, match, reach, recall, recall_continuous


def pbt(
    members=4,
    seeds=2,
    interval=4,
    threshold=1.0,
    search=None,
    factors=(0.5, 2.0),
    window=128,
    devices=1,
):
    def build(algorithm, environment, num_envs, num_steps):
        algorithm, environment, pit, lap = curriculum(
            algorithm,
            environment,
            members=members,
            seeds=seeds,
            interval=interval,
            fraction=0.25,
            threshold=threshold,
            window=window,
            search=search or {MULTIPLIER: (0.5, 1.0)},
            factors=factors,
        )
        config = anakin.AnakinConfig(
            num_envs=num_envs, num_steps=num_steps, mesh=mesh(devices)
        )
        return anakin.make(config, algorithm, environment, pit, lap)

    return build


def floats(algorithm) -> tuple[str, ...]:
    return tuple(
        field.name for field in fields(algorithm.cfg) if field.type in (float, "float")
    )


def injecting(algorithm, environment, num_envs, num_steps):
    return zoo.online(
        Injected(algorithm, names=floats(algorithm)), environment, num_envs, num_steps
    )


def copied(tree):
    return jax.tree.map(jnp.array, tree)


ONLINE = [
    pytest.param(zoo.ppo, corridor, id="ppo"),
    pytest.param(zoo.mmd, corridor, id="mmd"),
    pytest.param(zoo.grpo, corridor, id="grpo"),
    pytest.param(zoo.pqn, corridor, id="pqn"),
    pytest.param(zoo.dqn, corridor, id="dqn"),
    pytest.param(zoo.sac, reach, id="sac"),
    pytest.param(zoo.reppo, reach, id="reppo"),
    pytest.param(zoo.recurrent_ppo, recall, id="recurrent_ppo"),
    pytest.param(zoo.recurrent_pupo, recall, id="recurrent_pupo"),
    pytest.param(zoo.recurrent_grpo, recall, id="recurrent_grpo"),
    pytest.param(zoo.recurrent_pqn, recall, id="recurrent_pqn"),
    pytest.param(zoo.recurrent_dqn, recall, id="recurrent_dqn"),
    pytest.param(zoo.recurrent_sac, recall_continuous, id="recurrent_sac"),
]

OFFLINE = [
    pytest.param(zoo.bc, corridor, id="bc"),
    pytest.param(zoo.iql, reach, id="iql"),
    pytest.param(zoo.recurrent_bc, recall, id="recurrent_bc"),
]


@pytest.mark.parametrize("build, environment", ONLINE)
def test_every_float_in_an_online_config_can_be_injected(build, environment):
    podracer = build(environment(), podracer=injecting)
    assert podracer.algorithm.names
    state = podracer.init(jax.random.key(0))
    state, _ = podracer.train(state, jax.random.key(1), 2)
    podracer.evaluate(state, jax.random.key(2), 4)


@pytest.mark.parametrize("build, environment", OFFLINE)
def test_every_float_in_an_offline_config_can_be_injected(build, environment, monkeypatch):
    offline = zoo.offline

    def inject(algorithm, *args, **kwargs):
        return offline(Injected(algorithm, names=floats(algorithm)), *args, **kwargs)

    monkeypatch.setattr(zoo, "offline", inject)
    podracer = build(environment())
    assert podracer.algorithm.names
    state = podracer.init(jax.random.key(0))
    state, _ = podracer.train(state, jax.random.key(1), 2)
    podracer.evaluate(state, jax.random.key(2), 4)


def test_injected_values_drive_the_algorithm():
    podracer = zoo.ppo(match(), num_envs=8, podracer=injecting)
    state = podracer.init(jax.random.key(0))
    silenced = copied(state).replace(
        algorithm_state=state.algorithm_state.replace(
            values={**state.algorithm_state.values, "clip_coefficient": jnp.float32(0.0)}
        )
    )
    silenced = copied(silenced)
    trained, _ = podracer.train(state, jax.random.key(1), 1)
    clipped, _ = podracer.train(silenced, jax.random.key(1), 1)
    differences = [
        float(jnp.abs(left - right).max())
        for left, right in zip(
            jax.tree.leaves(trained.algorithm_state.params),
            jax.tree.leaves(clipped.algorithm_state.params),
        )
    ]
    assert max(differences) > 0


@pytest.mark.parametrize("names", [("num_minibatches",), ("clip_value_loss",), ("missing",)])
def test_only_float_config_fields_can_be_injected(names):
    podracer = zoo.ppo(match(), num_envs=8)
    with pytest.raises(ValueError):
        Injected(podracer.algorithm, names=names)


def test_equal_members_never_copy():
    fitness = jnp.full((4, 2), 0.5)
    sources, copies = choose_sources(fitness, jnp.ones(4, bool), jax.random.key(0), 0.25, 0.0)
    np.testing.assert_array_equal(copies, False)
    np.testing.assert_array_equal(sources, np.arange(4))


def test_a_gap_inside_the_noise_between_seeds_never_copies():
    fitness = jnp.array([[0.0, 1.0], [0.1, 0.9], [0.2, 1.0], [0.3, 1.1]])
    for seed in range(8):
        _, copies = choose_sources(fitness, jnp.ones(4, bool), jax.random.key(seed), 0.25, 2.0)
        np.testing.assert_array_equal(copies, False)


def test_the_worst_member_copies_the_best_when_the_gap_is_real():
    fitness = jnp.array([[0.0, 0.1], [0.5, 0.6], [0.4, 0.5], [1.0, 1.1]])
    sources, copies = choose_sources(fitness, jnp.ones(4, bool), jax.random.key(0), 0.25, 2.0)
    np.testing.assert_array_equal(copies, [True, False, False, False])
    np.testing.assert_array_equal(sources, [3, 1, 2, 3])


def test_members_without_a_full_window_neither_copy_nor_are_copied():
    fitness = jnp.array([[0.0, 0.1], [0.9, 1.0], [0.4, 0.5], [1.0, 1.1]])
    ready = jnp.array([False, True, True, False])
    sources, copies = choose_sources(fitness, ready, jax.random.key(0), 0.25, 2.0)
    np.testing.assert_array_equal(copies, [False, False, True, False])
    np.testing.assert_array_equal(sources, [0, 1, 1, 3])


def test_the_window_keeps_the_latest_episodes():
    window, filled = record_episodes(
        jnp.zeros(3), jnp.int32(2), jnp.arange(1.0, 6.0), jnp.array([True, False, True, True, True])
    )
    np.testing.assert_allclose(window, [3.0, 4.0, 5.0])
    assert int(filled) == 6


def test_lap_tallies_each_copys_episode_returns_across_an_episode_boundary():
    podracer = zoo.ppo(match(), num_envs=8, podracer=pbt(window=3))
    state = podracer.init(jax.random.key(0))
    rewards = [
        [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
        [5.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    dones = [
        [False, False, False, False, False, False, False, False],
        [True, False, False, False, False, False, False, True],
        [True, True, False, False, False, False, False, False],
    ]
    for reward, done in zip(rewards, dones):
        timestep = state.timestep.replace(
            reward=jnp.array(reward),
            terminated=jnp.array(done),
            truncated=jnp.zeros(8, bool),
        )
        state = podracer.lap(state.replace(timestep=timestep))
    tallied = state.algorithm_state
    np.testing.assert_array_equal(tallied.filled, [2, 1, 0, 0, 0, 0, 0, 1])
    np.testing.assert_allclose(tallied.window[0, :2], [2.0, 5.0])
    np.testing.assert_allclose(tallied.window[1, :1], [6.0])
    np.testing.assert_allclose(tallied.window[7, :1], [3.0])
    np.testing.assert_allclose(tallied.running, 0.0)


def test_the_multiplier_scales_the_update():
    podracer = zoo.ppo(match(), num_envs=8, podracer=pbt())
    optimizer = podracer.algorithm.algorithm.algorithm.algorithm.optimizer
    copy_state, *_ = podracer.init(jax.random.key(0)).algorithm_state.algorithm_states
    algorithm_state = copy_state.algorithm_state
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


def test_one_multiplier_reaches_every_optimizer():
    podracer = zoo.sac(reach(), num_envs=8)
    algorithm = attach_multiplier(podracer.algorithm)
    for name in ("actor_optimizer", "critic_optimizer", "alpha_optimizer"):
        state = getattr(algorithm, name).init({"weight": jnp.ones(2)})
        _, injected = write_multiplier(state, 2.0)
        assert float(injected.hyperparams["multiplier"]) == 2.0


def test_algorithms_without_an_optimizer_are_refused():
    podracer = zoo.ppo(match(), num_envs=8)
    with pytest.raises(ValueError, match="GradientTransformation"):
        attach_multiplier(Injected(podracer.algorithm))


def test_multi_optimizer_algorithms_train_under_pbt():
    build = pbt(members=4, seeds=2, interval=1)

    def resized(algorithm, environment, num_envs, num_steps):
        algorithm = replace(algorithm, buffer=zoo.transitions(num_envs // 8, batch_size=8))
        return build(algorithm, environment, num_envs, num_steps)

    podracer = zoo.sac(reach(), num_envs=8, podracer=resized)
    state = podracer.init(jax.random.key(0))
    state, logs = podracer.train(state, jax.random.key(1), 2)
    assert np.asarray(logs[f"pbt/member_0/{MULTIPLIER}"]).shape[0] == 2


def test_sebulba_refuses_pbt():
    def build(algorithm, environment, num_envs, num_steps):
        algorithm, environment, pit, lap = curriculum(
            algorithm, environment, members=2, seeds=1, interval=1, fraction=0.5,
            threshold=0.0, window=4, search={MULTIPLIER: (0.5, 1.0)}, factors=(0.8, 1.25),
        )
        return zoo.asynchronous(algorithm, environment, num_envs, num_steps)

    with pytest.raises(AssertionError, match="Use anakin"):
        zoo.ppo(match(), num_envs=8, podracer=build)


def prepared(search, interval=4, factors=(0.5, 2.0)):
    podracer = zoo.ppo(
        match(), num_envs=8, podracer=pbt(search=search, interval=interval, factors=factors)
    )
    state = podracer.init(jax.random.key(0))
    window = jnp.array([0.0, 0.1, 0.5, 0.6, 0.4, 0.5, 1.0, 1.1]).reshape(8, 1)
    algorithm_state = state.algorithm_state.replace(
        window=jnp.broadcast_to(window, (8, 128)),
        filled=jnp.full(8, 128, jnp.int32),
        updates=jnp.array(interval, jnp.int32),
        handled=jnp.array(interval - 1, jnp.int32),
    )
    return podracer, state.replace(algorithm_state=algorithm_state)


@pytest.mark.parametrize("factor, bound", [(100.0, 1.0), (0.01, 0.5)])
def test_a_value_perturbed_past_its_bound_lands_on_the_bound(factor, bound):
    search = {MULTIPLIER: (0.5, 1.0), "entropy_coefficient": (0.001, 0.1)}
    podracer, state = prepared(search, factors=(factor,))
    after = podracer.pit(state).algorithm_state
    values = read_hyperparameters(podracer.algorithm, after)
    np.testing.assert_allclose(values[MULTIPLIER][0], bound, rtol=1e-6)
    entropy = 0.1 if factor > 1 else 0.001
    np.testing.assert_allclose(values["entropy_coefficient"][0], entropy, rtol=1e-6)


def test_pit_copies_whole_seed_copies_and_perturbs_only_the_copier():
    search = {MULTIPLIER: (0.01, 100.0), "entropy_coefficient": (1e-4, 10.0)}
    podracer, state = prepared(search)
    container = podracer.algorithm
    before = read_hyperparameters(container, state.algorithm_state)

    after = podracer.pit(state).algorithm_state

    copies = after.algorithm_state.algorithm_states
    originals = state.algorithm_state.algorithm_state.algorithm_states
    for seed in range(2):
        for leaf, source in zip(
            jax.tree.leaves(copies[seed].params), jax.tree.leaves(originals[6 + seed].params)
        ):
            np.testing.assert_array_equal(leaf, source)
        moments, _ = copies[seed].algorithm_state.optimizer_state
        source, _ = originals[6 + seed].algorithm_state.optimizer_state
        for leaf, origin in zip(jax.tree.leaves(moments), jax.tree.leaves(source)):
            np.testing.assert_array_equal(leaf, origin)
    for index in range(2, 8):
        for leaf, source in zip(jax.tree.leaves(copies[index]), jax.tree.leaves(originals[index])):
            np.testing.assert_array_equal(leaf, source)

    for name, values in read_hyperparameters(container, after).items():
        ratio = float(values[0, 0] / before[name][3, 0])
        assert ratio == pytest.approx(0.5) or ratio == pytest.approx(2.0)
        np.testing.assert_array_equal(values[0, 0], values[0, 1])
        np.testing.assert_array_equal(values[1:], before[name][1:])
    np.testing.assert_array_equal(after.filled, [0, 0, 128, 128, 128, 128, 128, 128])
    np.testing.assert_array_equal(after.replaced, [1, 0, 0, 0])
    assert int(after.handled) == int(after.updates)


def test_pit_exploits_only_once_per_interval():
    podracer, state = prepared({MULTIPLIER: (0.5, 1.0)})
    handled = state.algorithm_state.replace(handled=state.algorithm_state.updates)
    after = podracer.pit(state.replace(algorithm_state=handled)).algorithm_state
    np.testing.assert_array_equal(after.replaced, 0)

    podracer, state = prepared({MULTIPLIER: (0.5, 1.0)}, interval=3)
    early = state.algorithm_state.replace(updates=jnp.array(2), handled=jnp.array(1))
    after = podracer.pit(state.replace(algorithm_state=early)).algorithm_state
    np.testing.assert_array_equal(after.replaced, 0)


def test_initial_values_are_log_uniform_and_shared_within_a_member():
    search = {MULTIPLIER: (0.1, 10.0), "entropy_coefficient": (0.001, 0.1)}
    podracer = zoo.ppo(match(), num_envs=8, podracer=pbt(search=search))
    state = podracer.init(jax.random.key(0)).algorithm_state
    for name, (low, high) in search.items():
        values = np.asarray(read_hyperparameters(podracer.algorithm, state)[name])
        np.testing.assert_array_equal(values[:, 0], values[:, 1])
        assert np.all((values >= low) & (values <= high))
        assert len(np.unique(values[:, 0])) == 4


def test_running_returns_shard_with_the_environments():
    podracer = zoo.ppo(
        match(), num_envs=16, podracer=pbt(interval=1, window=4, threshold=0.0, devices=4)
    )
    state = podracer.init(jax.random.key(0))
    state, logs = podracer.train(state, jax.random.key(1), 3)
    assert state.algorithm_state.running.sharding.spec == P("data")
    assert np.asarray(logs[f"pbt/member_0/{MULTIPLIER}"]).shape[0] == 3


def test_evaluation_neither_exploits_nor_logs_pbt():
    podracer = zoo.ppo(match(), num_envs=8, podracer=pbt(interval=1, window=4, threshold=0.0))
    state = podracer.init(jax.random.key(0))
    state, _ = podracer.train(state, jax.random.key(1), 3)
    evaluated, logs = podracer.evaluate(state, jax.random.key(2), 4)
    assert not any(name.startswith("pbt/") for name in logs)
    np.testing.assert_array_equal(evaluated.algorithm_state.replaced, state.algorithm_state.replaced)


def test_the_leaderboard_reports_the_leading_members_return():
    podracer = zoo.ppo(match(), num_envs=8, podracer=pbt())
    state = podracer.init(jax.random.key(0))
    window = jnp.array([0.0, 0.1, 0.5, 0.6, 1.0, 1.1, 0.4, 0.5]).reshape(8, 1)
    state = state.replace(
        algorithm_state=state.algorithm_state.replace(
            window=jnp.broadcast_to(window, (8, 128)), filled=jnp.full(8, 128, jnp.int32)
        )
    )
    returns = np.tile(np.array([0, 0, 1, 1, 5, 5, 2, 2], np.float32), (3, 1))
    returns[1] = np.nan
    logs = {"episode_statistics/episode_return": returns.reshape(-1, 1)}

    metrics = Leaderboard().craft(podracer.algorithm, podracer.environment, state, logs).data

    assert metrics["leaderboard/best_return"] == 5.0
    assert metrics["leaderboard/mean"] == pytest.approx(2.0)
    assert metrics["leaderboard/spread"] == pytest.approx(np.std([0, 1, 5, 2]))


def test_pbt_raises_a_learning_rate_too_small_to_solve_the_task():
    environment = reach()
    optimizer = zoo.adam(3e-5)

    search = {MULTIPLIER: (0.5, 100.0)}
    starts = jnp.broadcast_to(jnp.array([[0.55], [0.7], [0.85], [1.0]]), (4, 2))

    def run(podracer):
        state = podracer.init(jax.random.key(1))
        state = state.replace(
            algorithm_state=write_hyperparameters(state.algorithm_state, {MULTIPLIER: starts})
        )
        state, _ = podracer.train(state, jax.random.key(2), 150)
        _, logs = podracer.evaluate(state, jax.random.key(2), 24)
        return state, logs

    fixed = zoo.ppo(
        environment, num_envs=64, optimizer=optimizer, podracer=pbt(interval=1000, search=search)
    )
    _, logs = run(fixed)
    assert np.nanmean(logs["episode_statistics/episode_return"]) < environment.solved

    trained = zoo.ppo(environment, num_envs=64, optimizer=optimizer, podracer=pbt(search=search))
    state, logs = run(trained)
    assert np.nanmean(logs["episode_statistics/episode_return"]) >= environment.solved
    multipliers = read_hyperparameters(trained.algorithm, state.algorithm_state)[MULTIPLIER]
    assert float(multipliers.min()) > 1.0
    metrics = Leaderboard().craft(trained.algorithm, trained.environment, state, logs).data
    assert metrics["leaderboard/best_return"] >= environment.solved
