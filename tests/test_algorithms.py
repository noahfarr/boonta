import dataclasses
import re
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import PartitionSpec as P

import zoo
from boonta.algorithms.advantage_estimators import generalized_advantage_estimation
from boonta.algorithms.wrappers.population import Population
from boonta.utils import Timestep, Transition
from dummies import (Team, corridor, demonstrations, match, reach, recall,
                     recall_continuous, tallied)

LEARNERS = [
    pytest.param(zoo.ppo, corridor, 100, id="ppo-corridor"),
    pytest.param(zoo.ppo, reach, 100, id="ppo-reach"),
    pytest.param(zoo.mmd, corridor, 100, id="mmd-corridor"),
    pytest.param(zoo.grpo, corridor, 100, id="grpo-corridor"),
    pytest.param(zoo.pqn, corridor, 100, id="pqn-corridor"),
    pytest.param(zoo.dqn, corridor, 400, id="dqn-corridor"),
    pytest.param(zoo.sac, reach, 1000, id="sac-reach"),
    pytest.param(zoo.reppo, reach, 300, id="reppo-reach"),
    pytest.param(zoo.recurrent_ppo, recall, 100, id="recurrent_ppo-recall"),
    pytest.param(zoo.recurrent_pupo, recall, 100, id="recurrent_pupo-recall"),
    pytest.param(zoo.recurrent_grpo, recall, 100, id="recurrent_grpo-recall"),
    pytest.param(zoo.recurrent_pqn, recall, 100, id="recurrent_pqn-recall"),
    pytest.param(zoo.recurrent_dqn, recall, 400, id="recurrent_dqn-recall"),
    pytest.param(zoo.recurrent_sac, recall_continuous, 1000, id="recurrent_sac-recall"),
    pytest.param(zoo.bc, corridor, 200, id="bc-corridor"),
    pytest.param(zoo.iql, reach, 300, id="iql-reach"),
    pytest.param(zoo.recurrent_bc, recall, 200, id="recurrent_bc-recall"),
]


def team(environment):
    return lambda: Team(environment())


TEAMS = [
    pytest.param(zoo.ppo, team(match), 100, id="ppo-match"),
    pytest.param(zoo.mmd, team(match), 100, id="mmd-match"),
    pytest.param(zoo.recurrent_ppo, team(recall), 100, id="recurrent_ppo-recall"),
    pytest.param(zoo.recurrent_pupo, team(recall), 100, id="recurrent_pupo-recall"),
]


@pytest.mark.parametrize("build, environment, num_updates", TEAMS)
def test_learns_with_several_agents_and_counts_each_of_their_steps(build, environment, num_updates):
    environment = environment()
    podracer = build(environment)

    state = podracer.init(jax.random.key(0))
    state, _ = podracer.train(state, jax.random.key(1), num_updates)
    assert int(state.algorithm_state.step) == num_updates * podracer.batch_size
    assert podracer.batch_size % environment.num_agents == 0
    _, logs = podracer.evaluate(state, jax.random.key(2), 24)

    returns = np.asarray(logs["episode_statistics/episode_return"])
    assert np.nanmean(returns) >= environment.solved


@pytest.mark.parametrize("build, environment, num_updates", LEARNERS)
def test_learns(build, environment, num_updates):
    environment = environment()
    podracer = build(environment)

    state = podracer.init(jax.random.key(0))
    state, _ = podracer.train(state, jax.random.key(1), num_updates)
    _, logs = podracer.evaluate(state, jax.random.key(2), 24)
    podracer.close(state)

    returns = np.asarray(logs["episode_statistics/episode_return"])
    assert np.nanmean(returns) >= environment.solved


RECURRENT = [
    pytest.param(zoo.recurrent_ppo, recall, id="recurrent_ppo"),
    pytest.param(zoo.recurrent_pupo, recall, id="recurrent_pupo"),
    pytest.param(zoo.recurrent_grpo, recall, id="recurrent_grpo"),
    pytest.param(zoo.recurrent_pqn, recall, id="recurrent_pqn"),
    pytest.param(zoo.recurrent_dqn, recall, id="recurrent_dqn"),
    pytest.param(zoo.recurrent_sac, recall_continuous, id="recurrent_sac"),
]


@pytest.mark.parametrize("build, environment", RECURRENT)
def test_every_recurrent_carry_shards_with_its_environments(build, environment):
    podracer = build(environment(), num_envs=8, podracer=partial(zoo.online, devices=4))
    state = podracer.init(jax.random.key(0))
    state, _ = podracer.train(state, jax.random.key(1), 1)

    carries = [
        getattr(state.algorithm_state, name)
        for name in ("carry", "rollout_carry")
        if hasattr(state.algorithm_state, name)
    ]
    for leaf in jax.tree.leaves(carries):
        assert leaf.sharding.spec == P("data")


def reference_advantages(reward, values, bootstrap, terminated, truncated, importance, trace):
    gamma, gae_lambda = 0.9, 0.8
    advantages = np.zeros_like(values)
    advantage, next_value = np.zeros_like(bootstrap), bootstrap
    for index in reversed(range(len(values))):
        alive = (1.0 - terminated[index]) * (1.0 - truncated[index])
        delta = importance[index] * (
            reward[index] + gamma * (1.0 - terminated[index]) * next_value - values[index]
        )
        delta = delta * (1.0 - truncated[index])
        advantage = delta + gamma * gae_lambda * trace[index] * alive * advantage
        advantages[index] = advantage
        next_value = values[index]
    return advantages


@pytest.mark.parametrize("weighted", [False, True], ids=["gae", "vtrace"])
def test_generalized_advantage_estimation_follows_the_recursion(weighted):
    generator = np.random.default_rng(0)
    shape = (6, 3)
    reward, values = generator.normal(size=shape), generator.normal(size=shape)
    bootstrap = generator.normal(size=shape[1:])
    terminated = generator.random(shape) < 0.2
    truncated = (generator.random(shape) < 0.2) & ~terminated
    importance, trace = (
        generator.uniform(0.5, 1.5, shape) if weighted else np.ones(shape)
        for _ in range(2)
    )
    timestep = Timestep(
        obs=jnp.zeros(shape),
        action=jnp.zeros(shape),
        reward=jnp.asarray(reward, jnp.float32),
        terminated=jnp.asarray(terminated),
        truncated=jnp.asarray(truncated),
    )

    advantages, returns = generalized_advantage_estimation(
        Transition(first=timestep, second=timestep),
        jnp.asarray(values, jnp.float32),
        jnp.asarray(bootstrap, jnp.float32),
        0.9,
        0.8,
        importance=jnp.asarray(importance, jnp.float32),
        trace=jnp.asarray(trace, jnp.float32),
    )

    expected = reference_advantages(
        reward, values, bootstrap, terminated, truncated, importance, trace
    )
    np.testing.assert_allclose(advantages, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(returns, expected + values, rtol=1e-5, atol=1e-6)


def test_generalized_advantage_estimation_accumulates_a_bf16_critic_in_fp32():
    shape = (4, 2)
    timestep = Timestep(
        obs=jnp.zeros(shape),
        action=jnp.zeros(shape),
        reward=jnp.ones(shape, jnp.bfloat16),
        terminated=jnp.zeros(shape, bool),
        truncated=jnp.zeros(shape, bool),
    )
    advantages, returns = generalized_advantage_estimation(
        Transition(first=timestep, second=timestep),
        jnp.ones(shape, jnp.bfloat16),
        jnp.ones(shape[1:], jnp.bfloat16),
        0.99,
        0.95,
    )
    assert advantages.dtype == returns.dtype == jnp.float32


COLLECTIVE = re.compile(r"(all-gather|all-reduce|collective-permute|all-to-all)\(")
SHAPE = re.compile(r"\[([0-9,]*)\]")

REPLAYS = [
    pytest.param(zoo.dqn, corridor, id="dqn"),
    pytest.param(zoo.sac, reach, id="sac"),
    pytest.param(zoo.recurrent_dqn, recall, id="recurrent_dqn"),
    pytest.param(zoo.recurrent_sac, recall_continuous, id="recurrent_sac"),
]


@pytest.mark.parametrize("build, environment", REPLAYS)
def test_a_replay_buffer_shards_with_its_environments(build, environment):
    num_envs, devices = 8, 4
    podracer = build(
        environment(), num_envs=num_envs, podracer=partial(zoo.online, devices=devices)
    )
    state = podracer.init(jax.random.key(0))
    state, _ = podracer.train(state, jax.random.key(1), 40)

    experience = state.algorithm_state.buffer_state.experience
    for leaf in jax.tree.leaves(experience):
        assert leaf.sharding.spec == P("data")
    leaf, *_ = jax.tree.leaves(experience)
    _, time, *_ = leaf.shape

    text = podracer.train.lower(state, jax.random.key(1), 4).compile().as_text()
    moved = [
        dimensions.split(",")
        for line in text.splitlines()
        if (collective := COLLECTIVE.search(line))
        for dimensions in SHAPE.findall(line[: collective.start()].split("=", 1)[-1])
    ]
    assert moved, "the sampled batch should cross devices"
    assert not [dimensions for dimensions in moved if str(time) in dimensions]


def rollout(environment, boundary, perturbed, num_steps=4, num_envs=4):
    shape = environment.observation_space().shape
    space = environment.action_space()
    obs = jax.random.normal(jax.random.key(0), (num_steps + 1, num_envs, *shape))
    reward = jnp.broadcast_to(1.0 + jnp.arange(num_envs) / 2, (num_steps, num_envs))
    if perturbed:
        obs = obs.at[3:].set(-obs[3:])
        reward = reward.at[2:].set(50.0)
    action = jnp.zeros((num_steps, num_envs, *space.shape), space.dtype)
    flags = jnp.zeros((num_steps, num_envs), bool)
    terminated = flags.at[1].set(boundary == "terminated")
    truncated = flags.at[1].set(boundary == "truncated")
    previous = jnp.concatenate([jnp.zeros_like(reward[:1]), reward[:-1]])
    starts = jnp.concatenate([jnp.ones_like(flags[:1]), terminated[:-1]])
    return Transition(
        first=Timestep(
            obs=obs[:-1],
            action=action,
            reward=previous,
            terminated=starts,
            truncated=jnp.concatenate([flags[:1], truncated[:-1]]),
        ),
        second=Timestep(
            obs=obs[1:], action=action, reward=reward, terminated=terminated, truncated=truncated
        ),
        aux={
            "log_prob": jnp.full((num_steps, num_envs), -0.7),
            "value": jnp.zeros((num_steps, num_envs)),
        },
    )


def targets(build, environment, key, boundary, perturbed):
    seen = {}

    def store(obs, values):
        seen.update(zip(np.asarray(obs).ravel().tolist(), np.asarray(values).ravel()))

    def record(transitions, **kwargs):
        jax.debug.callback(store, transitions.first.obs[..., 0], transitions.aux[key])
        return 0.0

    algorithm = build(
        environment, num_envs=4, optimizer=optax.sgd(0.0), auxiliary_losses=(record,)
    ).algorithm
    transitions = rollout(environment, boundary, perturbed)
    first = jax.tree.map(lambda leaf: leaf[0], transitions.first)
    state = algorithm.init(jax.random.key(1), first)
    jax.block_until_ready(algorithm.update(state, jax.random.key(2), transitions))
    jax.effects_barrier()
    trained = 2 if boundary == "terminated" else 1
    before = np.asarray(transitions.first.obs[:trained, ..., 0]).ravel().tolist()
    return np.array([seen[obs] for obs in before])


BOUNDED = [
    pytest.param(zoo.ppo, corridor, "advantages", id="ppo"),
    pytest.param(zoo.mmd, corridor, "advantages", id="mmd"),
    pytest.param(zoo.grpo, corridor, "advantages", id="grpo"),
    pytest.param(zoo.reppo, reach, "target_values", id="reppo"),
    pytest.param(zoo.recurrent_ppo, corridor, "advantages", id="recurrent_ppo"),
    pytest.param(zoo.recurrent_grpo, corridor, "advantages", id="recurrent_grpo"),
]


@pytest.mark.parametrize("boundary", ["terminated", "truncated"])
@pytest.mark.parametrize("build, environment, key", BOUNDED)
def test_targets_never_look_past_an_episode_boundary(build, environment, key, boundary):
    calm = targets(build, environment(), key, boundary, perturbed=False)
    shifted = targets(build, environment(), key, boundary, perturbed=True)
    assert np.abs(calm).max() > 0
    np.testing.assert_allclose(calm, shifted, rtol=1e-6)


@pytest.mark.parametrize("weight, moves", [(0.0, False), (1.0, True)], ids=["zero", "one"])
def test_recurrent_bc_learns_only_from_weighted_steps(weight, moves):
    environment = recall()
    algorithm = zoo.recurrent_bc(environment).algorithm
    episodes = demonstrations(environment, jax.random.key(0), 32)
    batch = episodes.replace(aux={"weight": jnp.full(episodes.second.reward.shape, weight)})
    state = algorithm.init(jax.random.key(1), jax.tree.map(lambda leaf: leaf[:, 0], batch.first))
    updated = algorithm.update(state, jax.random.key(2), batch)

    changed = [
        not np.array_equal(before, after)
        for before, after in zip(jax.tree.leaves(state.params), jax.tree.leaves(updated.params))
    ]
    assert any(changed) == moves


def log_prob_gap(transitions, dist, **kwargs):
    return jnp.abs(dist.log_prob(transitions.second.action) - transitions.aux["log_prob"])


def q_value_gap(transitions, q_values, **kwargs):
    return jnp.abs(q_values - transitions.aux["q_values"])


@pytest.fixture
def gaps():
    recorded = []

    def record(gap):
        def loss(**kwargs):
            jax.debug.callback(
                lambda value: recorded.append(float(value)), gap(**kwargs).max()
            )
            return 0.0

        return loss

    yield recorded, record
    jax.effects_barrier()


PODRACERS = [
    pytest.param(partial(zoo.online, devices=2), id="anakin"),
    pytest.param(zoo.asynchronous, id="sebulba"),
]

REPLAYERS = [
    pytest.param(zoo.ppo, corridor, log_prob_gap, id="ppo"),
    pytest.param(zoo.mmd, corridor, log_prob_gap, id="mmd"),
    pytest.param(zoo.grpo, corridor, log_prob_gap, id="grpo"),
    pytest.param(zoo.pqn, corridor, q_value_gap, id="pqn"),
    pytest.param(zoo.recurrent_ppo, recall, log_prob_gap, id="recurrent_ppo"),
    pytest.param(zoo.recurrent_pupo, recall, log_prob_gap, id="recurrent_pupo"),
    pytest.param(zoo.recurrent_grpo, recall, log_prob_gap, id="recurrent_grpo"),
    pytest.param(zoo.recurrent_pqn, recall, q_value_gap, id="recurrent_pqn"),
]


def replay(gaps, build, **kwargs):
    recorded, record = gaps
    podracer = build(
        num_envs=8, num_steps=4, optimizer=optax.sgd(0.0), auxiliary_losses=(record,), **kwargs
    )
    state = podracer.init(jax.random.key(0))
    for epoch in range(2):
        state, _ = podracer.train(state, jax.random.key(epoch + 1), 3)
    jax.effects_barrier()
    return recorded


@pytest.mark.parametrize("podracer", PODRACERS)
@pytest.mark.parametrize("build, environment, gap", REPLAYERS)
def test_every_update_replays_what_its_rollout_acted_on(
    gaps, build, environment, gap, podracer
):
    recorded, record = gaps
    recorded = replay(
        (recorded, record(gap)), partial(build, environment()), podracer=podracer
    )
    assert recorded
    np.testing.assert_allclose(recorded, 0.0, atol=1e-5)


@pytest.mark.parametrize("podracer", PODRACERS)
@pytest.mark.parametrize("torso", zoo.TORSOS.values(), ids=zoo.TORSOS)
def test_every_torso_replays_what_it_acted_on(gaps, torso, podracer):
    recorded, record = gaps
    recorded = replay(
        (recorded, record(log_prob_gap)),
        partial(zoo.recurrent_ppo, recall()),
        podracer=podracer,
        torso=torso(),
    )
    assert recorded
    np.testing.assert_allclose(recorded, 0.0, atol=1e-5)


def populated(algorithm, *args):
    return zoo.online(Population(algorithm=algorithm, count=2), *args)


STATEFUL = [
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
    pytest.param(zoo.bc, corridor, id="bc"),
    pytest.param(zoo.iql, reach, id="iql"),
    pytest.param(zoo.recurrent_bc, recall, id="recurrent_bc"),
    pytest.param(partial(zoo.ppo, podracer=populated), corridor, id="population-ppo"),
    pytest.param(
        partial(zoo.recurrent_ppo, podracer=populated), recall, id="population-recurrent_ppo"
    ),
]


def trained(algorithm_state):
    if hasattr(algorithm_state, "algorithm_states"):
        for inner in algorithm_state.algorithm_states:
            yield from trained(inner)
        return
    names = {field.name for field in dataclasses.fields(algorithm_state)}
    for name in sorted(names):
        optimized = name.removesuffix("params") + "optimizer_state" in names
        if name == "params" or (name.endswith("_params") and optimized):
            yield name, getattr(algorithm_state, name)


def counts(algorithm_state):
    return [
        np.asarray(leaf)
        for _, variables in trained(algorithm_state)
        for leaf in jax.tree.leaves(variables["counts"])
    ]


def paths(tree):
    return [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(tree)]


@pytest.mark.parametrize("build, environment", STATEFUL)
def test_every_algorithm_carries_what_its_layers_write_during_the_update(
    build, environment, monkeypatch
):
    monkeypatch.setattr(zoo, "FeatureExtractor", tallied(zoo.FeatureExtractor))
    monkeypatch.setattr(zoo, "Parameter", tallied(zoo.Parameter))
    podracer = build(environment())
    initial = podracer.init(jax.random.key(0))
    assert all(int(count) == 0 for count in counts(initial.algorithm_state))

    state, _ = podracer.train(initial, jax.random.key(1), 10)
    advanced = counts(state.algorithm_state)
    assert advanced
    assert all(int(count) > 0 for count in advanced)
    assert not [path for path in paths(state.algorithm_state) if "intermediates" in path]
    assert jax.tree.structure(state.algorithm_state) == jax.tree.structure(
        initial.algorithm_state
    )
    assert not [
        path
        for path in paths(state.algorithm_state)
        if "optimizer_state" in path and "counts" in path
    ]

    stepped, _, _ = podracer.algorithm.step(
        state.algorithm_state, jax.random.key(2), state.timestep
    )
    evaluated, _ = podracer.evaluate(state, jax.random.key(3), 8)
    for after in (stepped, evaluated.algorithm_state):
        for before, now in zip(advanced, counts(after), strict=True):
            np.testing.assert_array_equal(before, now)
