from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import trueskill
from flax import struct
from hydra.utils import instantiate
from omegaconf import OmegaConf

from boonta.algorithms.wrappers.ensemble import Ensemble
from boonta.algorithms.wrappers.psro import (PSRO, Learner, Member, Meta,
                                             always, either, even, forgotten,
                                             fsp, hard, ladder, latest,
                                             mixture, nash, periodic, pfsp,
                                             pfsp_vs_lineage, self_play,
                                             threshold)
from boonta.algorithms.wrappers.wrapper import Wrapper
from boonta.artisans.trueskill import RATER, TrueSkill, conservative, rate
from boonta.curricula import league
from boonta.environments.wrappers import Opponent
from boonta.podracers import anakin
from boonta.utils import Timestep, Transition, mesh

DEFAULT = Path(__file__).parents[1] / "hangar/config/curriculum/default.yaml"
DECKS = ["alakazam", "thwackey"]

LEAGUE = """
_target_: boonta.curricula.league
_partial_: true
capacity: 1024
decay: 0.995
assignments:
  exploiter: alakazam
population:
  - {lineage: alakazam, checkpoint: alakazam.msgpack}
  - {lineage: thwackey, checkpoint: thwackey.msgpack}
learners:
  - _target_: boonta.algorithms.wrappers.psro.Learner
    lineage: alakazam
    warmstart: alakazam.msgpack
    solver:
      _target_: boonta.algorithms.wrappers.psro.mixture
      weights: [0.5, 0.5]
      solvers:
        - _target_: boonta.algorithms.wrappers.psro.self_play
        - _target_: boonta.algorithms.wrappers.psro.pfsp
          weighting:
            _target_: boonta.algorithms.wrappers.psro.hard
    admit:
      _target_: boonta.algorithms.wrappers.psro.periodic
      every: 50
  - _target_: boonta.algorithms.wrappers.psro.Learner
    lineage: thwackey
    warmstart: thwackey.msgpack
    solver:
      _target_: boonta.algorithms.wrappers.psro.self_play
    admit:
      _target_: boonta.algorithms.wrappers.psro.periodic
      every: 50
  - _target_: boonta.algorithms.wrappers.psro.Learner
    lineage: exploiter
    warmstart: alakazam.msgpack
    solver:
      _target_: boonta.algorithms.wrappers.psro.pfsp_vs_lineage
      lineage: alakazam
      current: true
      weighting:
        _target_: boonta.algorithms.wrappers.psro.hard
    admit:
      _target_: boonta.algorithms.wrappers.psro.threshold
      level: 0.7
    resets: true
"""


@struct.dataclass(frozen=True)
class MarginEnvState:
    clock: jax.Array


@dataclass
class MarginEnvironment:

    num_envs: int
    horizon: int = 3
    num_agents = 1

    def timestep(self, action, done):
        margin = jnp.tanh(action[..., 0] - action[..., 1])
        return Timestep(
            obs=jnp.zeros((self.num_envs, 1)),
            action=action,
            reward=jnp.stack([margin, -margin], axis=-1),
            terminated=jnp.stack([done, done], axis=-1),
            truncated=jnp.zeros((self.num_envs, 2), bool),
            info=None,
        )

    def init(self, key):
        clock = jnp.full((self.num_envs,), self.horizon)
        done = jnp.zeros((self.num_envs,), bool)
        return MarginEnvState(clock), self.timestep(
            jnp.zeros((self.num_envs, 2)), done
        )

    def step(self, key, state, action):
        clock = state.clock - 1
        done = clock == 0
        timestep = self.timestep(action, done)
        return MarginEnvState(jnp.where(done, self.horizon, clock)), timestep

    def close(self, state):
        return state


@struct.dataclass(frozen=True)
class SkillState:
    step: jax.Array
    params: jax.Array
    optimizer_state: optax.OptState


@dataclass
class SkillAlgorithm:

    optimizer: optax.GradientTransformation

    def init(self, key, timestep):
        params = jnp.zeros((1,))
        return SkillState(
            step=jnp.array(0),
            params=params,
            optimizer_state=self.optimizer.init(params),
        )

    def step(self, state, key, timestep, temperature=1.0):
        action = jnp.broadcast_to(state.params[0], (timestep.obs.shape[0],))
        return state, action, {}

    def loss(self, params):
        return -params[0]

    def update(self, state, key, transitions):
        grads = jax.grad(self.loss)(state.params)
        updates, optimizer_state = self.optimizer.update(
            grads, state.optimizer_state, state.params
        )
        return state.replace(
            params=optax.apply_updates(state.params, updates),
            optimizer_state=optimizer_state,
        )


@dataclass
class Anchorable(SkillAlgorithm):

    auxiliary_losses: tuple = ()

    def loss(self, params):
        def logits(params):
            return SimpleNamespace(
                logits=jax.nn.log_softmax(jnp.concatenate([params, jnp.zeros(1)]))
            )

        return -params[0] + sum(
            auxiliary(dist=logits(params), apply=logits)
            for auxiliary in self.auxiliary_losses
        )


@dataclass
class Thermal(SkillAlgorithm):
    def step(self, state, key, timestep, temperature=1.0):
        return state, jnp.full((timestep.obs.shape[0],), temperature), {}


@dataclass
class Recording(SkillAlgorithm):

    seen: list

    def update(self, state, key, transitions):
        seat = transitions.second
        self.seen.append((seat.action.shape, seat.reward.shape, seat.terminated.shape))
        return super().update(state, key, transitions)


@dataclass
class ToyAlgorithm:

    optimizer: optax.GradientTransformation

    def init(self, key, timestep):
        params = jnp.zeros((1,))
        return SkillState(
            step=jnp.array(0),
            params=params,
            optimizer_state=self.optimizer.init(params),
        )

    def step(self, state, key, timestep, temperature=1.0):
        return state, -timestep.obs[:, 0] * state.params[0], {}

    def update(self, state, key, transitions):
        def loss(params):
            prediction = transitions.first.obs[..., 0] * params[0]
            return jnp.mean((prediction - transitions.second.reward) ** 2)

        updates, optimizer_state = self.optimizer.update(
            jax.grad(loss)(state.params), state.optimizer_state, state.params
        )
        return state.replace(
            params=optax.apply_updates(state.params, updates),
            optimizer_state=optimizer_state,
        )


class Tiny(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(3)(x)


def skill_play(carry, params, timestep, key, temperature):
    return carry, jnp.broadcast_to(params[0], timestep.obs.shape[:1])


def thermal_play(carry, params, timestep, key, temperature):
    return carry, jnp.full(timestep.obs.shape[:1], temperature)


def counting_play(carry, params, timestep, key, temperature):
    return carry + 1.0, jnp.broadcast_to(params[0], timestep.obs.shape[:1])


def skill_initialize(key, timestep):
    return jnp.zeros((1,)), None


def counting_initialize(key, timestep):
    return jnp.zeros((1,)), jnp.zeros(timestep.obs.shape[:1])


def entry(lineage="main", solver=None, admit=None, **changes):
    return Learner(
        lineage=lineage,
        solver=solver or self_play(),
        admit=admit or periodic(2),
        **changes,
    )


def build(
    learners,
    num_envs=8,
    capacity=8,
    population=(),
    oracle=None,
    play=skill_play,
    initialize=skill_initialize,
    **extras,
):
    config = anakin.AnakinConfig(num_envs=num_envs, num_steps=4, mesh=mesh(2))
    algorithm, environment, pit, lap = league(
        oracle or SkillAlgorithm(optimizer=optax.sgd(1.0)),
        MarginEnvironment(num_envs=num_envs),
        learners=learners,
        play=play,
        initialize=initialize,
        capacity=capacity,
        decay=0.9,
        population=[
            SimpleNamespace(lineage=member.lineage, checkpoint=member.params)
            for member in population
        ],
        **extras,
    )
    return config, anakin.make(config, algorithm, environment, pit=pit, lap=lap)


def slices(num_envs):
    obs = jnp.concatenate([jnp.ones(num_envs // 2), -jnp.ones(num_envs // 2)])[:, None]
    return Timestep(
        obs=obs,
        reward=jnp.zeros(num_envs),
        terminated=jnp.zeros(num_envs, bool),
        truncated=jnp.zeros(num_envs, bool),
    )


def meta(capacity=4, **changes):
    base = dict(
        payoff=jnp.zeros((capacity, capacity)),
        counts=jnp.zeros((capacity, capacity)),
        lineages=jnp.array([0, 0, -1, -1]),
        members=jnp.array([1.0, 1.0, 0.0, 0.0]),
        learners=1,
        index=0,
        iteration=jnp.array(0),
        names=("main",),
    )
    return Meta(**(base | changes))


def record(wins, games):
    return dict(
        payoff=jnp.zeros((4, 4)).at[0, 1].set(wins),
        counts=jnp.zeros((4, 4)).at[0, 1].set(games),
    )


def divide_ensemble():
    Ensemble(ToyAlgorithm(optax.sgd(0.1)), count=3).init(jax.random.key(0), slices(8))


def divide_league():
    build([entry("main"), entry("exploiter", latest()), entry("rival", fsp())])


def test_a_league_trains_its_learners_and_keeps_the_meta_game():
    learners = [
        entry("main", latest()),
        entry("exploiter", pfsp(hard()), periodic(3), resets=True),
    ]
    config, podracer = build(learners)
    state = podracer.init(jax.random.key(0))

    algorithm_state = state.algorithm_state
    assert algorithm_state.population.shape == (8, 1)
    assert algorithm_state.payoff.shape == (8, 8)
    assert int(algorithm_state.members.sum()) == int(algorithm_state.cursor) == 2
    assert len(algorithm_state.learners) == 2

    updates = 6
    state, logs = podracer.train(state, jax.random.key(1), updates)

    algorithm_state = state.algorithm_state
    main, exploiter = algorithm_state.algorithm_states
    assert float(main.params[0]) == updates
    assert float(exploiter.params[0]) == 0.0
    cursor = int(algorithm_state.cursor)
    assert 2 < cursor <= 8
    assert int(algorithm_state.members.sum()) == cursor
    payoff = np.asarray(algorithm_state.payoff)
    counts = np.asarray(algorithm_state.counts)
    np.testing.assert_allclose(payoff, -payoff.T, atol=1e-5)
    np.testing.assert_allclose(counts, counts.T, atol=1e-5)
    assert np.all(np.isfinite(payoff))
    assert counts.sum() > 0
    assert np.asarray(logs["learner_1/admitted"]).sum() > 0
    podracer.close(state)


def test_every_learner_reads_the_step_count_of_the_whole_run():
    _, podracer = build([entry("main"), entry("exploiter", latest())])
    state = podracer.init(jax.random.key(0))
    state, _ = podracer.train(state, jax.random.key(1), 2)

    assert int(state.algorithm_state.step) == 2 * podracer.batch_size
    for inner in state.algorithm_state.algorithm_states:
        assert int(inner.step) == 2 * podracer.batch_size
    podracer.close(state)


def test_league_state_is_replicated_while_returns_and_rivals_follow_the_environments(
    assert_sharded, assert_replicated
):
    config, podracer = build([entry("main"), entry("exploiter", latest())])
    state = podracer.init(jax.random.key(0))
    state, _ = podracer.train(state, jax.random.key(1), 1)

    algorithm_state = state.algorithm_state
    for leaf in (
        algorithm_state.population,
        algorithm_state.payoff,
        algorithm_state.counts,
        algorithm_state.members,
    ):
        assert_replicated(leaf, config.mesh)
    for inner in algorithm_state.algorithm_states:
        assert_replicated(inner.params, config.mesh)
    for learner in algorithm_state.learners:
        assert_sharded(learner.returns, config.mesh)
    assert_replicated(state.environment_state.params, config.mesh)
    podracer.close(state)


def test_the_population_mirrors_the_current_policy_after_every_update():
    config, podracer = build([entry("main", admit=periodic(100))])
    state = podracer.init(jax.random.key(0))

    for updates in (1, 2, 3):
        state, _ = podracer.train(state, jax.random.key(updates), 1)
        (inner,) = state.algorithm_state.algorithm_states
        params = float(inner.params[0])
        assert params == updates
        np.testing.assert_allclose(
            float(np.asarray(state.algorithm_state.population)[0, 0]), params, atol=1e-2
        )
    podracer.close(state)


def test_a_restart_reads_the_population_as_it_stood_before_the_update():
    learners = [
        entry("main"),
        entry(
            "exploiter",
            pfsp(hard()),
            periodic(3),
            resets=True,
            restart=pfsp_vs_lineage("main", hard(), current=True),
        ),
    ]
    config, podracer = build(learners)
    state = podracer.init(jax.random.key(0))
    state, _ = podracer.train(state, jax.random.key(1), 6)

    main, exploiter = state.algorithm_state.algorithm_states
    assert float(main.params[0]) == 6.0
    assert float(exploiter.params[0]) == 5.0
    podracer.close(state)


def test_the_oracle_is_handed_its_own_seat():
    seen = []
    config, podracer = build([entry()], oracle=Recording(optax.sgd(1.0), seen))
    state = podracer.init(jax.random.key(0))
    state, _ = podracer.train(state, jax.random.key(1), 1)

    shape = (config.num_steps, config.num_envs)
    assert set(seen) == {(shape, shape, shape)}
    podracer.close(state)


def test_a_new_rival_starts_from_a_blank_carry():
    config, podracer = build(
        [entry()], play=counting_play, initialize=counting_initialize
    )

    state = podracer.init(jax.random.key(0))
    np.testing.assert_allclose(np.asarray(state.environment_state.carry), 0.0)

    state, _ = podracer.evaluate(state, jax.random.key(1), 3)
    np.testing.assert_allclose(np.asarray(state.environment_state.carry), 3.0)

    state, _ = podracer.train(state, jax.random.key(2), 1)
    np.testing.assert_allclose(np.asarray(state.environment_state.carry), 0.0)
    podracer.close(state)


def test_the_rival_samples_at_full_temperature_while_only_the_learner_is_greedy():
    config, podracer = build(
        [entry()], oracle=Thermal(optax.sgd(1.0)), play=thermal_play
    )
    state = podracer.init(jax.random.key(0))

    state, _ = podracer.evaluate(state, jax.random.key(1), 3)
    learner_action, rival_action = np.asarray(state.environment_state.joint.action).T
    np.testing.assert_allclose(learner_action, 0.0)
    np.testing.assert_allclose(rival_action, 1.0)

    state, _ = podracer.train(state, jax.random.key(2), 1)
    learner_action, rival_action = np.asarray(state.environment_state.joint.action).T
    np.testing.assert_allclose(learner_action, 1.0)
    np.testing.assert_allclose(rival_action, 1.0)
    podracer.close(state)


def test_the_pit_seats_the_drawn_opponents_before_the_deal():
    def deal(environment, learners, lineages, **table):
        def dress(state):
            seated = state.environment_state
            return state.replace(
                environment_state=seated.replace(params=seated.params + 100.0)
            )

        return dress

    config, podracer = build([entry(), entry("exploiter")], deal=deal)
    state = podracer.init(jax.random.key(0))
    np.testing.assert_allclose(np.asarray(state.environment_state.params), 100.0)

    state, _ = podracer.train(state, jax.random.key(1), 1)
    np.testing.assert_allclose(np.asarray(state.environment_state.params), 101.0)
    podracer.close(state)


def test_the_initial_population_is_never_overwritten_and_admission_stops_at_capacity():
    learners = [
        entry("main", pfsp_vs_lineage("rival", hard())),
        entry("exploiter", latest(), periodic(3), resets=True),
    ]
    population = [
        Member(lineage="rival", params=jnp.array([1.5])),
        Member(lineage="rival", params=jnp.array([0.5])),
    ]
    config, podracer = build(learners, capacity=6, population=population)

    state = podracer.init(jax.random.key(0))
    algorithm_state = state.algorithm_state
    assert int(algorithm_state.cursor) == 4
    np.testing.assert_allclose(np.asarray(algorithm_state.population)[2:4, 0], [1.5, 0.5])
    assert np.asarray(algorithm_state.lineages)[2] == 2

    state, logs = podracer.train(state, jax.random.key(1), 6)
    algorithm_state = state.algorithm_state
    assert int(algorithm_state.cursor) == 6
    assert int(algorithm_state.members.sum()) == 6
    population = np.asarray(algorithm_state.population)
    np.testing.assert_allclose(population[2:4, 0], [1.5, 0.5])
    np.testing.assert_allclose(population[4:6, 0], [2.0, 3.0])
    assert np.all(np.isfinite(np.asarray(logs["learner_0/reference"])))
    podracer.close(state)


def test_only_the_learners_that_name_a_warmstart_start_from_it():
    learners = [
        entry("main", latest(), warmstart=jnp.array([3.0])),
        entry("exploiter", latest(), periodic(3)),
    ]
    config, podracer = build(learners)

    state = podracer.init(jax.random.key(0))
    started, cold = state.algorithm_state.algorithm_states
    warm, _ = state.algorithm_state.learners

    np.testing.assert_allclose(np.asarray(started.params), [3.0])
    np.testing.assert_allclose(np.asarray(cold.params), [0.0])
    np.testing.assert_allclose(np.asarray(warm.initial_state.params), [3.0])


@pytest.mark.parametrize("divide", [divide_ensemble, divide_league])
def test_a_batch_that_does_not_divide_among_the_copies_is_rejected(divide):
    with pytest.raises(AssertionError, match="multiple of 3"):
        divide()


def test_anchoring_needs_an_oracle_that_takes_auxiliary_losses():
    learners = [entry(kl_coefficient=0.01)]
    with pytest.raises(AssertionError, match="auxiliary losses"):
        PSRO(
            algorithm=SkillAlgorithm(optimizer=optax.sgd(1.0)),
            learners=learners,
            capacity=4,
            decay=0.9,
        )


def test_a_learner_with_a_kl_coefficient_is_held_near_its_initial_parameters():
    def train(coefficient):
        config, podracer = build(
            [entry(admit=periodic(100), kl_coefficient=coefficient)],
            oracle=Anchorable(optax.sgd(1.0)),
        )
        state = podracer.init(jax.random.key(0))
        state, _ = podracer.train(state, jax.random.key(1), 6)
        (inner,) = state.algorithm_state.algorithm_states
        (learner,) = state.algorithm_state.learners
        podracer.close(state)
        return float(inner.params[0]), np.asarray(learner.initial_state.params)

    free, free_initial = train(0.0)
    held, held_initial = train(5.0)

    assert free == 6.0
    assert 0.0 < held < 2.0
    np.testing.assert_allclose(free_initial, [0.0])
    np.testing.assert_allclose(held_initial, [0.0])


def test_a_league_wraps_an_ensemble_around_the_oracle():
    oracle = SkillAlgorithm(optimizer=optax.sgd(1.0))
    wrapped = PSRO(
        algorithm=oracle,
        learners=[entry(resets=True)],
        capacity=4,
        decay=0.9,
    )
    assert isinstance(wrapped.algorithm, Ensemble)
    assert wrapped.algorithm.count == 1
    assert wrapped.oracle is oracle
    assert wrapped.optimizer is oracle.optimizer
    assert wrapped.wraps(SkillAlgorithm)
    assert not wrapped.wraps(MarginEnvironment)


def test_a_wrapper_missing_its_algorithm_raises_instead_of_recursing():
    bare = Wrapper.__new__(Wrapper)
    with pytest.raises(AttributeError):
        bare.anything


def test_each_ensemble_copy_acts_and_learns_on_its_own_slice():
    ensemble = Ensemble(ToyAlgorithm(optax.sgd(0.1)), count=2)
    timestep = slices(8)
    state = ensemble.init(jax.random.key(0), timestep)
    state = state.replace(
        algorithm_states=tuple(
            inner.replace(params=jnp.array([value]))
            for inner, value in zip(state.algorithm_states, (1.0, 2.0))
        ),
        step=jnp.array(40, state.step.dtype),
    )

    state, action, _ = ensemble.step(state, jax.random.key(1), timestep)
    np.testing.assert_allclose(np.asarray(action), [-1.0] * 4 + [2.0] * 4)
    first, second = state.algorithm_states
    assert (int(first.step), int(second.step)) == (40, 40)

    window = jax.tree.map(lambda leaf: leaf[None], timestep)
    target = jnp.concatenate([jnp.full(4, 5.0), jnp.zeros(4)])
    transitions = Transition(first=window, second=window.replace(reward=target[None]))
    updated = ensemble.update(state, jax.random.key(2), transitions)

    first, second = updated.algorithm_states
    assert float(first.params[0]) > 1.0
    assert float(second.params[0]) < 2.0


def test_a_league_state_offers_trueskill_one_subject_per_role():
    config, podracer = build([entry(), entry("exploiter", latest())])
    state = podracer.init(jax.random.key(0))

    subjects = TrueSkill(player=None, opponents={}).subjects(state)
    assert sorted(subjects) == ["role:0", "role:1"]


@pytest.mark.parametrize(
    "solver",
    [
        self_play(),
        latest(),
        fsp(),
        pfsp(hard()),
        pfsp(even()),
        nash(iterations=8),
        pfsp_vs_lineage("main", hard()),
        forgotten("main", hard()),
        mixture([fsp(), latest()], [0.3, 0.7]),
        ladder(latest(), fsp()),
    ],
    ids=[
        "self_play",
        "latest",
        "fsp",
        "pfsp-hard",
        "pfsp-even",
        "nash",
        "pfsp_vs_lineage",
        "forgotten",
        "mixture",
        "ladder",
    ],
)
def test_solvers_return_a_distribution_over_occupied_slots(solver):
    distribution = np.asarray(solver(meta(**record(2.0, 4.0))))
    assert distribution.shape == (4,)
    np.testing.assert_allclose(distribution.sum(), 1.0, atol=1e-5)
    assert np.all(distribution >= 0)
    np.testing.assert_array_equal(distribution[2:], 0.0)


@pytest.mark.parametrize(
    "solver, allowed",
    [
        (pfsp_vs_lineage("main", hard()), {0, 2}),
        (pfsp_vs_lineage("rival", hard()), {1, 3}),
        (pfsp_vs_lineage("main", hard(), current=True), {0}),
        (forgotten("rival", hard()), {1}),
    ],
    ids=["main", "rival", "current-main", "forgotten-rival"],
)
def test_lineage_solvers_pick_only_from_the_members_they_are_meant_to(solver, allowed):
    lineages = meta(
        payoff=jnp.zeros((4, 4)).at[0, 3].set(2.0),
        counts=jnp.zeros((4, 4)).at[0, 1].set(2.0).at[0, 3].set(2.0),
        lineages=jnp.array([0, 1, 0, 1]),
        members=jnp.ones(4),
        learners=2,
        names=("main", "rival"),
    )
    distribution = np.asarray(solver(lineages))
    outside = [slot for slot in range(4) if slot not in allowed]
    np.testing.assert_allclose(distribution.sum(), 1.0, atol=1e-5)
    np.testing.assert_array_equal(distribution[outside], 0.0)
    assert np.all(distribution[sorted(allowed)] > 0)


def test_a_ladder_falls_back_only_while_the_primary_is_losing():
    solver = ladder(latest(), fsp(), level=0.3)

    losing = meta(**record(-3.0, 3.0))
    np.testing.assert_allclose(np.asarray(solver(losing)), np.asarray(fsp()(losing)))

    winning = meta(**record(3.0, 3.0))
    np.testing.assert_allclose(
        np.asarray(solver(winning)), np.asarray(latest()(winning))
    )


@pytest.mark.parametrize(
    "admit, wins, games, iteration, admitted",
    [
        (always(), 0.0, 0.0, 0, True),
        (periodic(2), 0.0, 0.0, 1, True),
        (periodic(2), 0.0, 0.0, 0, False),
        (threshold(level=0.7), 4.0, 4.0, 0, True),
        (threshold(level=0.7), -4.0, 4.0, 0, False),
        (threshold(level=0.7), 0.0, 0.0, 0, False),
        (either(threshold(level=0.99), periodic(1)), -4.0, 4.0, 0, True),
        (either(threshold(level=0.99), periodic(2)), 4.0, 4.0, 0, True),
        (either(threshold(level=0.99), periodic(2)), -4.0, 4.0, 0, False),
    ],
    ids=[
        "always",
        "periodic-on-its-turn",
        "periodic-off-its-turn",
        "threshold-beating-the-pool",
        "threshold-losing-to-the-pool",
        "threshold-without-games",
        "either-by-schedule",
        "either-by-strength",
        "either-by-neither",
    ],
)
def test_admission_follows_the_schedule_and_the_win_rate(
    admit, wins, games, iteration, admitted
):
    state = meta(iteration=jnp.array(iteration), **record(wins, games))
    assert bool(admit(state)) is admitted


def test_league_composes_psro_learners_and_an_opponent_environment():
    curriculum = instantiate(OmegaConf.create(LEAGUE))
    game = SimpleNamespace(num_agents=2)
    oracle = object()
    dealt = {}

    def deal(environment, learners, lineages, assignments):
        dealt.update(
            environment=environment,
            learners=learners,
            lineages=lineages,
            assignments=assignments,
        )
        return lambda state: state

    algorithm, environment, pit, lap = curriculum(
        oracle,
        game,
        play=lambda *args: args,
        initialize=lambda key, timestep: (None, None),
        load=lambda checkpoint: checkpoint,
        deal=deal,
    )

    assert isinstance(algorithm, PSRO)
    assert isinstance(algorithm.algorithm, Ensemble)
    assert algorithm.oracle is oracle
    assert algorithm.algorithm.count == 3
    assert [learner.lineage for learner in algorithm.learners] == DECKS + ["exploiter"]
    assert [member.lineage for member in algorithm.population] == DECKS
    assert algorithm.names == (*DECKS, "exploiter")
    assert algorithm.population[0].params == "alakazam.msgpack"
    assert sorted(algorithm.warmstarts) == list(range(3))
    assert algorithm.warmstarts[2] == "alakazam.msgpack"
    assert algorithm.capacity == 1024

    assert isinstance(environment, Opponent)
    assert environment.num_agents == 1
    assert game.num_agents == 2

    assert dealt["environment"] is environment
    assert dealt["learners"] == algorithm.learners
    assert dealt["lineages"] == algorithm.names
    assert dict(dealt["assignments"]) == {"exploiter": "alakazam"}
    assert lap("state") == "state"


def test_the_default_curriculum_changes_nothing_and_ignores_the_league_extras():
    algorithm, environment, state = object(), object(), object()
    curriculum = instantiate(OmegaConf.load(DEFAULT))

    wrapped, placed, pit, lap = curriculum(
        algorithm,
        environment,
        cell=None,
        play=None,
        initialize=None,
        load=None,
        deal=None,
    )

    assert wrapped is algorithm
    assert placed is environment
    assert pit(state) is state
    assert lap(state) is state


@pytest.fixture(scope="module")
def params():
    return Tiny().init(jax.random.key(0), jnp.zeros((1, 2)))


@pytest.fixture
def artisan():
    return TrueSkill(player=None, opponents={})


def board(params):
    return SimpleNamespace(algorithm_state=SimpleNamespace(params=params))


def test_flax_still_uses_the_params_collection(params):
    assert "params" in params, (
        f"flax no longer puts parameters in a 'params' collection "
        f"(got {list(params)}); TrueSkill.subjects walks a tree looking for it"
    )


@pytest.mark.parametrize(
    "grow, names",
    [
        (lambda params: params, ["learner"]),
        (lambda params: {"role:0": params, "role:1": params}, ["role:0", "role:1"]),
        (lambda params: {"deck": {"a": params, "b": params}}, ["deck:a", "deck:b"]),
    ],
    ids=["single-policy", "one-per-role", "nesting-flattened-into-the-name"],
)
def test_each_policy_in_the_params_tree_is_one_subject(artisan, params, grow, names):
    subjects = artisan.subjects(board(grow(params)))
    assert sorted(subjects) == names
    assert all("params" in tree for tree in subjects.values())


def test_a_policy_with_other_collections_is_not_split(artisan, params):
    tree = {**params, "batch_stats": {"Dense_0": {"mean": jnp.zeros(3)}}}
    subjects = artisan.subjects(board(tree))
    assert list(subjects) == ["learner"]
    assert "batch_stats" in subjects["learner"]


@pytest.mark.parametrize("score, winner, loser", [(1.0, "a", "b"), (0.0, "b", "a")])
def test_the_score_not_the_order_decides_the_winner(score, winner, loser):
    ratings = {"a": RATER.create_rating(), "b": RATER.create_rating()}
    rate(ratings, [("a", "b", score)])
    assert ratings[winner].mu > RATER.mu > ratings[loser].mu
    assert ratings[winner].sigma < RATER.sigma


def test_a_draw_pulls_the_ratings_together():
    ratings = {"a": trueskill.Rating(mu=35.0), "b": trueskill.Rating(mu=15.0)}
    before = ratings["a"].mu - ratings["b"].mu
    rate(ratings, [("a", "b", 0.5)])
    after = ratings["a"].mu - ratings["b"].mu
    assert 0 < after < before


def test_a_draw_is_symmetric_in_argument_order():
    forward = {"a": RATER.create_rating(), "b": RATER.create_rating()}
    reverse = {"a": RATER.create_rating(), "b": RATER.create_rating()}
    rate(forward, [("a", "b", 0.5)])
    rate(reverse, [("b", "a", 0.5)])
    for name in ("a", "b"):
        assert forward[name].mu == pytest.approx(reverse[name].mu)
        assert forward[name].sigma == pytest.approx(reverse[name].sigma)


def test_repeated_wins_shrink_sigma():
    ratings = {"a": RATER.create_rating(), "b": RATER.create_rating()}
    rate(ratings, [("a", "b", 1.0)])
    once = ratings["a"].sigma
    rate(ratings, [("a", "b", 1.0)] * 20)
    assert ratings["a"].sigma < once < RATER.sigma


def test_conservative_is_three_sigma_below_the_mean():
    assert conservative(trueskill.Rating(mu=25.0, sigma=8.0)) == pytest.approx(1.0)
