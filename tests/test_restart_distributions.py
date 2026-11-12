from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import lox
import pytest
import zoo
from dummies import Corridor
from flax import struct

from boonta.curricula import erd, psr
from boonta.curricula.psr import (
    Advantage,
    CountedRelevance,
    Diagonal,
    GreedyAdvantage,
    Kernel,
    ModelAdvantage,
    Softmax,
    SuccessorFootprint,
    SuccessorRepresentation,
    TopK,
    UniformFootprint,
    UniformGain,
    UniformRelevance,
    WaterFilling,
    signal,
)
from boonta.environments.environment import Environment
from boonta.environments.spaces import Space
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.environments.wrappers import archive_auto_reset
from boonta.environments.wrappers.archive_auto_reset import (
    ArchiveAutoReset,
    ArchiveAutoResetState,
)
from boonta.podracers import anakin
from boonta.utils import Timestep, Transition, mesh

LENGTH = 6
NUM_ENVS = 8


@struct.dataclass
class Rung:
    tag: jnp.ndarray


class Ladder(Environment):
    def observe(self, state, key=None):
        return state.tag.astype(jnp.float32)[None]

    def frame(self, state, action):
        return Timestep(
            obs=self.observe(state),
            action=action,
            reward=state.tag.astype(jnp.float32),
            terminated=state.tag >= LENGTH,
            truncated=jnp.bool_(False),
            info={},
        )

    def init(self, key):
        state = Rung(jnp.int32(0))
        return state, self.frame(state, jnp.int32(0))

    def step(self, key, state, action):
        state = Rung(state.tag + jnp.asarray(action, jnp.int32))
        return state, self.frame(state, action)

    def observation_space(self):
        return Space(shape=(1,), dtype=jnp.float32)

    def action_space(self):
        return Space(shape=(), dtype=jnp.int32, low=0, high=1)

    def horizon(self):
        return LENGTH


def stack(timesteps):
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *timesteps)


def selector(
    footprint,
    relevance,
    columns=None,
    gain=Advantage(),
    score=Diagonal(),
    allocation=psr.Marginal(),
):
    return psr.Selector(
        successor=SuccessorRepresentation(gamma=0.75, columns=columns),
        footprint=footprint,
        relevance=relevance,
        gain=gain,
        score=score,
        allocation=allocation,
        k=2.0,
    )


def place(selector, env, state, ledger, key, transitions):
    update_key, select_key, assign_key = jax.random.split(key, 3)
    if selector == "erd":
        slots = erd.ERD(env, k=2.0).select(state, select_key)
        return env.update(state, assign_key, slots=slots), ledger
    if ledger is None:
        ledger = selector.init(env.archive.archive_size, env.num_actions)
    if transitions is not None:
        ledger = selector.update(ledger, update_key, transitions)
    slots = selector.select(ledger, env, state, select_key)
    return env.update(state, assign_key, slots=slots), ledger


def archived(environment_state):
    def found(node):
        return isinstance(node, ArchiveAutoResetState)

    (state,) = [
        node for node in jax.tree.leaves(environment_state, is_leaf=found) if found(node)
    ]
    return state


def train(selector, updates=3, steps=4):
    env = ArchiveAutoReset(
        Ladder(),
        num_envs=NUM_ENVS,
        cell_fn=lambda env: (lambda state: state.tag),
        archive_size=32,
    )
    key = jax.random.key(0)
    archive_auto_reset_state, timestep = env.init(key)
    archive_auto_reset_state, ledger = place(
        selector, env, archive_auto_reset_state, None, key, None
    )
    for _ in range(updates):
        firsts, seconds = [], []
        for _ in range(steps):
            key, act, move = jax.random.split(key, 3)
            archive_auto_reset_state, after = env.step(
                move,
                archive_auto_reset_state,
                jax.random.randint(act, (NUM_ENVS,), 0, 2),
            )
            firsts.append(timestep)
            seconds.append(after)
            timestep = after
        blank = jnp.zeros((steps, NUM_ENVS))
        transitions = Transition(
            first=stack(firsts),
            second=stack(seconds),
            aux={
                "value": blank,
                "log_prob": blank,
                "logits": jnp.zeros((steps, NUM_ENVS, 2)),
            },
        )
        key, sub = jax.random.split(key)
        archive_auto_reset_state, ledger = place(
            selector, env, archive_auto_reset_state, ledger, sub, transitions
        )
    key, sub = jax.random.split(key)
    _, cut = env.step(sub, archive_auto_reset_state, jnp.ones(NUM_ENVS, jnp.int32))
    return env, archive_auto_reset_state, cut, ledger


@pytest.mark.parametrize(
    "selector",
    [
        selector(SuccessorFootprint(), CountedRelevance()),
        selector(SuccessorFootprint(), CountedRelevance(), columns=4),
        selector(SuccessorFootprint(), UniformRelevance()),
        selector(UniformFootprint(), CountedRelevance()),
        selector(UniformFootprint(), UniformRelevance()),
        selector(SuccessorFootprint(), CountedRelevance(), gain=GreedyAdvantage()),
        selector(SuccessorFootprint(), CountedRelevance(), gain=ModelAdvantage()),
        selector(
            SuccessorFootprint(),
            CountedRelevance(),
            gain=ModelAdvantage(),
            allocation=TopK(width=4),
        ),
        selector(
            SuccessorFootprint(),
            CountedRelevance(),
            gain=ModelAdvantage(),
            allocation=Softmax(),
        ),
        selector(SuccessorFootprint(), CountedRelevance(), allocation=WaterFilling()),
        selector(UniformFootprint(), CountedRelevance(), allocation=WaterFilling()),
        selector(SuccessorFootprint(), CountedRelevance(), allocation=psr.Uniform()),
        "erd",
    ],
    ids=[
        "psr-successor",
        "psr-sparse-successor",
        "psr-uniform-relevance",
        "psr-uniform-footprint",
        "psr-uniform",
        "psr-greedy-advantage",
        "psr-model-advantage",
        "psr-top-k",
        "psr-softmax",
        "psr-water-filling",
        "psr-uniform-footprint-water-filling",
        "psr-uniform-allocation",
        "erd",
    ],
)
def test_a_restart_distribution_runs_and_restarts_the_trailing_block(selector):
    env, archive_auto_reset_state, cut, _ = train(selector)
    block = jnp.arange(NUM_ENVS) >= NUM_ENVS // 2
    eligible = env.archive.eligible(archive_auto_reset_state.archive_state)
    assert bool(jnp.all(archive_auto_reset_state.due_mask == block))
    assert bool(jnp.all(eligible[archive_auto_reset_state.assigned_slots[block]]))
    assert bool(jnp.all(cut.done[block]))


def test_a_uniform_footprint_keeps_the_counted_relevance():
    uniform = selector(UniformFootprint(), CountedRelevance())
    _, state, _, ledger = train(uniform)
    relevance = uniform.relevant(ledger)
    assert float(jnp.std(relevance[state.archive_state.mask])) > 0.0
    np.testing.assert_allclose(jnp.sum(relevance), 1.0, rtol=1e-5)


def window(cell, action, reward, truncated):
    steps, envs = jnp.shape(cell)
    return Transition(
        first=Timestep(obs=None, info={"cell": cell}),
        second=Timestep(
            obs=None,
            action=action,
            reward=reward,
            terminated=jnp.zeros((steps, envs), bool),
            truncated=truncated,
        ),
        aux={
            "value": jnp.zeros((steps, envs)),
            "log_prob": jnp.full((steps, envs), jnp.log(0.5)),
        },
    )


def test_advantage_gain_is_the_policy_weighted_spread_of_action_means():
    steps, envs = 5, 4
    action = jnp.broadcast_to(jnp.array([0, 0, 1, 1]), (steps, envs))
    reward = jnp.where(action == 0, 1.0, -1.0).at[-1].set(100.0)
    transitions = window(
        jnp.zeros((steps, envs), jnp.int32),
        action,
        reward,
        jnp.zeros((steps, envs), bool),
    )
    gain = Advantage(gamma=0.0)
    state = gain.update(gain.init(1, 2), transitions)
    assert jnp.allclose(gain(state), jnp.array([0.25 * 1.0 + 0.25 * 1.0]))


def test_advantage_ignores_returns_past_a_truncation_and_the_window_end():
    cell = jnp.array([[0, 0], [1, 1], [1, 1], [1, 1]])
    action = jnp.array([[0, 1], [0, 0], [0, 0], [0, 0]])
    reward = jnp.zeros((4, 2)).at[2, 0].set(100.0).at[3, 1].set(100.0)
    truncated = jnp.zeros((4, 2), bool).at[1, 0].set(True)
    gain = Advantage(gamma=0.9, gae_lambda=1.0)
    state = gain.update(gain.init(2, 2), window(cell, action, reward, truncated))
    assert float(gain(state)[0]) == 0.0


@pytest.mark.parametrize(
    "curriculum, options",
    [
        (psr.make, {"selector": selector(SuccessorFootprint(), CountedRelevance())}),
        (
            psr.make,
            {"selector": selector(SuccessorFootprint(), CountedRelevance(), score=Kernel())},
        ),
        (erd.make, {"k": 2.0}),
        (
            psr.make,
            {
                "selector": selector(SuccessorFootprint(), CountedRelevance()),
                "restart_at": "episode",
            },
        ),
        (erd.make, {"k": 2.0, "restart_at": "episode"}),
        (
            psr.make,
            {
                "selector": selector(SuccessorFootprint(), CountedRelevance(), allocation=psr.Importance()),
                "importance_weights": True,
            },
        ),
    ],
    ids=["psr", "psr-kernel", "erd", "psr-episode", "erd-episode", "psr-importance"],
)
def test_ppo_trains_through_the_archive_and_the_curriculum_places_restarts(
    curriculum, options
):
    def archive(environment, num_envs):
        return RecordEpisodeStatistics(
            ArchiveAutoReset(
                environment,
                num_envs=num_envs,
                cell_fn=lambda env: (lambda state: state.position),
                archive_size=32,
            )
        )

    def race(algorithm, environment, num_envs, num_steps):
        algorithm, environment, pit, lap = curriculum(algorithm, environment, **options)
        config = anakin.AnakinConfig(
            num_envs=num_envs, num_steps=num_steps, mesh=mesh(1)
        )
        return anakin.make(config, algorithm, environment, pit=pit, lap=lap)

    podracer = zoo.ppo(Corridor(), podracer=race, wrapper=archive)
    state = podracer.init(jax.random.key(0))
    state, logs = podracer.train(state, jax.random.key(1), 20)
    archive = archived(state.environment_state)
    assert int(jnp.sum(archive.archive_state.mask)) == Corridor().length - 1
    assert bool(jnp.any(archive.assigned_slots >= 0))
    assert bool(jnp.any(archive.due_mask)) == (options.get("restart_at") != "episode")
    assert np.isfinite(np.asarray(logs["archive/placed"])).all()
    assert all(
        np.isfinite(leaf).all()
        for leaf in jax.tree.leaves(state.algorithm_state.params)
    )
    if curriculum is psr.make:
        assert float(jnp.sum(state.pit_state.reach)) > 0.0

    evaluated, _ = podracer.evaluate(state, jax.random.key(2), 4)
    for kept, held in zip(jax.tree.leaves(state.pit_state), jax.tree.leaves(evaluated.pit_state)):
        np.testing.assert_array_equal(kept, held)


@pytest.mark.parametrize(
    "curriculum, options",
    [
        (psr.make, {"selector": selector(SuccessorFootprint(), CountedRelevance())}),
        (erd.make, {"k": 2.0}),
    ],
    ids=["psr", "erd"],
)
def test_a_restart_curriculum_needs_an_archive_auto_reset(curriculum, options):
    with pytest.raises(AssertionError, match="ArchiveAutoReset"):
        curriculum(None, RecordEpisodeStatistics(Corridor()), **options)


@pytest.mark.parametrize(
    "k, num_placed", [(2.0, 4), (4.0, 6), (1.0, 0), (float("inf"), NUM_ENVS)]
)
def test_k_rounds_the_fraction_of_environments_a_curriculum_places(k, num_placed):
    env = ArchiveAutoReset(
        Ladder(), num_envs=NUM_ENVS, cell_fn=lambda env: (lambda state: state.tag)
    )
    explorer = erd.ERD(env, k=k)
    prioritized = replace(selector(SuccessorFootprint(), CountedRelevance()), k=k)
    assert explorer.num_placed == num_placed
    assert prioritized.num_placed(NUM_ENVS) == num_placed


def test_merging_only_the_writers_gives_the_same_table_as_merging_every_environment():
    archive_size, num_envs, width = 32, 64, 8
    everyone = SuccessorRepresentation(gamma=0.75, columns=width)
    writers = SuccessorRepresentation(gamma=0.75, columns=width, writers=16)
    table, only_writers = everyone.init(archive_size), writers.init(archive_size)
    for step in range(6):
        leaving_key, landing_key, key = jax.random.split(jax.random.key(step), 3)
        leaving = jax.random.randint(leaving_key, (num_envs,), -1, 10)
        landing = jax.random.randint(landing_key, (num_envs,), -1, archive_size)
        onward = landing >= 0
        table = everyone.backup(table, leaving, landing, onward, key)
        only_writers = writers.backup(only_writers, leaving, landing, onward, key)
    np.testing.assert_array_equal(only_writers.columns, table.columns)
    np.testing.assert_allclose(only_writers.weights, table.weights, rtol=1e-6)


def test_merging_fewer_writers_than_cells_still_updates_a_subset():
    archive_size, num_envs = 32, 64
    writers = SuccessorRepresentation(gamma=0.75, columns=8, writers=2)
    table = writers.init(archive_size)
    leaving = jnp.arange(num_envs) % 10
    table = writers.backup(table, leaving, (leaving + 1) % archive_size, jnp.ones(num_envs, bool), jax.random.key(0))
    assert bool(jnp.all(jnp.isfinite(table.weights)))
    assert 0 < int(jnp.sum(jnp.any(table.weights > 0.0, axis=-1))) <= 2


class Dice(Ladder):
    def init(self, key):
        state = Rung(jax.random.randint(key, (), 100, 100000))
        return state, self.frame(state, jnp.int32(0))


def finishing(env):
    state, _ = env.init(jax.random.key(0))
    state = state.replace(env_state=state.env_state.replace(tag=jnp.full(NUM_ENVS, LENGTH - 1)))
    return env.step(jax.random.key(1), state, jnp.ones(NUM_ENVS, jnp.int32))


def starts(env):
    state, _ = finishing(env)
    return {int(tag) for tag in state.env_state.tag}


def test_a_pool_hands_out_only_the_states_it_sampled_once():
    cell_fn = lambda env: (lambda state: state.tag)
    own = ArchiveAutoReset(Dice(), num_envs=NUM_ENVS, cell_fn=cell_fn, archive_size=32)
    pooled = ArchiveAutoReset(
        Dice(), num_envs=NUM_ENVS, cell_fn=cell_fn, archive_size=32,
        fresh_fn=partial(archive_auto_reset.pool, size=3),
    )
    single = ArchiveAutoReset(
        Dice(), num_envs=NUM_ENVS, cell_fn=cell_fn, archive_size=32,
        fresh_fn=partial(archive_auto_reset.pool, size=1),
    )
    sampled, _ = jax.vmap(Dice().init)(jax.random.split(jax.random.key(0), 3))
    assert starts(pooled) <= {int(tag) for tag in sampled.tag}
    assert len(starts(single)) == 1
    assert len(starts(own)) > 3


def test_the_relative_efficiency_compares_the_batch_with_one_from_rho():
    archive_size, num_envs = 4, 4
    tabular = selector(SuccessorFootprint(), CountedRelevance())
    ledger = tabular.init(archive_size, num_actions=2).replace(
        successor=psr.Successor(
            columns=jnp.arange(archive_size)[None], weights=jnp.eye(archive_size)
        ),
        reach=jnp.array([3.0, 1.0, 0.0, 0.0]),
    )
    worth = jnp.array([0.0, 1.0, 0.0, 0.0])

    def relative_efficiency():
        tabular.log_relative_efficiency(
            ledger, worth, jnp.ones(archive_size, bool), jnp.array([1, 1]), num_envs
        )

    _, logs = lox.spool(relative_efficiency)()
    np.testing.assert_allclose(logs["relative_efficiency/linear"], 2.5)
    np.testing.assert_allclose(logs["relative_efficiency/saturated"], 2.5)
    np.testing.assert_allclose(logs["relative_efficiency/ceiling"], 4.0)


def test_water_filling_gives_each_start_its_share_of_the_level():
    archive_size, count = 4, 64
    tabular = selector(SuccessorFootprint(), CountedRelevance())
    ledger = tabular.init(archive_size, num_actions=2)
    ledger = ledger.replace(
        successor=psr.Successor(
            columns=jnp.arange(archive_size)[None], weights=jnp.eye(archive_size)
        ),
        gain=ledger.gain.replace(scale=jnp.float32(8.0)),
    )
    worth = jnp.array([4.0, 1.0, 1.0, 0.0])
    eligible_mask = jnp.array([True, True, False, True])
    slots = WaterFilling()(
        tabular, ledger, worth, eligible_mask, 0.0, jax.random.key(0), count
    )
    restarts = np.bincount(np.asarray(slots), minlength=archive_size)
    inverse_root_level = (count + 2 * 8.0) / (8.0 * (2.0 + 1.0))
    expected = 8.0 * (np.array([2.0, 1.0]) * inverse_root_level - 1.0)
    assert restarts.sum() == count
    assert restarts[2] == restarts[3] == 0
    np.testing.assert_allclose(restarts[:2], expected, atol=1.0)



def test_a_blend_sends_its_uniform_share_anywhere_and_the_rest_by_its_inner_rule():
    slots = jnp.arange(8)
    inner = lambda selector, state, worth, eligible, residents, key, count: jnp.full(count, 7)
    chosen = psr.Blend(inner=inner, uniform_share=0.25)(
        None, None, None, slots < 4, None, jax.random.key(0), 8
    )
    assert chosen.shape == (8,)
    assert bool(jnp.all(chosen[:2] < 4)) and bool(jnp.all(chosen[2:] == 7))


def test_importance_weights_undo_the_restarts_so_the_batch_starts_from_rho():
    env = ArchiveAutoReset(
        Ladder(), num_envs=NUM_ENVS, cell_fn=lambda env: (lambda state: state.tag), archive_size=8
    )
    state, _ = env.init(jax.random.key(0))
    reach = jnp.asarray([4.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    ledger = selector(SuccessorFootprint(), CountedRelevance()).init(8, 2).replace(reach=reach)
    slots = jnp.asarray([2, 2, 3, 3], jnp.int32)
    state = state.replace(current_slots=jnp.asarray([0, 0, 1, 2, -1, -1, -1, -1], jnp.int32))
    weights = selector(SuccessorFootprint(), CountedRelevance()).window_weights(ledger, env, state, slots)
    relevance = reach / reach.sum()
    mixture = 0.5 * relevance + 0.5 * jnp.zeros(8).at[slots].add(0.25)
    np.testing.assert_allclose(weights[4:], relevance[slots] / mixture[slots], rtol=1e-6)
    np.testing.assert_allclose(weights[:4], relevance[jnp.asarray([0, 0, 1, 2])] / mixture[jnp.asarray([0, 0, 1, 2])], rtol=1e-6)


def gains(counts, sums, squares):
    counts, sums, squares = (jnp.asarray(value, jnp.float32)[None] for value in (counts, sums, squares))
    return psr.GainState(
        action_counts=counts,
        advantage_sums=sums,
        square_sums=squares,
        probability_sums=jnp.zeros_like(counts),
        cell_visits=jnp.sum(counts, axis=-1),
        scale=jnp.zeros((), jnp.float32),
    )


def test_the_signal_is_the_spread_of_action_means_when_advantages_are_exact():
    counts = [10.0, 30.0]
    state = gains(counts, [10.0 * 2.0, 30.0 * -2.0 / 3.0], [10.0 * 4.0, 30.0 * 4.0 / 9.0])
    grand = (20.0 - 20.0) / 40.0
    expected = (10.0 * (2.0 - grand) ** 2 + 30.0 * (-2.0 / 3.0 - grand) ** 2) / 40.0
    np.testing.assert_allclose(float(signal(state)[0]), expected, rtol=1e-5)


@pytest.mark.parametrize("gap, expected", [(0.0, 0.0), (1.0, 0.25)])
def test_the_signal_removes_what_sampling_noise_alone_would_spread(gap, expected):
    generator = np.random.default_rng(0)
    samples = [generator.normal(mean, 3.0, 20_000) for mean in (gap / 2.0, -gap / 2.0)]
    state = gains(
        [len(sample) for sample in samples],
        [sample.sum() for sample in samples],
        [(sample**2).sum() for sample in samples],
    )
    assert abs(float(signal(state)[0]) - expected) < 0.01


def test_a_gated_selector_leaves_every_environment_at_rho_when_every_cell_is_worth_the_same():
    gated = replace(
        selector(SuccessorFootprint(), UniformRelevance(), gain=UniformGain()), gate=True
    )
    _, archive_auto_reset_state, _, _ = train(gated)
    assert not bool(jnp.any(archive_auto_reset_state.assigned_slots >= 0))
    assert not bool(jnp.any(archive_auto_reset_state.due_mask))


def test_a_target_relevance_weighs_each_slot_by_how_often_the_trained_policy_stood_in_its_cell():
    from boonta.curricula.psr import TargetRelevance, fold

    relevance = TargetRelevance(checkpoint="unused")
    visited = jnp.array([[3], [7], [7], [7]], jnp.int32)
    keys, counts = np.unique(np.asarray(fold(visited)), return_counts=True)
    relevance.keys = jnp.asarray(keys)
    relevance.weights = jnp.asarray(counts / counts.sum(), jnp.float32)

    archive_keys = jnp.array([[7], [3], [5], [0]], jnp.int32)
    mask = jnp.array([True, True, True, False])
    weights = relevance(None, None, jnp.zeros(4), keys=archive_keys, mask=mask)

    np.testing.assert_allclose(weights, [0.75, 0.25, 0.0, 0.0])


def test_a_target_advantage_is_the_learners_advantage_for_the_targets_actions_and_never_negative():
    from boonta.curricula.psr import TargetAdvantage, TargetState

    gain = TargetAdvantage(checkpoint="unused")
    table = jnp.zeros((2, 2))
    state = TargetState(
        action_counts=jnp.ones((2, 2)),
        advantage_sums=jnp.array([[2.0, 0.0], [2.0, 0.0]]),
        square_sums=table,
        probability_sums=table,
        cell_visits=jnp.array([2.0, 2.0]),
        scale=jnp.zeros(()),
        target_sums=jnp.array([[1.8, 0.2], [0.2, 1.8]]),
        target=None,
    )

    np.testing.assert_allclose(gain(state), [0.8, 0.0], atol=1e-6)


def halved(sums_per_half, counts_per_half):
    """A gain state whose halves hold the given per-action advantage sums and counts."""
    from boonta.curricula.psr import DoubleGreedyAdvantage

    gain = DoubleGreedyAdvantage(shrinkage=0.0)
    halves_counts = jnp.asarray(counts_per_half, jnp.float32)
    halves_sums = jnp.asarray(sums_per_half, jnp.float32)
    _, archive_size, num_actions = halves_counts.shape
    state = gain.init(archive_size, num_actions).replace(
        action_counts=halves_counts.sum(0),
        advantage_sums=halves_sums.sum(0),
        cell_visits=halves_counts.sum((0, 2)),
        half_counts=halves_counts,
        half_sums=halves_sums,
    )
    return gain, state


def unclipped_scores(gain, state):
    counts, sums = np.asarray(state.half_counts), np.asarray(state.half_sums)
    means = sums / counts
    centred = means - (sums.sum(-1) / counts.sum(-1))[..., None]
    picked = centred.argmax(-1)
    return np.take_along_axis(centred[::-1], picked[..., None], -1)[..., 0].mean(0)


def test_the_double_greedy_gain_drops_the_bias_of_a_maximum_over_noise():
    rng = np.random.default_rng(0)
    cells, actions, samples = 2000, 4, 8
    counts = np.full((2, cells, actions), samples, np.float32)
    # Every action is equally good, so there is nothing to learn anywhere.
    noise = rng.normal(size=(2, cells, actions, samples)).sum(-1)
    gain, state = halved(noise, counts)
    single, double = np.asarray(gain.greedy(state)), np.asarray(gain(state))
    assert single.mean() > 0.15  # about the noise of one mean, 1 / sqrt(16)
    # What is left comes from the clip at zero, not from picking the winner.
    assert double.mean() < 0.5 * single.mean()
    assert abs(float(np.mean(unclipped_scores(gain, state)))) < 0.02


def test_the_double_greedy_gain_keeps_an_action_that_is_better_in_both_halves():
    counts = np.full((2, 1, 4), 10.0, np.float32)
    sums = np.zeros((2, 1, 4), np.float32)
    sums[:, 0, 2] = 10.0 * 3.0  # action 2 is 3 better than the others in both halves
    gain, state = halved(sums, counts)
    np.testing.assert_allclose(np.asarray(gain(state)), np.asarray(gain.greedy(state)), rtol=1e-5)
    assert float(gain(state)[0]) > 2.0


def test_rank_correlation_matches_spearman_with_ties_and_a_mask():
    from scipy.stats import spearmanr

    from boonta.curricula.psr import rank_correlation

    rng = np.random.default_rng(1)
    a = np.round(rng.normal(size=200), 1)
    b = a + rng.normal(size=200)
    mask = rng.random(200) < 0.7
    expected = spearmanr(a[mask], b[mask]).statistic
    np.testing.assert_allclose(float(rank_correlation(jnp.asarray(a), jnp.asarray(b), jnp.asarray(mask))), expected, rtol=1e-4)


def test_the_hindsight_gain_is_the_high_quantile_of_the_returns_above_the_value():
    from boonta.curricula.psr import HindsightGain

    gain = HindsightGain(quantile=0.75)
    rng = np.random.default_rng(0)
    returns = np.stack([rng.normal(5.0, 2.0, 20000), rng.normal(1.0, 0.1, 20000), np.zeros(20000)])
    state = gain.init(3, 4).replace(
        return_counts=jnp.asarray([20000.0, 20000.0, 0.0]),
        return_sums=jnp.asarray(returns.sum(-1), jnp.float32),
        return_squares=jnp.asarray((returns**2).sum(-1), jnp.float32),
        value_counts=jnp.asarray([1.0, 1.0, 1.0]),
        value_sums=jnp.asarray([4.0, 2.0, -1.0]),
    )
    p = np.asarray(gain(state))
    # 5 + 0.674 * 2 - 4; the second cell's returns never beat its value; the third saw no return.
    np.testing.assert_allclose(p, [2.349, 0.0, 0.0], atol=0.05)


def test_rank_metrics_count_the_directions_the_features_span():
    from boonta.algorithms.ppo import rank_metrics

    rng = np.random.default_rng(0)
    basis = rng.normal(size=(3, 64))
    features = rng.normal(size=(4096, 3)) @ basis + 7.0  # three directions plus a shared offset
    metrics = {name: float(value) for name, value in rank_metrics(jnp.asarray(features)).items()}
    assert metrics["n99"] <= 3 and metrics["entropy"] <= 3.01
    assert metrics["dead"] == 0.0
    isotropic = {name: float(value) for name, value in rank_metrics(jnp.asarray(rng.normal(size=(4096, 64)))).items()}
    assert isotropic["n99"] > 55 and isotropic["entropy"] > 55
