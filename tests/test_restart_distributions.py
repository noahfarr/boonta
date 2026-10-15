from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import lox
import pytest
import zoo
from dummies import Corridor
from flax import struct

from boonta.curricula import erd, prd
from boonta.curricula.prd import (
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
    UniformRelevance,
    WaterFilling,
)
from boonta.environments.environment import Environment
from boonta.environments.spaces import Space
from boonta.environments.wrappers import RecordEpisodeStatistics
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
    allocation=prd.Marginal(),
):
    return prd.Selector(
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
        selector(SuccessorFootprint(), CountedRelevance(), allocation=prd.Uniform()),
        "erd",
    ],
    ids=[
        "prd-successor",
        "prd-sparse-successor",
        "prd-uniform-relevance",
        "prd-uniform-footprint",
        "prd-uniform",
        "prd-greedy-advantage",
        "prd-model-advantage",
        "prd-top-k",
        "prd-softmax",
        "prd-water-filling",
        "prd-uniform-footprint-water-filling",
        "prd-uniform-allocation",
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
        (prd.make, {"selector": selector(SuccessorFootprint(), CountedRelevance())}),
        (
            prd.make,
            {"selector": selector(SuccessorFootprint(), CountedRelevance(), score=Kernel())},
        ),
        (erd.make, {"k": 2.0}),
        (
            prd.make,
            {
                "selector": selector(SuccessorFootprint(), CountedRelevance()),
                "restart_at": "episode",
            },
        ),
        (erd.make, {"k": 2.0, "restart_at": "episode"}),
        (
            prd.make,
            {
                "selector": selector(SuccessorFootprint(), CountedRelevance(), allocation=prd.Importance()),
                "importance_weights": True,
            },
        ),
    ],
    ids=["prd", "prd-kernel", "erd", "prd-episode", "erd-episode", "prd-importance"],
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
    if curriculum is prd.make:
        assert float(jnp.sum(state.pit_state.reach)) > 0.0

    evaluated, _ = podracer.evaluate(state, jax.random.key(2), 4)
    for kept, held in zip(jax.tree.leaves(state.pit_state), jax.tree.leaves(evaluated.pit_state)):
        np.testing.assert_array_equal(kept, held)


@pytest.mark.parametrize(
    "curriculum, options",
    [
        (prd.make, {"selector": selector(SuccessorFootprint(), CountedRelevance())}),
        (erd.make, {"k": 2.0}),
    ],
    ids=["prd", "erd"],
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


def test_the_relative_efficiency_compares_the_batch_with_one_from_rho():
    archive_size, num_envs = 4, 4
    tabular = selector(SuccessorFootprint(), CountedRelevance())
    ledger = tabular.init(archive_size, num_actions=2).replace(
        successor=prd.Successor(
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
        successor=prd.Successor(
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
    chosen = prd.Blend(inner=inner, uniform_share=0.25)(
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
