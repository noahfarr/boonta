import jax
import jax.numpy as jnp
import pytest
from flax import struct

import numpy as np

import zoo
from boonta.curricula import erd, prd
from boonta.curricula.prd import (Advantage, CountedRelevance, Diagonal,
                                  GreedyAdvantage, Kernel, SuccessorFootprint,
                                  SuccessorRepresentation, UniformFootprint,
                                  UniformRelevance)
from boonta.environments.environment import Environment
from boonta.environments.spaces import Space
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.environments.wrappers.archive_auto_reset import ArchiveAutoReset, locate, share
from boonta.podracers import anakin
from boonta.utils import Timestep, Transition, mesh
from dummies import Corridor

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


def selector(footprint, relevance, columns=None, gain=Advantage(), score=Diagonal()):
    return prd.Selector(
        successor=SuccessorRepresentation(gamma=0.75, columns=columns),
        footprint=footprint,
        relevance=relevance,
        gain=gain,
        score=score,
        k=2.0,
    )


def place(selector, env, state, key, transitions):
    update_key, select_key, assign_key = jax.random.split(key, 3)
    if isinstance(selector, erd.Selector):
        return env.update(state, assign_key, slots=selector.select(env, state, select_key))
    ledger = state.ledger
    if not jax.tree.leaves(ledger):
        ledger = selector.init(env.archive.archive_size, env.num_actions)
    if transitions is not None:
        ledger = selector.update(ledger, update_key, transitions)
    slots = selector.select(ledger, env, state, select_key, transitions=transitions)
    return env.update(state, assign_key, slots=slots, ledger=ledger)


def train(selector, updates=3, steps=4):
    env = ArchiveAutoReset(
        Ladder(),
        num_envs=NUM_ENVS,
        cell_fn=lambda env: (lambda state: state.tag),
        capacity=32,
    )
    key = jax.random.key(0)
    archive_auto_reset_state, timestep = env.init(key)
    archive_auto_reset_state = place(selector, env, archive_auto_reset_state, key, None)
    for _ in range(updates):
        firsts, seconds = [], []
        for _ in range(steps):
            key, act, move = jax.random.split(key, 3)
            archive_auto_reset_state, after = env.step(
                move, archive_auto_reset_state, jax.random.randint(act, (NUM_ENVS,), 0, 2)
            )
            firsts.append(timestep)
            seconds.append(after)
            timestep = after
        blank = jnp.zeros((steps, NUM_ENVS))
        transitions = Transition(
            first=stack(firsts), second=stack(seconds), aux={"value": blank, "log_prob": blank}
        )
        key, sub = jax.random.split(key)
        archive_auto_reset_state = place(selector, env, archive_auto_reset_state, sub, transitions)
    key, sub = jax.random.split(key)
    _, cut = env.step(sub, archive_auto_reset_state, jnp.ones(NUM_ENVS, jnp.int32))
    return env, archive_auto_reset_state, cut


@pytest.mark.parametrize(
    "selector",
    [
        selector(SuccessorFootprint(), CountedRelevance()),
        selector(SuccessorFootprint(), CountedRelevance(), columns=4),
        selector(SuccessorFootprint(), UniformRelevance()),
        selector(UniformFootprint(), CountedRelevance()),
        selector(UniformFootprint(), UniformRelevance()),
        selector(SuccessorFootprint(), CountedRelevance(), gain=GreedyAdvantage()),
        erd.Selector(k=2.0),
    ],
    ids=[
        "prd-successor",
        "prd-sparse-successor",
        "prd-uniform-relevance",
        "prd-uniform-footprint",
        "prd-uniform",
        "prd-greedy-advantage",
        "erd",
    ],
)
def test_a_restart_distribution_runs_and_restarts_the_trailing_block(selector):
    env, archive_auto_reset_state, cut = train(selector)
    block = jnp.arange(NUM_ENVS) >= NUM_ENVS - share(2.0, NUM_ENVS)
    eligible = env.archive.eligible(archive_auto_reset_state.archive_state)
    assert bool(jnp.all(archive_auto_reset_state.due_mask == block))
    assert bool(jnp.all(eligible[archive_auto_reset_state.assigned_slots[block]]))
    assert bool(jnp.all(cut.done[block]))


def test_a_uniform_footprint_keeps_the_counted_relevance():
    uniform = selector(UniformFootprint(), CountedRelevance())
    _, state, _ = train(uniform)
    relevance = uniform.relevant(state.ledger)
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
        aux={"value": jnp.zeros((steps, envs)), "log_prob": jnp.full((steps, envs), jnp.log(0.5))},
    )


def test_advantage_gain_is_the_policy_weighted_spread_of_action_means():
    steps, envs = 5, 4
    action = jnp.broadcast_to(jnp.array([0, 0, 1, 1]), (steps, envs))
    reward = jnp.where(action == 0, 1.0, -1.0).at[-1].set(100.0)
    transitions = window(jnp.zeros((steps, envs), jnp.int32), action, reward, jnp.zeros((steps, envs), bool))
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
    "curriculum, selector",
    [
        (prd.pit, selector(SuccessorFootprint(), CountedRelevance())),
        (prd.pit, selector(SuccessorFootprint(), CountedRelevance(), score=Kernel())),
        (erd.pit, erd.Selector()),
    ],
    ids=["prd", "prd-kernel", "erd"],
)
def test_ppo_trains_through_the_archive_and_the_curriculum_places_restarts(curriculum, selector):
    def archive(environment, num_envs):
        return RecordEpisodeStatistics(
            ArchiveAutoReset(
                environment,
                num_envs=num_envs,
                cell_fn=lambda env: (lambda state: state.position),
                capacity=32,
            )
        )

    def race(algorithm, environment, num_envs, num_steps):
        algorithm, environment, pit, lap = curriculum(algorithm, environment, selector)
        config = anakin.AnakinConfig(num_envs=num_envs, num_steps=num_steps, mesh=mesh(1))
        return anakin.make(config, algorithm, environment, pit=pit, lap=lap)

    podracer = zoo.ppo(Corridor(), podracer=race, wrapper=archive)
    state = podracer.init(jax.random.key(0))
    state, logs = podracer.train(state, jax.random.key(1), 20)
    archived = locate(state.environment_state)
    assert int(jnp.sum(archived.archive_state.mask)) == Corridor().length - 1
    assert bool(jnp.any(archived.due_mask))
    assert np.isfinite(np.asarray(logs["archive/placed"])).all()
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(state.algorithm_state.params))
