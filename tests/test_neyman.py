import jax
import jax.numpy as jnp
import numpy as np
import pytest
import zoo
from dummies import Corridor

from boonta.curricula import neyman
from boonta.curricula.neyman import Neyman, Outcomes, capped
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.environments.wrappers.archive_auto_reset import (
    ArchiveAutoReset,
    ArchiveAutoResetState,
)
from boonta.podracers import anakin
from boonta.utils import Timestep, Trajectory, Transition, mesh

ARCHIVE_SIZE = 8


def archive(num_envs=4):
    return ArchiveAutoReset(
        Corridor(),
        num_envs=num_envs,
        cell_fn=lambda env: (lambda state: state.position),
        archive_size=ARCHIVE_SIZE,
    )


def episodes(reward, done, cell, rho, truncated=None):
    truncated = jnp.zeros_like(done) if truncated is None else truncated
    return Transition(
        first=Timestep(obs=None, info={"cell": cell, "rho": rho}),
        second=Timestep(
            obs=None,
            reward=reward,
            terminated=done & ~truncated,
            truncated=truncated,
            info={"cell": cell, "rho": rho},
        ),
    )


def test_each_finished_episode_counts_its_discounted_return_at_the_cell_it_started_from():
    pit = Neyman(archive(num_envs=2), gamma=0.5)
    state = pit.init(jax.random.key(0), None).replace(
        origin=jnp.array([3, 5], jnp.int32), from_rho=jnp.array([False, True])
    )
    reward = jnp.array([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])
    done = jnp.array([[False, False], [True, False], [False, True]])
    cell = jnp.array([[3, 5], [6, 5], [6, 1]])
    rho = jnp.array([[False, True], [False, True], [False, False]])
    state = pit.update(
        state, jax.random.key(1), Trajectory(transitions=episodes(reward, done, cell, rho)), None
    )
    np.testing.assert_allclose(state.cells.episodes[3], 1.0)
    np.testing.assert_allclose(state.cells.returns[3], 1.0 + 0.5 * 2.0)
    np.testing.assert_allclose(state.cells.lengths[3], 2.0)
    np.testing.assert_allclose(state.rho.returns, 1.0 + 0.5 * 2.0 + 0.25 * 4.0)
    np.testing.assert_allclose(state.rho.lengths, 3.0)
    assert int(state.origin[0]) == 6 and not bool(state.from_rho[0])
    np.testing.assert_allclose(state.earned[0], 4.0)


def test_a_truncated_episode_leaves_no_outcome():
    pit = Neyman(archive(num_envs=1))
    state = pit.init(jax.random.key(0), None).replace(origin=jnp.array([2], jnp.int32))
    transitions = episodes(
        jnp.ones((2, 1)),
        jnp.array([[False], [True]]),
        jnp.array([[2], [0]]),
        jnp.zeros((2, 1), bool),
        truncated=jnp.array([[False], [True]]),
    )
    state = pit.update(state, jax.random.key(1), Trajectory(transitions=transitions), None)
    assert float(jnp.sum(state.cells.episodes)) == 0.0


def test_each_start_remembers_about_its_history_length_of_episodes():
    outcomes = Outcomes(*(jnp.array([90.0]),) * 4)
    kept = capped(outcomes, jnp.array([20.0]), jnp.array([20.0]), jnp.array([20.0]), jnp.array([20.0]), 100)
    np.testing.assert_allclose(kept.episodes, 100.0)
    np.testing.assert_allclose(kept.returns, 100.0)


def outcomes(returns_per_cell, length_per_cell):
    returns = np.asarray(returns_per_cell, np.float32)
    lengths = np.broadcast_to(np.asarray(length_per_cell, np.float32)[:, None], returns.shape)
    return Outcomes(
        episodes=jnp.full(len(returns), returns.shape[1], jnp.float32),
        returns=jnp.asarray(returns.sum(1)),
        squares=jnp.asarray((returns**2).sum(1)),
        lengths=jnp.asarray(lengths.sum(1)),
    )


def test_relevance_allocates_episodes_by_weight_times_spread_over_the_root_of_length():
    pit = Neyman(archive(), relevance=True, prior_count=0.0)
    returns = [[0, 0], [0, 2], [0, 2], [0, 4]] + [[0, 0]] * 4
    lengths = [1, 1, 4, 1, 1, 1, 1, 1]
    reach = jnp.array([2.0, 1.0, 1.0, 1.0, 0, 0, 0, 0])
    state = pit.init(jax.random.key(0), None).replace(
        cells=outcomes(returns, lengths),
        rho=Outcomes(jnp.float32(2.0), jnp.float32(0.0), jnp.float32(0.0), jnp.float32(2.0)),
        reach=reach,
        origins=jnp.zeros(ARCHIVE_SIZE).at[0].set(1.0),
    )
    probabilities, rho_probability = pit.allocate(state, jnp.arange(ARCHIVE_SIZE) < 4)
    spread = jnp.array([0.0, 1.0, 1.0 / 2.0, 2.0])
    scores = reach[:4] / reach.sum() * spread
    np.testing.assert_allclose(probabilities[:4], scores / scores.sum(), rtol=1e-5)
    np.testing.assert_allclose(probabilities[4:], 0.0)
    np.testing.assert_allclose(rho_probability, 0.0, atol=1e-6)


def test_without_relevance_rho_keeps_its_share_and_the_rest_follows_spread_over_root_length():
    pit = Neyman(archive(), relevance=False, rho_share=0.25, prior_count=0.0)
    returns = [[0, 2], [0, 4]] + [[0, 0]] * 6
    state = pit.init(jax.random.key(0), None).replace(cells=outcomes(returns, [1] * 8))
    probabilities, rho_probability = pit.allocate(state, jnp.arange(ARCHIVE_SIZE) < 2)
    np.testing.assert_allclose(rho_probability, 0.25)
    np.testing.assert_allclose(probabilities[:2], 0.75 * jnp.array([1.0, 2.0]) / 3.0, rtol=1e-5)


def test_untried_starts_borrow_the_pooled_spread_so_they_still_get_tried():
    pit = Neyman(archive(), relevance=False, prior_count=2.0)
    state = pit.init(jax.random.key(0), None).replace(
        rho=Outcomes(jnp.float32(2.0), jnp.float32(2.0), jnp.float32(4.0), jnp.float32(2.0))
    )
    probabilities, _ = pit.allocate(state, jnp.arange(ARCHIVE_SIZE) < 3)
    np.testing.assert_allclose(probabilities[:3], 0.75 / 3.0, rtol=1e-5)


def test_slot_weights_set_each_step_s_sample_weight_from_the_cell_it_is_in():
    env = archive(num_envs=2)
    state, _ = env.init(jax.random.key(0))
    weights = jnp.arange(ARCHIVE_SIZE, dtype=jnp.float32) + 1.0
    state = env.update(state, jax.random.key(1), slots=jnp.array([-1, -1], jnp.int32),
                       preempt=False, slot_weights=weights)
    state, timestep = env.step(jax.random.key(2), state, jnp.ones(2, jnp.int32))
    cells = timestep.info["cell"]
    np.testing.assert_allclose(timestep.info["sample_weight"], jnp.where(cells >= 0, weights[cells], 1.0))


@pytest.mark.parametrize(
    "options",
    [{"relevance": True, "importance_weights": True}, {"relevance": False}],
    ids=["relevance-weighted", "coverage"],
)
def test_ppo_trains_through_the_archive_and_neyman_restarts_whole_episodes(options):
    def wrapped(environment, num_envs):
        return RecordEpisodeStatistics(
            ArchiveAutoReset(
                environment,
                num_envs=num_envs,
                cell_fn=lambda env: (lambda state: state.position),
                archive_size=32,
            )
        )

    def race(algorithm, environment, num_envs, num_steps):
        algorithm, environment, pit, lap = neyman.make(algorithm, environment, **options)
        config = anakin.AnakinConfig(num_envs=num_envs, num_steps=num_steps, mesh=mesh(1))
        return anakin.make(config, algorithm, environment, pit=pit, lap=lap)

    podracer = zoo.ppo(Corridor(), podracer=race, wrapper=wrapped)
    state = podracer.init(jax.random.key(0))
    state, logs = podracer.train(state, jax.random.key(1), 20)

    def found(node):
        return isinstance(node, ArchiveAutoResetState)

    (held,) = [node for node in jax.tree.leaves(state.environment_state, is_leaf=found) if found(node)]
    assert not bool(jnp.any(held.due_mask))
    assert bool(jnp.any(held.assigned_slots >= 0))
    assert float(state.pit_state.rho.episodes) > 0.0
    assert float(jnp.sum(state.pit_state.cells.episodes)) > 0.0
    assert np.isfinite(np.asarray(logs["mu/rho_share"])).all()
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(state.algorithm_state.params))


def test_neyman_needs_an_archive_auto_reset():
    with pytest.raises(AssertionError, match="ArchiveAutoReset"):
        neyman.make(None, RecordEpisodeStatistics(Corridor()))
