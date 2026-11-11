import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip('jumanji')

from boonta.environments import dune_sea
from boonta.environments.dune_sea.boxoban_tower import BoxobanTower, cell, parse_level
from boonta.environments.wrappers.archive_auto_reset import ArchiveAutoReset
from boonta.environments.wrappers.same_step_auto_reset import SameStepAutoReset

KEY = jax.random.key(0)


def room(*rows):
    return '\n'.join([*rows, *(['##########'] * (10 - len(rows)))])


ROOM = room('##########', '# @$.    #', '# ***    #')
PLACEMENT_ROOM = room('##########', '#        #', '# @$ .   #', '# $   .  #', '# **     #', '#        #')


def tower(**kwargs):
    kwargs.setdefault('num_floors', 2)
    return BoxobanTower(levels=[ROOM, ROOM], **kwargs)


def assert_leaf_equal(actual, expected):
    if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
        actual, expected = jax.random.key_data(actual), jax.random.key_data(expected)
    np.testing.assert_array_equal(actual, expected)


def test_solving_advances_and_only_summit_terminates():
    env = tower()
    state, timestep = jax.jit(env.init)(KEY)
    assert timestep.terminated and not state.ended
    step = jax.jit(env.step)
    state, timestep = step(KEY, state, jnp.int32(1))
    assert state.floor == 1 and not timestep.done
    assert float(timestep.reward) == pytest.approx(10.9)
    assert state.floor_steps == 0 and state.steps == 1
    np.testing.assert_array_equal(state.boxes, env.initial_boxes[1])
    state, timestep = step(KEY, state, jnp.int32(1))
    assert state.floor == 2 and timestep.terminated and not timestep.truncated
    assert float(timestep.reward) == pytest.approx(10.9)
    final, timestep = step(KEY, state, jnp.int32(0))
    assert timestep.reward == 0 and timestep.terminated
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(final)):
        assert_leaf_equal(before, after)


def test_moves_rewards_and_timeouts_match_jumanji_step_for_step():
    env = BoxobanTower(levels=[PLACEMENT_ROOM], num_floors=1, floor_horizon=8)
    state, _ = env.init(KEY)
    native_state = state.puzzle
    for action in [1, 1, 1, 0, 0, 3, 3, 0]:
        native_state, native = env.sokoban.step(native_state, jnp.int32(action))
        state, timestep = jax.jit(env.step)(KEY, state, jnp.int32(action))
        for actual, expected in zip(jax.tree.leaves(state.puzzle), jax.tree.leaves(native_state)):
            assert_leaf_equal(actual, expected)
        assert float(timestep.reward) == pytest.approx(float(native.reward))
        assert bool(timestep.terminated) == bool(native.last())
        assert not timestep.truncated
    assert state.ended and state.floor_steps == 8


def test_default_floor_timeout_is_120_actions_and_terminal():
    env = tower()
    state, _ = env.init(KEY)
    step = jax.jit(env.step)
    for i in range(120):
        state, timestep = step(KEY, state, jnp.int32(0))
        assert bool(timestep.terminated) == (i == 119)
        assert not timestep.truncated
    assert state.floor == 0 and state.steps == 120
    assert float(timestep.reward) == pytest.approx(-0.1)
    final, timestep = step(KEY, state, jnp.int32(1))
    assert final.floor == 0 and final.steps == 120 and timestep.reward == 0


def test_solve_on_floor_deadline_advances_and_solve_on_summit_deadline_wins():
    env = tower(floor_horizon=1, horizon=2)
    state, _ = env.init(KEY)
    state, timestep = env.step(KEY, state, jnp.int32(1))
    assert state.floor == 1 and state.floor_steps == 0 and not timestep.done
    state, timestep = env.step(KEY, state, jnp.int32(1))
    assert state.floor == 2 and timestep.terminated and not timestep.truncated
    assert float(timestep.reward) == pytest.approx(10.9)


def test_overall_budget_ends_episode_even_when_floor_is_solved():
    env = tower(horizon=1)
    state, _ = env.init(KEY)
    state, timestep = env.step(KEY, state, jnp.int32(1))
    assert state.floor == 1 and state.ended and timestep.terminated
    assert float(timestep.reward) == pytest.approx(10.9)


def test_auto_reset_starts_new_tower_after_floor_timeout():
    env = SameStepAutoReset(tower(floor_horizon=2), num_envs=1)
    state, _ = jax.jit(env.init)(KEY)
    for action in [1, 0, 0]:
        state, timestep = jax.jit(env.step)(KEY, state, jnp.array([action], jnp.int32))
    assert timestep.terminated.all() and not timestep.truncated.any()
    np.testing.assert_array_equal(state.floor, [0])
    np.testing.assert_array_equal(state.steps, [0])
    np.testing.assert_array_equal(state.floor_steps, [0])


def test_sampling_local_files_is_reproducible_without_replacement(tmp_path):
    rooms = [ROOM.replace('# @$.    #', row) for row in ['# @$.    #', '#@ $.    #', '#@$.     #']]
    (tmp_path / '000.txt').write_text('\n\n'.join(f'; {i}\n{puzzle}' for i, puzzle in enumerate(rooms)))
    env = BoxobanTower(dataset_path=str(tmp_path), num_floors=3, seed=7)
    again = BoxobanTower(dataset_path=str(tmp_path), num_floors=3, seed=7)
    np.testing.assert_array_equal(env.variable_grids, again.variable_grids)
    assert len({grid.tobytes() for grid in np.asarray(env.variable_grids)}) == 3
    with pytest.raises(ValueError, match='found only'):
        BoxobanTower(dataset_path=str(tmp_path), num_floors=4)


def test_vmap_clock_observation_four_actions_and_render():
    env = dune_sea.make('boxoban_tower', levels=[ROOM, ROOM], num_floors=2)
    states, timesteps = jax.jit(jax.vmap(env.init))(jax.random.split(KEY, 4))
    assert timesteps.obs.shape == (4, 10, 10, 6)
    assert env.action_space().num_actions == 4
    states, timesteps = jax.jit(jax.vmap(env.step))(jax.random.split(KEY, 4), states, jnp.array([0, 1, 2, 3]))
    np.testing.assert_allclose(timesteps.obs[0, ..., 5], 1 / 120)
    np.testing.assert_allclose(timesteps.obs[1, ..., 5], 0)
    np.testing.assert_allclose(timesteps.obs[1, ..., 4], 0.5)
    assert cell(env)(states).shape == (4, 103)
    assert not np.array_equal(cell(env)(states)[0], cell(env)(states)[1])
    state, _ = env.init(KEY)
    image = jax.jit(lambda s: env.render(s, tile=4))(state)
    assert image.shape == (40, 40, 3) and image.dtype == jnp.uint8


def test_psr_archive_stores_and_restores_native_puzzle_states():
    env = ArchiveAutoReset(tower(), num_envs=4, cell_fn=cell, archive_size=32)
    state, _ = jax.jit(env.init)(KEY)
    state, timestep = jax.jit(env.step)(KEY, state, jnp.ones(4, jnp.int32))
    assert timestep.obs.shape == (4, 10, 10, 6)
    assert int(state.archive_state.mask.sum()) >= 2
    state = env.assign(state, state.current_slots)
    state, timestep = jax.jit(env.step)(KEY, state, jnp.ones(4, jnp.int32))
    np.testing.assert_array_equal(state.env_state.floor, np.ones(4, np.int32))
    np.testing.assert_array_equal(state.env_state.steps, np.ones(4, np.int32))
    assert timestep.terminated.all()


@pytest.mark.parametrize('puzzle', ['', '##\n#', '#@x$.#', '#@$$.#', '# $.#', '#@*#'])
def test_invalid_levels_are_rejected(puzzle):
    with pytest.raises(ValueError):
        parse_level(puzzle)


def test_native_shape_and_box_count_are_required():
    with pytest.raises(ValueError, match='10 by 10'):
        BoxobanTower(levels=['######\n# @$.#\n######'], num_floors=1)
    with pytest.raises(ValueError, match='four boxes'):
        BoxobanTower(levels=[room('##########', '# @$.    #')], num_floors=1)


@pytest.mark.parametrize('kwargs', [{'floor_horizon': 0}, {'horizon': 0}, {'num_floors': 0}])
def test_budgets_and_floor_count_must_be_positive(kwargs):
    with pytest.raises(ValueError, match='positive'):
        tower(**kwargs)


def test_missing_dataset_explains_how_to_fetch_or_configure_it(tmp_path):
    with pytest.raises(FileNotFoundError, match='git clone --depth 1'):
        BoxobanTower(dataset_path=str(tmp_path / 'missing'))
