from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf import OmegaConf

from boonta import datasets
from boonta.datasets.kinetix import Kinetix
from boonta.datasets.minari import Minari, convert, stack
from boonta.utils import mesh

from dummies import publish

LENGTHS = [5, 1, 7, 3]
CONFIGS = Path(__file__).parents[1] / "hangar/config/dataset"
MANY = [3 + index % 5 for index in range(20)]


def episode(length, index, dict_obs=False):
    flags = np.zeros((length,), bool)
    flags[-1] = index % 2 == 0
    observations = 100.0 * index + np.arange(length + 1, dtype=np.float64)[:, None]
    return SimpleNamespace(
        observations={"position": observations, "velocity": -observations}
        if dict_obs
        else observations,
        actions=100.0 * index + np.arange(length * 2, dtype=np.float64).reshape(length, 2) + 1,
        rewards=100.0 * index + np.arange(1, length + 1, dtype=np.float64),
        terminations=flags,
        truncations=~flags,
    )


def episodes(lengths=LENGTHS, dict_obs=False):
    return [episode(length, index, dict_obs) for index, length in enumerate(lengths)]


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path))

    def write(lengths=LENGTHS, dict_obs=False):
        return publish(episodes(lengths, dict_obs))

    return write


def unsharded(dataset):
    return jax.tree.map(lambda _: None, dataset.init())


def whole(archive, dict_obs=False):
    return Minari(archive(dict_obs=dict_obs)).init().transitions


def streamed(archive, dict_obs=False):
    dataset = Minari(archive(dict_obs=dict_obs), pool_size=sum(LENGTHS) - 1)
    pool = dataset.update(dataset.init(), jax.random.key(0), unsharded(dataset))
    dataset.close()
    return jax.device_get(pool.transitions)


SOURCES = [whole, streamed]


@pytest.mark.parametrize("source", SOURCES)
def test_first_marks_episode_starts_and_carries_the_previous_step(source, archive):
    transitions = source(archive)
    index = np.asarray(transitions.second.reward) % 100
    starts = index == 1

    np.testing.assert_array_equal(np.asarray(transitions.first.terminated), starts)
    np.testing.assert_array_equal(np.asarray(transitions.first.reward)[starts], 0.0)
    np.testing.assert_array_equal(np.asarray(transitions.first.action)[starts], 0.0)
    later = ~starts
    np.testing.assert_array_equal(
        np.asarray(transitions.first.reward)[later],
        np.asarray(transitions.second.reward)[later] - 1,
    )
    np.testing.assert_array_equal(
        np.asarray(transitions.first.action)[later],
        np.asarray(transitions.second.action)[later] - 2,
    )


@pytest.mark.parametrize("source", SOURCES)
def test_observations_are_paired_one_step_apart(source, archive):
    transitions = source(archive)
    np.testing.assert_array_equal(
        np.asarray(transitions.second.obs), np.asarray(transitions.first.obs) + 1
    )


@pytest.mark.parametrize("source", SOURCES)
def test_flags_stay_boolean_and_values_single_precision(source, archive):
    transitions = source(archive)
    for flag in (transitions.first.terminated, transitions.second.truncated):
        assert flag.dtype == np.bool_
    assert transitions.first.obs.dtype == np.float32
    assert transitions.second.reward.dtype == np.float32


@pytest.mark.parametrize("source", SOURCES)
def test_dict_observations_survive_the_round_trip(source, archive):
    transitions = source(archive, dict_obs=True)
    position, velocity = (transitions.first.obs[name] for name in ("position", "velocity"))
    np.testing.assert_array_equal(np.asarray(velocity), -np.asarray(position))


def test_a_whole_dataset_matches_the_episodes_it_was_written_from(archive):
    expected = jax.tree.map(stack, *[convert(e) for e in episodes(dict_obs=True)])
    got = whole(archive, dict_obs=True)
    jax.tree.map(np.testing.assert_array_equal, got, expected)
    for got_leaf, expected_leaf in zip(jax.tree.leaves(got), jax.tree.leaves(expected)):
        assert got_leaf.dtype == expected_leaf.dtype


@pytest.mark.parametrize("devices", [2, 4])
def test_whole_datasets_trim_rows_to_the_device_count(devices, archive):
    rows = sum(LENGTHS) // devices * devices
    transitions = Minari(archive(), num_devices=devices).init().transitions
    assert {len(leaf) for leaf in jax.tree.leaves(transitions)} == {rows}


def test_a_pool_as_large_as_the_dataset_loads_it_whole(archive):
    dataset_id = archive()
    jax.tree.map(
        np.testing.assert_array_equal,
        Minari(dataset_id, pool_size=sum(LENGTHS)).init().transitions,
        Minari(dataset_id).init().transitions,
    )


def test_a_streamed_pool_holds_distinct_transitions_placed_as_asked(archive):
    dataset = Minari(archive(), pool_size=7, num_devices=2)
    sharding = jax.tree.map(lambda _: NamedSharding(mesh(2), P("data")), dataset.init())
    pool = dataset.update(dataset.init(), jax.random.key(0), sharding).transitions
    dataset.close()

    rewards = np.asarray(pool.second.reward)
    assert len(rewards) == 6
    assert len(set(rewards.tolist())) == 6
    assert pool.first.obs.sharding.spec == P("data")


def runs(rewards):
    episode, step = np.divmod(rewards.astype(np.int64), 100)
    cuts = np.flatnonzero(np.diff(episode)) + 1
    return [
        (int(owner[0]), steps) for owner, steps in zip(np.split(episode, cuts), np.split(step, cuts))
    ]


@pytest.mark.parametrize("seed", range(4))
def test_a_pool_is_whole_episodes_with_only_the_last_cut_short(seed, archive):
    dataset = Minari(archive(MANY), pool_size=20)
    pool = dataset.stage(jax.random.key(seed), unsharded(dataset)).transitions
    dataset.close()

    found = runs(np.asarray(pool.second.reward))
    *complete, (_, last) = found
    assert sum(len(steps) for _, steps in found) == 20
    for _, steps in found:
        np.testing.assert_array_equal(steps, np.arange(1, len(steps) + 1))
    for owner, steps in complete:
        assert len(steps) == MANY[owner]
    assert len({owner for owner, _ in found}) == len(found)


def test_pools_depend_only_on_the_key(archive):
    dataset = Minari(archive(MANY), pool_size=20)
    sharding = unsharded(dataset)
    stage = dataset.stage
    first, again, other = (
        np.asarray(stage(jax.random.key(seed), sharding).transitions.second.reward)
        for seed in (0, 0, 1)
    )
    dataset.close()

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_each_update_hands_over_the_pool_staged_the_call_before(archive):
    dataset = Minari(archive(), pool_size=4)
    sharding = unsharded(dataset)
    first, second = jax.random.key(1), jax.random.key(2)

    handed = [dataset.update(dataset.init(), key, sharding) for key in (first, second)]
    dataset.close()

    jax.tree.map(np.testing.assert_array_equal, handed[0], dataset.stage(first, sharding))
    jax.tree.map(
        np.testing.assert_array_equal,
        handed[1],
        dataset.stage(jax.random.fold_in(first, 1), sharding),
    )


def test_close_stops_the_staging_thread(archive):
    dataset = Minari(archive(), pool_size=4)
    dataset.update(dataset.init(), jax.random.key(0), unsharded(dataset))
    dataset.close()
    assert dataset.executor._shutdown


def test_the_config_hands_the_device_count_and_pool_size_to_the_dataset(monkeypatch):
    received = {}
    monkeypatch.setitem(
        datasets.registry, "minari", lambda dataset_id, **kwargs: received.update(kwargs)
    )
    run = OmegaConf.merge(
        {"environment": {"env_id": "hopper"}, "podracer": {"config": {"mesh": {"count": 2}}}},
        {"dataset": OmegaConf.load(CONFIGS / "minari/mujoco/expert.yaml")},
    )
    datasets.make(**run.dataset)
    assert received == {"num_devices": 2, "pool_size": 0}


TRAJECTORIES, STEPS, DIMS = 2, 6, 3
ENDS = np.array(
    [[False, False, True, False, False, True], [False, True, False, False, True, False]]
)
STARTS = np.array(
    [[True, False, False, True, False, False], [True, False, True, False, False, True]]
)


def manager():
    action = np.arange(TRAJECTORIES * STEPS * DIMS, dtype=np.int32) + 1
    batch = SimpleNamespace(
        action=action.reshape(TRAJECTORIES, STEPS, DIMS),
        action_mask=np.ones((TRAJECTORIES, STEPS, DIMS), bool),
        done=ENDS,
        env_state=np.zeros((TRAJECTORIES, STEPS, 1), np.float32),
    )
    return SimpleNamespace(load_next_batch=lambda: batch, batch_size=TRAJECTORIES)


def kinetix_sample():
    dataset = Kinetix(manager(), render=None, batch_size=TRAJECTORIES)
    return jax.jit(
        lambda: dataset.sample(dataset.init(), jax.random.key(0), (TRAJECTORIES, STEPS))
    )()


def test_kinetix_marks_starts_and_shifts_the_label_out_of_view():
    transitions = kinetix_sample()
    action = manager().load_next_batch().action
    np.testing.assert_array_equal(np.asarray(transitions.first.terminated), STARTS)
    np.testing.assert_array_equal(np.asarray(transitions.second.action), action)
    np.testing.assert_array_equal(np.asarray(transitions.first.action)[:, 1:], action[:, :-1])
    np.testing.assert_array_equal(np.asarray(transitions.first.action)[:, 0], 0)


def test_kinetix_packs_every_leaf_into_fewer_transfers_and_back():
    dataset = Kinetix(manager(), render=None, batch_size=TRAJECTORIES)
    original = dataset.gather()
    packed = dataset.serve()
    assert len(packed) < len(jax.tree.leaves(original))
    restored = dataset.unpack({name: jnp.asarray(value) for name, value in packed.items()})
    for want, got in zip(jax.tree.leaves(original), jax.tree.leaves(restored)):
        assert want.dtype == got.dtype
        np.testing.assert_array_equal(want, np.asarray(got))


@pytest.mark.parametrize("shape", [(TRAJECTORIES,), (TRAJECTORIES, STEPS - 1)])
def test_kinetix_refuses_a_shape_it_cannot_serve(shape):
    dataset = Kinetix(manager(), render=None, batch_size=TRAJECTORIES)
    with pytest.raises(AssertionError):
        dataset.sample(dataset.init(), jax.random.key(0), shape)
