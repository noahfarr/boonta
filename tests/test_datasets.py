from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra.core.config_store import ConfigStore
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf import OmegaConf

import hangar.config  # noqa: F401
from boonta import datasets
from boonta.datasets.disk import Disk, write
from boonta.datasets.kinetix import Kinetix
from boonta.datasets.minari import Minari, convert, load, stack
from boonta.utils import mesh

LENGTHS = [5, 1, 7, 3]


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


def episodes(dict_obs=False):
    return [episode(length, index, dict_obs) for index, length in enumerate(LENGTHS)]


def from_minari(directory, monkeypatch, dict_obs=False):
    import minari

    monkeypatch.setattr(
        minari,
        "load_dataset",
        lambda *args, **kwargs: SimpleNamespace(
            iterate_episodes=lambda: episodes(dict_obs)
        ),
    )
    return Minari(load("anything")).init().transitions


def from_disk(directory, monkeypatch, dict_obs=False):
    write(directory, episodes(dict_obs))
    return Disk(directory).init().transitions


def from_a_streamed_pool(directory, monkeypatch, dict_obs=False):
    write(directory, episodes(dict_obs))
    dataset = Disk(directory, pool_size=sum(LENGTHS) - 1)
    sharding = jax.tree.map(lambda _: None, dataset.init())
    pool = dataset.update(dataset.init(), jax.random.key(0), sharding).transitions
    dataset.close()
    return jax.device_get(pool)


SOURCES = [from_minari, from_disk, from_a_streamed_pool]


@pytest.mark.parametrize("source", SOURCES)
def test_first_marks_episode_starts_and_carries_the_previous_step(
    source, tmp_path, monkeypatch
):
    transitions = source(tmp_path, monkeypatch)
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
def test_observations_are_paired_one_step_apart(source, tmp_path, monkeypatch):
    transitions = source(tmp_path, monkeypatch)
    np.testing.assert_array_equal(
        np.asarray(transitions.second.obs), np.asarray(transitions.first.obs) + 1
    )


@pytest.mark.parametrize("source", SOURCES)
def test_flags_stay_boolean_and_values_single_precision(source, tmp_path, monkeypatch):
    transitions = source(tmp_path, monkeypatch)
    for flag in (transitions.first.terminated, transitions.second.truncated):
        assert flag.dtype == np.bool_
    assert transitions.first.obs.dtype == np.float32
    assert transitions.second.reward.dtype == np.float32


def test_disk_matches_minari_exactly_with_dict_observations(tmp_path, monkeypatch):
    expected = jax.tree.map(stack, *[convert(e) for e in episodes(dict_obs=True)])
    got = from_disk(tmp_path, monkeypatch, dict_obs=True)
    jax.tree.map(np.testing.assert_array_equal, got, expected)
    for got_leaf, expected_leaf in zip(jax.tree.leaves(got), jax.tree.leaves(expected)):
        assert got_leaf.dtype == expected_leaf.dtype


@pytest.mark.parametrize("devices", [2, 4])
def test_whole_datasets_trim_rows_to_the_device_count(devices, tmp_path, monkeypatch):
    import minari

    monkeypatch.setattr(
        minari,
        "load_dataset",
        lambda *args, **kwargs: SimpleNamespace(iterate_episodes=episodes),
    )
    write(tmp_path, episodes())
    rows = sum(LENGTHS) // devices * devices
    for transitions in (
        load("anything", num_devices=devices),
        Disk(tmp_path, num_devices=devices).init().transitions,
    ):
        assert {len(leaf) for leaf in jax.tree.leaves(transitions)} == {rows}


def test_a_streamed_pool_holds_distinct_transitions_placed_as_asked(tmp_path):
    write(tmp_path, episodes())
    dataset = Disk(tmp_path, pool_size=6, num_devices=2)
    sharding = jax.tree.map(lambda _: NamedSharding(mesh(2), P("data")), dataset.init())
    pool = dataset.update(dataset.init(), jax.random.key(0), sharding).transitions
    dataset.close()

    rewards = np.asarray(pool.second.reward)
    assert len(set(rewards.tolist())) == 6
    assert pool.first.obs.sharding.spec == P("data")


def test_each_update_hands_over_the_pool_staged_the_call_before(tmp_path):
    write(tmp_path, episodes())
    dataset = Disk(tmp_path, pool_size=4)
    sharding = jax.tree.map(lambda _: None, dataset.init())
    first, second = jax.random.key(1), jax.random.key(2)

    handed = [dataset.update(dataset.init(), key, sharding) for key in (first, second)]
    dataset.close()

    jax.tree.map(np.testing.assert_array_equal, handed[0], dataset.stage(first, sharding))
    jax.tree.map(
        np.testing.assert_array_equal,
        handed[1],
        dataset.stage(jax.random.fold_in(first, 1), sharding),
    )


def test_minari_export_writes_what_minari_loads(tmp_path, monkeypatch):
    import minari

    monkeypatch.setattr(
        minari,
        "load_dataset",
        lambda *args, **kwargs: SimpleNamespace(iterate_episodes=episodes),
    )
    datasets.minari.export("anything", tmp_path)
    jax.tree.map(
        np.testing.assert_array_equal,
        Disk(tmp_path).init().transitions,
        load("anything"),
    )


@pytest.mark.parametrize(
    "config, namespace", [("minari/mujoco/expert.yaml", "minari"), ("disk.yaml", "disk")]
)
def test_the_config_hands_the_device_count_to_the_dataset(config, namespace, monkeypatch):
    received = {}
    monkeypatch.setitem(
        datasets.registry, namespace, lambda dataset_id, **kwargs: received.update(kwargs)
    )
    run = OmegaConf.merge(
        {"environment": {"env_id": "hopper"}, "podracer": {"config": {"mesh": {"count": 2}}}},
        {"dataset": ConfigStore.instance().load(f"dataset/{config}").node},
    )
    if namespace == "disk":
        run.dataset.dataset_id = "somewhere"
    datasets.make(**run.dataset)
    assert received["num_devices"] == 2


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
