import threading

import jax
import jax.numpy as jnp
import lox
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from boonta.algorithms.wrappers.ensemble import Ensemble
from boonta.datasets.minari import Minari
from boonta.environments.wrappers import Vectorize
from boonta.podracers import anakin, quadinaros, sebulba
from boonta.utils import mesh

from dummies import Dial, Episodes, Probe, demonstrations, flatten, publish, recordings

NUM_ENVS, NUM_STEPS, ACTORS = 8, 3, 2


def on_anakin(algorithm=None, environment=None, **hooks):
    config = anakin.AnakinConfig(num_envs=NUM_ENVS, num_steps=NUM_STEPS, mesh=mesh(2))
    environment = Vectorize(environment or Dial(), num_envs=NUM_ENVS)
    return anakin.make(config, algorithm or Probe(), environment, **hooks)


def on_sebulba(algorithm=None, environment=None, **hooks):
    config = sebulba.SebulbaConfig(
        num_envs=NUM_ENVS,
        num_steps=NUM_STEPS,
        learner=mesh(ACTORS),
        actor=mesh(ACTORS, start=ACTORS),
    )
    environment = Vectorize(environment or Dial(), num_envs=NUM_ENVS)
    return sebulba.make(config, algorithm or Probe(), environment, **hooks)


def on_quadinaros(algorithm=None, environment=None, dataset=None, **hooks):
    config = quadinaros.QuadinarosConfig(
        num_envs=NUM_ENVS, batch_shape=(16,), mesh=mesh(2)
    )
    environment = Vectorize(environment or Dial(), num_envs=NUM_ENVS)
    dataset = dataset or Episodes(flatten(demonstrations(Dial(), jax.random.key(0), 16)))
    return quadinaros.make(config, algorithm or Probe(), environment, dataset, **hooks)


ONLINE = [on_anakin, on_sebulba]
EVERY = [on_anakin, on_sebulba, on_quadinaros]


def environment_states(state):
    if hasattr(state, "actors"):
        return [actor.environment_state for actor in state.actors]
    return [state.environment_state]


def turn(state):
    setting = state.algorithm_state.version + 100.0
    return state.replace(
        environment_state=Vectorize(Dial(), NUM_ENVS).update(
            state.environment_state, setting=setting
        )
    )


def tick(state):
    return state.replace(
        algorithm_state=state.algorithm_state.replace(
            step=state.algorithm_state.step + 1000
        )
    )


@pytest.mark.parametrize("build", EVERY)
def test_train_runs_exactly_num_updates(build):
    podracer = build()
    state = podracer.init(jax.random.key(0))
    for _ in range(2):
        state, logs = podracer.train(state, jax.random.key(1), 3)
        assert len(np.asarray(logs["probe/version"])) == 3
    assert float(state.algorithm_state.version) == 6.0


@pytest.mark.parametrize("build", EVERY)
def test_step_counts_the_samples_every_update_trains_on(build):
    podracer = build()
    state = podracer.init(jax.random.key(0))
    state, _ = podracer.train(state, jax.random.key(1), 3)
    assert int(state.algorithm_state.step) == 3 * podracer.batch_size


@pytest.mark.parametrize("build", EVERY)
def test_lap_fires_after_every_environment_step(build):
    podracer = build(lap=tick)
    actors = ACTORS if build is on_sebulba else 1
    offline = build is on_quadinaros
    laps = 0 if offline else 2 * actors * NUM_STEPS
    steps = 2 * podracer.batch_size

    state = podracer.init(jax.random.key(0))
    assert int(state.algorithm_state.step) == actors * 1000

    state, _ = podracer.train(state, jax.random.key(1), 2)
    assert int(state.algorithm_state.step) == actors * 1000 + laps * 1000 + steps

    trained = int(state.algorithm_state.step)
    state, _ = podracer.evaluate(state, jax.random.key(2), 5)
    assert int(state.algorithm_state.step) == trained + 1000 + 5 * (NUM_ENVS + 1000)


@pytest.mark.parametrize("build", ONLINE)
def test_every_rollout_sees_the_setting_its_parameters_chose(build):
    podracer = build(pit=turn)
    state = podracer.init(jax.random.key(0))
    _, logs = podracer.train(state, jax.random.key(1), 4)
    np.testing.assert_array_equal(
        np.asarray(logs["probe/setting"]), np.asarray(logs["probe/acted"]) + 100.0
    )


@pytest.mark.parametrize("build", EVERY)
def test_pit_fires_at_init_after_training_and_at_evaluate(build):
    podracer = build(pit=turn)
    state = podracer.init(jax.random.key(0))
    for environment_state in environment_states(state):
        np.testing.assert_array_equal(np.asarray(environment_state.setting), 100.0)

    state, _ = podracer.train(state, jax.random.key(1), 3)
    state, _ = podracer.evaluate(state, jax.random.key(2), 2)
    for environment_state in environment_states(state):
        np.testing.assert_array_equal(np.asarray(environment_state.setting), 103.0)


@pytest.mark.parametrize("build", EVERY)
def test_evaluation_is_greedy_and_leaves_the_parameters(build):
    podracer = build()
    state = podracer.init(jax.random.key(0))
    state, _ = podracer.train(state, jax.random.key(1), 2)
    evaluated, logs = podracer.evaluate(state, jax.random.key(2), 5)

    np.testing.assert_array_equal(np.asarray(logs["probe/temperature"]), 0.0)
    assert float(evaluated.algorithm_state.version) == 2.0


@pytest.mark.parametrize("build", ONLINE)
def test_training_samples_at_temperature_one(build):
    podracer = build()
    state = podracer.init(jax.random.key(0))
    _, logs = podracer.train(state, jax.random.key(1), 2)
    np.testing.assert_array_equal(np.asarray(logs["probe/temperature"]), 1.0)


@pytest.mark.parametrize("build", EVERY)
def test_a_fixed_seed_reproduces_the_run(build):
    def run():
        podracer = build()
        state = podracer.init(jax.random.key(0))
        state, logs = podracer.train(state, jax.random.key(1), 3)
        state, evaluation = podracer.evaluate(state, jax.random.key(2), 3)
        return jax.device_get((state.algorithm_state, logs, evaluation))

    jax.tree.map(np.testing.assert_array_equal, run(), run())


@pytest.mark.parametrize("build", [on_anakin, on_quadinaros])
def test_environments_and_carries_shard_and_parameters_replicate(
    build, shards, assert_sharded, assert_replicated
):
    podracer = build()
    state = podracer.init(jax.random.key(0))
    for _ in range(2):
        for leaf in jax.tree.leaves((state.environment_state, state.timestep.obs)):
            assert_sharded(leaf, mesh(2))
        assert_sharded(state.algorithm_state.carry, mesh(2))
        assert_replicated(state.algorithm_state.version, mesh(2))
        first, second = shards(state.environment_state.noise)
        assert not np.array_equal(first, second)
        state, _ = podracer.train(state, jax.random.key(1), 2)


@pytest.mark.parametrize("build", [on_anakin, on_quadinaros])
def test_init_never_hands_train_one_buffer_twice(build):
    podracer = build()
    state = podracer.init(jax.random.key(0))
    pointers = [
        shard.data.unsafe_buffer_pointer()
        for leaf in jax.tree.leaves(state)
        for shard in leaf.addressable_shards
    ]
    assert len(pointers) == len(set(pointers))
    podracer.train(state, jax.random.key(1), 2)


def test_sebulba_trains_on_every_actor_at_once():
    podracer = on_sebulba()
    state = podracer.init(jax.random.key(0))
    first, second = (np.asarray(states.noise) for states in environment_states(state))
    assert not np.array_equal(first, second)

    state, logs = podracer.train(state, jax.random.key(1), 1)
    np.testing.assert_array_equal(np.asarray(logs["probe/width"]), ACTORS * NUM_ENVS)
    assert podracer.batch_size == ACTORS * NUM_ENVS * NUM_STEPS


def test_sebulba_actors_keep_their_environments_and_carries():
    podracer = on_sebulba()
    state = podracer.init(jax.random.key(0))
    for _ in range(2):
        state, _ = podracer.train(state, jax.random.key(1), 2)
    for actor in state.actors:
        np.testing.assert_array_equal(np.asarray(actor.environment_state.clock), 4 * NUM_STEPS)
        np.testing.assert_array_equal(np.asarray(actor.carry), 4 * NUM_STEPS)


def test_sebulba_rollouts_are_exactly_one_update_stale():
    podracer = on_sebulba()
    state = podracer.init(jax.random.key(0))
    _, logs = podracer.train(state, jax.random.key(1), 5)
    np.testing.assert_array_equal(np.asarray(logs["probe/acted"]), [0, 0, 1, 2, 3])


def test_sebulba_surfaces_an_actor_failure_instead_of_hanging():
    class Exploding(Dial):
        def step(self, key, state, action):
            raise RuntimeError("actor exploded")

    podracer = on_sebulba(environment=Exploding())
    state = podracer.init(jax.random.key(0))
    raised = []

    def train():
        try:
            podracer.train(state, jax.random.key(1), 1)
        except BaseException as error:
            raised.append(error)

    thread = threading.Thread(target=train, daemon=True)
    thread.start()
    thread.join(timeout=60)

    assert not thread.is_alive(), "train deadlocked on a failing actor"
    error, *_ = raised
    assert "actor exploded" in str(error)


@pytest.mark.parametrize("build", ONLINE)
def test_the_learner_shards_what_the_algorithm_keeps_per_environment(build):
    podracer = build()
    state = podracer.init(jax.random.key(0))
    shards = [
        state.algorithm_state.carry.sharding.spec,
        state.algorithm_state.version.sharding.spec,
    ]
    state, _ = podracer.train(state, jax.random.key(1), 2)
    shards += [
        state.algorithm_state.carry.sharding.spec,
        state.algorithm_state.version.sharding.spec,
    ]
    assert shards == [P("data"), P(), P("data"), P()]


@pytest.mark.parametrize("build", EVERY)
def test_a_podracer_refuses_a_component_it_would_ignore(build):
    with pytest.raises(TypeError, match="curriculum"):
        build(curriculum=lambda state: state)


def test_sebulba_rejects_ensembles():
    with pytest.raises(AssertionError, match="Use anakin"):
        on_sebulba(algorithm=Ensemble(Probe(), count=2))


class Ledger:
    def __init__(self, dataset):
        self.dataset = dataset
        self.updates = 0
        self.closed = False

    def init(self):
        return self.dataset.init()

    def update(self, state, key, sharding):
        self.updates += 1
        return state

    def sample(self, state, key, batch_shape):
        return self.dataset.sample(state, key, batch_shape)

    def close(self):
        self.closed = True


def test_quadinaros_updates_the_dataset_once_per_train_call_and_closes_it():
    dataset = Ledger(Episodes(flatten(demonstrations(Dial(), jax.random.key(0), 16))))
    podracer = on_quadinaros(dataset=dataset)
    state = podracer.init(jax.random.key(0))
    assert dataset.updates == 0
    for _ in range(3):
        state, _ = podracer.train(state, jax.random.key(1), 2)
    assert dataset.updates == 3
    podracer.close(state)
    assert dataset.closed


def test_quadinaros_pit_steers_the_data():
    def steer(state):
        transitions = state.dataset_state.transitions
        obs = transitions.first.obs.at[..., 1].set(state.algorithm_state.version)
        first = transitions.first.replace(obs=obs)
        return state.replace(
            dataset_state=state.dataset_state.replace(
                transitions=transitions.replace(first=first)
            )
        )

    class Reader(Probe):
        def update(self, state, key, transitions):
            lox.log({"reader/seen": jnp.mean(transitions.first.obs[..., 1])})
            return super().update(state, key, transitions)

    podracer = on_quadinaros(algorithm=Reader(), pit=steer)
    state = podracer.init(jax.random.key(0))
    _, logs = podracer.train(state, jax.random.key(1), 3)
    np.testing.assert_array_equal(np.asarray(logs["reader/seen"]), [0.0, 1.0, 2.0])


def test_quadinaros_streams_a_pool_of_episodes(tmp_path, monkeypatch, assert_sharded):
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(tmp_path))
    dataset_id = publish(recordings(demonstrations(Dial(), jax.random.key(0), 8)))
    dataset = Minari(dataset_id, pool_size=16, num_devices=2)
    podracer = on_quadinaros(dataset=dataset)
    state = podracer.init(jax.random.key(0))
    for epoch in range(3):
        state, logs = podracer.train(state, jax.random.key(epoch), 2)
        assert len(np.asarray(logs["probe/version"])) == 2
    assert_sharded(state.dataset_state.transitions.first.obs, mesh(2))
    podracer.close(state)
    assert dataset.executor._shutdown


def test_quadinaros_never_gathers_the_whole_dataset_onto_one_device():
    rows = 64
    podracer = on_quadinaros()
    state = podracer.init(jax.random.key(0))
    text = podracer.fit.lower(state, jax.random.key(1), 4).compile().as_text()

    assert f"f32[{rows},3]" not in text
    assert f"f32[{rows // 2},3]" in text
    gathers = [line for line in text.splitlines() if "all-gather(" in line]
    assert all(f"[{rows}," not in line for line in gathers)
