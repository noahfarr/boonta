import jax
import jax.numpy as jnp
import lox
import numpy as np
import optax
import pytest

import zoo
from boonta.algorithms import DR3
from boonta.utils import Timestep, Transition
from dummies import corridor, reach, recall, recall_continuous

NUM_ENVS, NUM_STEPS, MINIBATCHES = 8, 4, 4

SOWN = {
    zoo.ppo: (NUM_ENVS * NUM_STEPS // MINIBATCHES, zoo.WIDTH),
    zoo.recurrent_ppo: (NUM_ENVS // MINIBATCHES, NUM_STEPS, zoo.WIDTH),
}


def named(build):
    return build.__name__


def learner(build, *auxiliary_losses):
    return build(
        corridor(),
        num_envs=NUM_ENVS,
        num_steps=NUM_STEPS,
        optimizer=optax.sgd(0.1),
        auxiliary_losses=auxiliary_losses,
    )


def train(podracer, num_updates=1):
    state = podracer.init(jax.random.key(0))
    return podracer.train(state, jax.random.key(1), num_updates)


def energy(weight):
    def loss(intermediates, **kwargs):
        features = intermediates["intermediates"]["features"].astype(jnp.float32)
        mean_square = jnp.mean(features**2)
        lox.log({"features/energy": mean_square})
        return weight * mean_square

    return loss


@pytest.mark.parametrize("build", SOWN, ids=named)
def test_auxiliary_losses_receive_the_sown_features_and_the_update_state(build):
    shapes, keywords = [], []

    def record(intermediates, **kwargs):
        shapes.append(intermediates["intermediates"]["features"].shape)
        keywords.append(set(kwargs))
        return 0.0

    train(learner(build, record))

    assert set(shapes) == {SOWN[build]}
    expected = {"params", "apply", "transitions", "dist", "value"}
    assert all(expected <= received for received in keywords)


@pytest.mark.parametrize("build", SOWN, ids=named)
def test_nothing_sown_reaches_the_params(build):
    podracer = learner(build, energy(1.0))
    state = podracer.init(jax.random.key(0))
    assert set(state.algorithm_state.params) == {"params"}

    state, _ = podracer.train(state, jax.random.key(1), 1)
    assert set(state.algorithm_state.params) == {"params"}


@pytest.mark.parametrize("build", SOWN, ids=named)
def test_a_loss_on_the_sown_features_trains_the_network(build):
    _, plain = train(learner(build, energy(0.0)), 2)
    _, shrunk = train(learner(build, energy(10.0)), 2)

    plain, shrunk = np.asarray(plain["features/energy"]), np.asarray(shrunk["features/energy"])
    np.testing.assert_allclose(shrunk[0], plain[0], rtol=1e-5)
    assert shrunk.mean() < plain.mean()


SHARED = {"params", "apply", "transitions", "dist", "intermediates"}

EVERY = [
    pytest.param(zoo.ppo, corridor, {"value"}, id="ppo"),
    pytest.param(zoo.mmd, corridor, {"value"}, id="mmd"),
    pytest.param(zoo.grpo, corridor, set(), id="grpo"),
    pytest.param(zoo.pqn, corridor, {"q_values"}, id="pqn"),
    pytest.param(zoo.dqn, corridor, {"q_values"}, id="dqn"),
    pytest.param(zoo.sac, reach, set(), id="sac"),
    pytest.param(zoo.reppo, reach, set(), id="reppo"),
    pytest.param(zoo.tdmpc2, reach, set(), id="tdmpc2"),
    pytest.param(zoo.recurrent_ppo, recall, {"value", "carry"}, id="recurrent_ppo"),
    pytest.param(zoo.recurrent_pupo, recall, {"value", "carry"}, id="recurrent_pupo"),
    pytest.param(zoo.recurrent_grpo, recall, {"carry"}, id="recurrent_grpo"),
    pytest.param(zoo.recurrent_pqn, recall, {"q_values", "carry"}, id="recurrent_pqn"),
    pytest.param(zoo.recurrent_dqn, recall, {"q_values", "carry"}, id="recurrent_dqn"),
    pytest.param(zoo.recurrent_sac, recall_continuous, {"carry"}, id="recurrent_sac"),
    pytest.param(zoo.bc, corridor, set(), id="bc"),
    pytest.param(zoo.iql, reach, set(), id="iql"),
    pytest.param(zoo.recurrent_bc, recall, {"carry"}, id="recurrent_bc"),
]


@pytest.mark.parametrize("build, environment, extra", EVERY)
def test_every_algorithm_hands_its_auxiliary_losses_the_same_inputs(
    build, environment, extra
):
    received, replayed = [], []

    def record(params, apply, transitions, dist, intermediates, **kwargs):
        received.append({"params", "apply", "transitions", "dist", "intermediates", *kwargs})
        replayed.append(apply(params).log_prob(transitions.second.action).shape)
        assert set(intermediates["intermediates"]) == {"features"}
        return 0.0

    podracer = build(environment(), auxiliary_losses=(record,))
    state = podracer.init(jax.random.key(0))
    podracer.train(state, jax.random.key(1), 1)

    assert received
    assert all(keywords == SHARED | extra for keywords in received)
    assert all(shape == replayed[0] for shape in replayed)


def trajectory(flags, ending="terminated"):
    flags = jnp.asarray(flags, bool)
    never = jnp.zeros(flags.shape, bool)
    zeros = jnp.zeros(flags.shape)
    timestep = Timestep(
        obs=zeros,
        action=zeros,
        reward=zeros,
        terminated=flags if ending == "terminated" else never,
        truncated=flags if ending == "truncated" else never,
    )
    return Transition(first=timestep, second=timestep)


def intermediates(features):
    return {"intermediates": {"features": jnp.asarray(features, jnp.float32)}}


FEATURES = [
    [[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]],
    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
]


@pytest.mark.parametrize("coefficient", [1.0, 0.5])
def test_dr3_is_the_coefficient_times_the_mean_dot_product_of_consecutive_features(
    coefficient,
):
    loss = DR3(coefficient=coefficient)(
        intermediates=intermediates(FEATURES),
        transitions=trajectory([[False] * 3] * 2),
    )

    np.testing.assert_allclose(float(loss), coefficient * (2.0 + 3.0 + 2.0 + 2.0) / 4)


@pytest.mark.parametrize("ending", ["terminated", "truncated"])
def test_dr3_skips_pairs_that_cross_an_episode_end(ending):
    first, *_ = FEATURES
    loss = DR3(coefficient=1.0)(
        intermediates=intermediates([first]),
        transitions=trajectory([[False, True, False]], ending),
    )

    np.testing.assert_allclose(float(loss), 2.0)


def test_dr3_is_zero_when_every_pair_crosses_an_episode_end():
    first, *_ = FEATURES
    loss = DR3(coefficient=1.0)(
        intermediates=intermediates([first]),
        transitions=trajectory([[True, True, True]]),
    )

    assert float(loss) == 0.0


def test_dr3_gradient_pushes_consecutive_features_apart():
    features = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]]])
    transitions = trajectory([[False, False]])

    def loss(features):
        return DR3(coefficient=2.0)(
            intermediates=intermediates(features), transitions=transitions
        )

    gradient = jax.grad(loss)(features)
    stepped = features - 0.1 * gradient

    np.testing.assert_allclose(gradient, 2.0 * features[:, ::-1])
    assert float(loss(stepped)) < float(loss(features))


def test_dr3_rejects_shuffled_single_transitions():
    with pytest.raises(AssertionError, match="trajectories"):
        DR3(coefficient=1.0)(
            intermediates=intermediates([[1.0, 0.0], [0.0, 1.0]]),
            transitions=trajectory([False, False]),
        )


def test_dr3_logs_its_term_inside_recurrent_ppo():
    _, logs = train(learner(zoo.recurrent_ppo, DR3(coefficient=1.0)))

    assert np.isfinite(np.asarray(logs["auxiliary_loss/dr3"])).all()


def test_dr3_is_rejected_by_ppo_whose_minibatches_are_shuffled_transitions():
    with pytest.raises(AssertionError, match="trajectories"):
        train(learner(zoo.ppo, DR3(coefficient=1.0)))
