import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hangar.recipes.ppo_sokoban import (CNNEncoder, PufferEncoder, cnn_planes,
                                        observation_extractor, one_hot_flatten, puffer_planes)


def board():
    obs = jnp.zeros((10, 10, 3), jnp.uint8)
    obs = obs.at[2, 3, 0].set(3)  # player
    obs = obs.at[0, 0, 1].set(1)  # wall
    obs = obs.at[4, 5, 0].set(4)  # box on target: both channels must stay present
    return obs.at[4, 5, 1].set(2)


def test_binary_planes_match_puffer_channel_order_and_keep_overlap():
    planes = np.asarray(jax.jit(puffer_planes)(board()))
    assert planes.shape == (400,) and planes.dtype == np.float32
    np.testing.assert_array_equal(np.flatnonzero(planes), [23, 100, 245, 345])
    np.testing.assert_array_equal(planes[planes != 0], np.ones(4))


@pytest.mark.parametrize('shape', [(), (3,), (2, 3)])
def test_encoder_preserves_batch_and_sequence_dimensions(shape):
    obs = jnp.broadcast_to(board(), (*shape, 10, 10, 3))
    encoder = PufferEncoder()
    variables = encoder.init(jax.random.key(0), obs)
    encoded = jax.jit(encoder.apply)(variables, obs)
    assert encoded.shape == (*shape, 1024)
    assert jnp.isfinite(encoded).all()
    assert 'bias' not in variables['params']['Dense_0']


def test_clock_does_not_affect_puffer_encoder():
    encoder = PufferEncoder(features=32)
    obs = board()
    variables = encoder.init(jax.random.key(0), obs)
    np.testing.assert_array_equal(
        encoder.apply(variables, obs),
        encoder.apply(variables, obs.at[..., 2].set(120)),
    )


@pytest.mark.parametrize('name', ['one_hot', 'cnn', 'puffer', 'flat'])
def test_feedforward_body_supports_training_with_either_encoder(name):
    model = observation_extractor(encoder=name, features=64)
    obs = jnp.stack([board(), board().at[2, 3, 0].set(0)])
    variables = model.init(jax.random.key(0), obs)
    def loss(params):
        output = model.apply({'params': params}, obs)
        return jnp.square(output - 1).mean()
    value, gradients = jax.jit(jax.value_and_grad(loss))(variables['params'])
    assert model.apply(variables, obs).shape == (2, 256)
    assert jnp.isfinite(value)
    assert all(jnp.isfinite(leaf).all() for leaf in jax.tree.leaves(gradients))
    assert any(jnp.any(leaf != 0) for leaf in jax.tree.leaves(gradients))


def test_unknown_encoder_is_rejected():
    with pytest.raises(ValueError, match='Unknown Sokoban encoder'):
        observation_extractor(encoder='unknown')


def test_cnn_input_keeps_binary_planes_and_normalizes_the_clock():
    obs = board().at[..., 2].set(60)
    planes = np.asarray(cnn_planes(obs, time_limit=120))
    assert planes.shape == (10, 10, 5)
    assert planes[2, 3, 0] == 1  # player
    assert planes[0, 0, 1] == 1  # wall
    assert planes[4, 5, 2] == 1 and planes[4, 5, 3] == 1  # box on target
    np.testing.assert_array_equal(planes[..., 4], np.full((10, 10), 0.5))
    np.testing.assert_allclose(cnn_planes(obs, time_limit=60)[..., 4], 1.0)


@pytest.mark.parametrize('shape', [(), (3,), (2, 3)])
def test_cnn_preserves_batch_and_sequence_dimensions(shape):
    obs = jnp.broadcast_to(board(), (*shape, 10, 10, 3))
    encoder = CNNEncoder(features=64)
    variables = encoder.init(jax.random.key(0), obs)
    encoded = jax.jit(encoder.apply)(variables, obs)
    assert encoded.shape == (*shape, 64)
    assert jnp.isfinite(encoded).all()


def test_cnn_observes_time_without_recurrent_state():
    encoder = CNNEncoder(features=64)
    obs = board()
    variables = encoder.init(jax.random.key(0), obs)
    early = encoder.apply(variables, obs)
    late = encoder.apply(variables, obs.at[..., 2].set(120))
    assert not np.allclose(early, late)
    assert set(variables) == {'params'}


def test_one_hot_encodes_each_board_layer_and_keeps_overlap():
    encoded = np.asarray(jax.jit(one_hot_flatten)(board().at[..., 2].set(60)))
    assert encoded.shape == (801,) and encoded.dtype == np.float32
    cells = encoded[:-1].reshape(10, 10, 8)
    np.testing.assert_array_equal(cells[..., :5].sum(axis=-1), np.ones((10, 10)))
    np.testing.assert_array_equal(cells[..., 5:].sum(axis=-1), np.ones((10, 10)))
    assert cells[2, 3, 3] == 1  # player
    assert cells[0, 0, 6] == 1  # wall
    assert cells[4, 5, 4] == 1 and cells[4, 5, 7] == 1  # box on target
    assert encoded[-1] == 0.5
    assert float(one_hot_flatten(board().at[..., 2].set(60), time_limit=60)[-1]) == 1.0


@pytest.mark.parametrize('shape', [(), (3,), (2, 3)])
def test_one_hot_preserves_batch_and_sequence_dimensions(shape):
    obs = jnp.broadcast_to(board(), (*shape, 10, 10, 3))
    encoded = jax.jit(one_hot_flatten)(obs)
    assert encoded.shape == (*shape, 801)


def test_clock_changes_only_one_input_to_the_one_hot_mlp():
    early = np.asarray(one_hot_flatten(board()))
    late = np.asarray(one_hot_flatten(board().at[..., 2].set(120)))
    np.testing.assert_array_equal(early[:-1], late[:-1])
    assert early[-1] == 0 and late[-1] == 1
