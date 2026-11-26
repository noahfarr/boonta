import jax
import jax.numpy as jnp
import lox
import numpy as np
import optax

from boonta.algorithms.eggroll import EGGROLL, EGGROLLConfig
from boonta.environments.spaces import Space
from boonta.networks.low_rank_perturbations import LowRankPerturbedMLP
from boonta.utils import Timestep, Transition


def test_mirrored_low_rank_forward_and_update_match_dense_reference():
    cfg = EGGROLLConfig(population_size=4, rank=2, sigma=0.2)
    algorithm = EGGROLL(cfg, LowRankPerturbedMLP((2,)), Space(dtype=jnp.int32, high=1), optax.sgd(1.0))
    obs = jnp.arange(24, dtype=jnp.float32).reshape(8, 3) / 24
    timestep = Timestep(obs=obs, reward=jnp.zeros(8))
    init_key, generation_key = jax.random.split(jax.random.key(0))
    state, _ = algorithm.init(init_key, timestep)
    noise = algorithm.draw(state.params, state.noise_key)
    direction = noise[0]
    kernel_noise = direction.inputs @ direction.outputs
    signs = jnp.array([1.0, -1.0])
    kernel = state.params["dense_0"]["kernel"]
    bias = state.params["dense_0"]["bias"]
    candidate_kernels = kernel + cfg.sigma * signs[None, :, None, None] * kernel_noise[:, None]
    candidate_biases = bias + cfg.sigma * signs[None, :, None] * direction.bias[:, None]
    expected = obs.reshape(2, 2, 2, 3) @ candidate_kernels + candidate_biases[:, :, None]
    outputs = jax.jit(algorithm.network.apply)({"params": state.params}, obs, noise, sigma=cfg.sigma)
    np.testing.assert_allclose(outputs, expected.reshape(8, 2), atol=1e-6)

    first_reward = jnp.repeat(jnp.array([4.0, 2.0, 1.0, 3.0]), 2)
    done = jnp.array([[True] * 8, [False] * 8])
    second = Timestep(
        obs=None,
        reward=jnp.stack((first_reward, jnp.arange(8) * 100.0)),
        terminated=done,
        truncated=jnp.zeros_like(done),
    )
    update = jax.jit(lox.spool(algorithm.update))
    updated, _ = update(state, generation_key, Transition(second=second))

    weights = jnp.array([2.0, -2.0]) / (first_reward.std() * cfg.population_size * cfg.sigma)
    expected_kernel = kernel + jnp.einsum("p,pio->io", weights, kernel_noise)
    expected_bias = bias + weights @ direction.bias
    np.testing.assert_allclose(updated.params["dense_0"]["kernel"], expected_kernel, atol=1e-6)
    np.testing.assert_allclose(updated.params["dense_0"]["bias"], expected_bias, atol=1e-6)


def test_continuous_policy_maps_actions_to_environment_bounds():
    space = Space(shape=(2,), low=(-3.0, 1.0), high=(5.0, 7.0))
    algorithm = EGGROLL(EGGROLLConfig(), LowRankPerturbedMLP((2,)), space, optax.sgd(1.0))
    timestep = Timestep(obs=jnp.zeros((64, 1)), reward=jnp.zeros(64))
    key = jax.random.key(0)
    state, carry = algorithm.init(key, timestep)
    _, _, actions, _ = jax.jit(algorithm.step, static_argnames="temperature")(
        state, carry, key, timestep, temperature=0.0
    )
    np.testing.assert_allclose(actions, jnp.tile(jnp.array([1.0, 4.0]), (64, 1)))


def test_input_support_filters_each_mirrored_pair_and_excludes_resets():
    cfg = EGGROLLConfig(population_size=4, rank=2, sigma=0.2)
    space = Space(dtype=jnp.int32, high=1)
    network = LowRankPerturbedMLP((4, 2))
    timestep = Timestep(obs=jnp.zeros((4, 3)), reward=jnp.zeros(4))
    key = jax.random.key(71)
    algorithm = EGGROLL(cfg, network, space, optax.sgd(1))
    state, _ = algorithm.init(key, timestep)
    obs = jnp.array([[[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]],
                     [[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0]]])
    reward = jnp.array([[4., 1., 2., 3.], [100., 0., 100., 0.]])
    done = jnp.array([[True, False, True, False], [False] * 4])
    transitions = Transition(first=Timestep(obs=obs), second=Timestep(
        obs=None, reward=reward, terminated=done, truncated=jnp.zeros_like(done)
    ))
    noise = algorithm.draw(state.params, state.noise_key)[0]
    update_weight = jnp.array([3., -1.]) / (reward[0].std() * cfg.population_size * cfg.sigma)
    masks = {
        "none": [[1, 1, 1], [1, 1, 1]],
        "pair": [[1, 1, 0], [0, 1, 0]],
        "population": [[1, 1, 0], [1, 1, 0]],
    }
    for mode, mask in masks.items():
        method = EGGROLL(cfg.replace(input_support=mode), network, space, optax.sgd(1))
        updated, _ = jax.jit(lox.spool(method.update))(state, key, transitions)
        expected = jnp.einsum("p,pi,pio->io", update_weight, jnp.array(mask), noise.inputs @ noise.outputs)
        np.testing.assert_allclose(updated.params["dense_0"]["kernel"],
                                   state.params["dense_0"]["kernel"] + expected, atol=1e-6)
