import jax
import jax.numpy as jnp
import lox
import numpy as np
import optax
import pytest

from boonta.algorithms.eggroll import EGGROLL, EGGROLLConfig
from boonta.networks import Categorical, SquashedGaussian
from boonta.networks.low_rank_perturbations import LowRankPerturbedMLP
from boonta.utils import Timestep, Transition


def test_mirrored_low_rank_forward_and_update_match_dense_reference():
    cfg = EGGROLLConfig(population_size=4, rank=2, sigma=0.2)
    mlp = LowRankPerturbedMLP((2,))
    algorithm = EGGROLL(cfg, Categorical(mlp), optax.sgd(1.0))
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
    outputs = jax.jit(mlp.apply)({"params": state.params}, obs, noise, sigma=cfg.sigma)
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


@pytest.mark.parametrize("head", [Categorical, SquashedGaussian], ids=["discrete", "continuous"])
def test_at_temperature_zero_eggroll_plays_the_unperturbed_greedy_policy(head):
    mlp = LowRankPerturbedMLP((5, 4))
    algorithm = EGGROLL(EGGROLLConfig(population_size=4, temperature=1.0), head(mlp), optax.sgd(1.0))
    obs = jax.random.normal(jax.random.key(1), (8, 3))
    timestep = Timestep(obs=obs, reward=jnp.zeros(8))
    state, carry = algorithm.init(jax.random.key(0), timestep)
    step = jax.jit(algorithm.step)

    _, _, greedy, _ = step(state, carry, jax.random.key(2), timestep, 0.0)
    outputs = mlp.apply({"params": state.params}, obs)
    expected = outputs.argmax(-1) if head is Categorical else jnp.tanh(outputs[..., :2])
    np.testing.assert_allclose(greedy, expected, atol=1e-6)

    _, _, explored, _ = step(state, carry, jax.random.key(2), timestep, 1.0)
    assert not np.allclose(explored, greedy)
