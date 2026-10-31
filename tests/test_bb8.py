import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from boonta.algorithms import bb8

DIM = 64
NUM_ENVS = 4096
NUM_GROUPS = 64


def quadratic(key, correlated_steps=1):
    """Per-environment gradients of L(θ) = ½‖A(θ − θ*)‖² at θ = 0, with Gaussian noise of known covariance.

    With `correlated_steps` T > 1 an environment's gradient is the mean over T steps that share one segment-level
    draw z and add independent step noise ξ, so its covariance is Σ_z + Σ_ξ / T.
    """
    a_key, target_key, scale_key, step_key = jax.random.split(key, 4)
    a = jax.random.normal(a_key, (DIM, DIM)) / jnp.sqrt(DIM)
    target = jax.random.normal(target_key, (DIM,))
    mean = a.T @ a @ (0.0 - target)
    segment_std = jnp.exp(jax.random.uniform(scale_key, (DIM,), minval=-1.0, maxval=1.0))
    step_std = jnp.exp(jax.random.uniform(step_key, (DIM,), minval=-1.0, maxval=1.0))
    covariance = segment_std**2 + step_std**2 / correlated_steps
    true_scale = float(jnp.sum(covariance) / jnp.sum(mean**2))
    # Noise scale near 100 environments, so the signal estimate is well above its own error at B = 4096.
    factor = jnp.sqrt(100.0 / true_scale)

    def sample(key):
        segment_key, step_key = jax.random.split(key)
        z = segment_std * jax.random.normal(segment_key, (NUM_ENVS, DIM))
        xi = step_std * jax.random.normal(step_key, (correlated_steps, NUM_ENVS, DIM))
        return mean + factor * (z[None] + xi)  # (T, N, DIM): a rollout of per-step gradients

    return sample, true_scale * factor**2


def group_means(per_step):
    """Group gradient = mean over the group's T × N / m steps, with groups made of whole environment columns."""
    return jnp.mean(bb8.group_columns(per_step, NUM_GROUPS), axis=1)


@pytest.mark.parametrize("correlated_steps", [1, 16])
def test_the_estimator_recovers_the_noise_scale_of_a_quadratic(correlated_steps):
    sample, true_scale = quadratic(jax.random.key(0), correlated_steps)
    noises, signals = [], []
    for key in jax.random.split(jax.random.key(1), 64):
        noise, signal = bb8.noise_and_signal(group_means(sample(key)), NUM_ENVS)
        noises.append(noise)
        signals.append(signal)
    estimate = float(np.mean(noises) / np.mean(signals))
    assert estimate == pytest.approx(true_scale, rel=0.1)


def test_the_smoothed_probe_matches_and_an_sgd_update_has_the_noise_scale_of_its_gradient():
    sample, true_scale = quadratic(jax.random.key(2))
    params = jnp.zeros(DIM)
    optimizer = optax.sgd(0.1)
    cfg = bb8.BB8Config(noise_scale_probe=True, num_groups=NUM_GROUPS, decay=0.9)
    state = bb8.init(("loss",))
    for key in jax.random.split(jax.random.key(3), 64):
        state, logs = bb8.probe(cfg, state, {"loss": group_means(sample(key))}, optimizer, optimizer.init(params),
                                params, NUM_ENVS, 1)
    gradient = float(logs["bb8/loss/gradient/noise_scale_envs"])
    assert gradient == pytest.approx(true_scale, rel=0.1)
    assert float(logs["bb8/loss/update/noise_scale_envs"]) == pytest.approx(gradient, rel=1e-4)
    assert float(logs["bb8/loss/fresh/noise_scale_envs"]) == pytest.approx(gradient, rel=1e-4)
    assert float(logs["bb8/loss/difference/noise_scale_envs"]) == pytest.approx(gradient, rel=1e-4)


def test_a_frozen_adam_update_has_the_noise_scale_of_the_preconditioned_gradient():
    sample, _ = quadratic(jax.random.key(4))
    params = jnp.zeros(DIM)
    optimizer = optax.adam(1e-3)
    # A late optimizer state: second moments of very different sizes per coordinate and a first moment of zero, so
    # u(g) − u(0) ≈ −lr (1 − β1) g / √ν̂ while (1 − β2) g² is small against ν.
    nu = jnp.exp(jax.random.uniform(jax.random.key(5), (DIM,), minval=-4.0, maxval=4.0)) * 1e4
    adam = optax.ScaleByAdamState(count=jnp.asarray(10_000, jnp.int32), mu=jnp.zeros(DIM), nu=nu)
    optimizer_state = (adam, optax.EmptyState())
    preconditioner = 1.0 / (jnp.sqrt(nu / (1.0 - 0.999**10_001)) + 1e-8)
    noises, signals, fresh_noises, fresh_signals = [], [], [], []
    for key in jax.random.split(jax.random.key(6), 32):
        grads = group_means(sample(key))
        noise, signal = bb8.noise_and_signal(grads * preconditioner, NUM_ENVS)
        _, fresh, _ = bb8.frozen_updates(optimizer, grads, optimizer_state, params)
        fresh_noise, fresh_signal = bb8.noise_and_signal(fresh, NUM_ENVS)
        noises.append(noise), signals.append(signal)
        fresh_noises.append(fresh_noise), fresh_signals.append(fresh_signal)
    expected = np.mean(noises) / np.mean(signals)
    assert np.mean(fresh_noises) / np.mean(fresh_signals) == pytest.approx(expected, rel=0.1)


def test_groups_are_whole_environment_columns():
    rollout = jnp.arange(4 * 8).reshape(4, 8)  # (T, N): value = t * N + env
    groups = bb8.group_columns(rollout, 2)
    environments = np.asarray(groups) % 8
    assert groups.shape == (2, 16)
    assert set(environments[0]) == {0, 1, 2, 3} and set(environments[1]) == {4, 5, 6, 7}


def test_the_fresh_update_freezes_adams_second_moment_so_it_does_not_depend_on_the_group_size():
    """With ν fixed the fresh update is linear in g, so a large group gradient is not damped by its own g²."""
    params = jnp.zeros(DIM)
    optimizer = optax.adam(1e-3)
    nu = jnp.full(DIM, 1e-4)  # small second moment: (1 - β2) g² of a noisy group (g ~ 1) would dominate it
    adam = optax.ScaleByAdamState(count=jnp.asarray(100, jnp.int32), mu=jnp.zeros(DIM), nu=nu)
    grads = jax.random.normal(jax.random.key(7), (8, DIM))
    clipped = optax.chain(optax.clip_by_global_norm(0.5), optimizer)  # as the recipes build it
    _, fresh, difference = bb8.frozen_updates(clipped, grads, (optax.EmptyState(), (adam, optax.EmptyState())), params)
    linear = -1e-3 * 0.1 / (1 - 0.9**101) * grads / (jnp.sqrt(0.999 * nu / (1 - 0.999**101)) + 1e-8)
    np.testing.assert_allclose(fresh, linear, rtol=1e-3)
    assert float(jnp.max(jnp.abs(difference))) < 0.5 * float(jnp.max(jnp.abs(fresh)))
