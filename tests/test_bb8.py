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
        _, fresh, _, _ = bb8.frozen_updates(optimizer, grads, optimizer_state, params)
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
    _, fresh, difference, _ = bb8.frozen_updates(clipped, grads, (optax.EmptyState(), (adam, optax.EmptyState())), params)
    linear = -1e-3 * 0.1 / (1 - 0.9**101) * grads / (jnp.sqrt(0.999 * nu / (1 - 0.999**101)) + 1e-8)
    np.testing.assert_allclose(fresh, linear, rtol=1e-3)
    assert float(jnp.max(jnp.abs(difference))) < 0.5 * float(jnp.max(jnp.abs(fresh)))


def caller_batches(path, key, num_groups, num_segments, num_steps, num_rollouts):
    """The group means each caller builds: online (one rollout), large (R rollouts stacked env-major), subset (column
    subsets of a larger batch, as the bias test does). Returns a list of (group means, B)."""
    base, *keys = jax.random.split(key, num_rollouts + 1)  # `base` fixes the mean gradient for every rollout
    if path == "online":
        batch = segments_with_mean(base, keys[0], num_steps, num_segments)
        return [(bb8.group_columns(batch, num_groups).mean(axis=1), num_segments)]
    if path == "large":
        per = num_segments // num_rollouts
        batch = bb8.stack_rollouts([segments_with_mean(base, k, num_steps, per) for k in keys])
        return [(bb8.group_columns(batch, num_groups).mean(axis=1), num_segments)]
    batch = segments_with_mean(base, keys[0], num_steps, 2 * num_segments)
    return [(bb8.group_columns(batch[:, i * num_segments:(i + 1) * num_segments], num_groups).mean(axis=1), num_segments)
            for i in range(2)]


def segments_with_mean(base, key, num_steps, num_envs, dim=16, true_scale=100.0):
    """Per-step gradients (T, N, dim) with mean from `base` and a per-segment noise scale of `true_scale` segments: a
    shared draw per segment (std s) plus step noise (std s √T), so the per-segment covariance is 2 s² I for any T."""
    mean = jax.random.normal(base, (dim,))
    std = jnp.sqrt(true_scale * jnp.sum(mean**2) / (2 * dim))
    segment_key, step_key = jax.random.split(key)
    z = std * jax.random.normal(segment_key, (num_envs, dim))
    xi = std * jnp.sqrt(num_steps) * jax.random.normal(step_key, (num_steps, num_envs, dim))
    return mean + z[None] + xi


CALLERS = [("online", 1), ("subset", 1), ("large", 1), ("large", 4)]


@pytest.mark.parametrize("path, num_rollouts", CALLERS)
@pytest.mark.parametrize("num_groups", [16, 64])
@pytest.mark.parametrize("num_segments", [256, 8192])
@pytest.mark.parametrize("num_steps", [1, 16])
def test_every_caller_recovers_the_noise_scale_in_segments(path, num_rollouts, num_groups, num_segments, num_steps):
    """The estimate (in segments) does not depend on the caller, m, B, T or the number of rollouts."""
    noises, signals = [], []
    repeats = 48 if num_segments == 256 else 12
    for key in jax.random.split(jax.random.key(11), repeats):
        for group_means, batch in caller_batches(path, key, num_groups, num_segments, num_steps, num_rollouts):
            noise, signal = bb8.noise_and_signal(group_means, batch)
            noises.append(noise)
            signals.append(signal)
    assert float(np.mean(noises) / np.mean(signals)) == pytest.approx(100.0, rel=0.12)


def test_the_probe_reports_transitions_as_segments_times_t():
    group_means = bb8.group_columns(segments_with_mean(jax.random.key(0), jax.random.key(1), 16, 256), 16).mean(axis=1)
    state, logs = bb8.probe(bb8.BB8Config(noise_scale_probe=True, num_groups=16), bb8.init(("x",)), {"x": group_means},
                            optax.sgd(1.0), optax.sgd(1.0).init(jnp.zeros(16)), jnp.zeros(16), 256, 16)
    assert float(logs["bb8/x/gradient/noise_scale_transitions"]) == pytest.approx(
        16 * float(logs["bb8/x/gradient/noise_scale_envs"]), rel=1e-6)
