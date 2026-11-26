import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.scipy.special import erf, erfc

from boonta import environments
from boonta.algorithms.eggroll import EGGROLL, EGGROLLConfig
from boonta.environments.wrappers import SameStepAutoReset
from boonta.networks import Categorical
from boonta.networks.low_rank_perturbations import LowRankPerturbedMLP
from boonta.utils import load_checkpoint


def truncated_mean(lower, upper):
    root_two = jnp.sqrt(2.0)
    density_difference = (jnp.exp(-0.5 * lower**2) - jnp.exp(-0.5 * upper**2)) / jnp.sqrt(2 * jnp.pi)
    near = jnp.minimum(jnp.abs(lower), jnp.abs(upper))
    far = jnp.maximum(jnp.abs(lower), jnp.abs(upper))
    tail_mass = 0.5 * (erfc(near / root_two) - erfc(far / root_two))
    mass = jnp.where(
        (lower >= 0) | (upper <= 0), tail_mass,
        0.5 * (erf(upper / root_two) - erf(lower / root_two)),
    )
    mean = density_difference / jnp.maximum(mass, 1e-30)
    return jnp.where(upper - lower < 1e-3, (lower + upper) * 0.5, jnp.clip(mean, lower, upper))


def interval(logits, actions, slope, z, alive):
    chosen = actions[..., None] == jnp.arange(logits.shape[-1])
    best = logits.max(axis=-1, keepdims=True)
    runner_up = jnp.where(chosen, -jnp.inf, logits).max(axis=-1, keepdims=True)
    threshold = z + (jnp.where(chosen, runner_up, best) - logits) / jnp.where(slope != 0, slope, 1)
    lower_bound = chosen == (slope > 0)
    active = alive[..., None] & (slope != 0)
    lower = jnp.where(active & lower_bound, threshold, -jnp.inf).max(axis=(1, 2))
    upper = jnp.where(active & ~lower_bound, threshold, jnp.inf).min(axis=(1, 2))
    return lower, upper


def make_batch(env, algorithm, mlp, horizon, conditioned, normalize_fitness=False):
    pairs = algorithm.cfg.population_size // 2
    sigma = algorithm.cfg.sigma
    last_hidden = f"dense_{len(mlp.widths) - 2}"

    @jax.jit
    def batch(params, key):
        noise_key, reset_key, step_key = jax.random.split(key, 3)
        noise = algorithm.draw(params, noise_key)
        env_state, timestep = env.init(reset_key)
        inputs = noise[-1].inputs[..., 0]
        z = noise[-1].outputs[:, 0, :]
        zeros = jnp.zeros_like(z)
        shape = (pairs, 2, -1)

        def step(carry, step_key):
            env_state, timestep, alive, total, lower, upper, prefix = carry
            args = ({"params": params}, timestep.obs, noise)
            if conditioned:
                logits, variables = mlp.apply(
                    *args, sigma=sigma, mutable=["intermediates"],
                    capture_intermediates=lambda module, method: module.name == last_hidden and method == "__call__",
                )
                hidden = jnp.tanh(variables["intermediates"][last_hidden]["__call__"][0])
                hidden = hidden.reshape(*shape, hidden.shape[-1])
                slope = jnp.einsum("psbi,pi->psb", hidden, inputs) * sigma
                slope *= jnp.array([1, -1])[None, :, None]
            else:
                logits = mlp.apply(*args, sigma=sigma)
            actions = jnp.argmax(logits, axis=-1)
            if conditioned:
                lo, hi = interval(
                    logits.reshape(*shape, z.shape[-1]), actions.reshape(shape),
                    slope[..., None], z[:, None, None, :], alive.reshape(shape),
                )
                lower, upper = jnp.maximum(lower, lo), jnp.minimum(upper, hi)
            env_state, timestep = env.step(step_key, env_state, actions)
            scores = jnp.where(alive, timestep.reward, 0).reshape(shape).mean(axis=2)
            diff = scores[:, 0] - scores[:, 1]
            if conditioned and not normalize_fitness:
                prefix += diff[:, None] * truncated_mean(lower, upper)
            return (
                env_state, timestep, alive & ~timestep.done, total + scores, lower, upper, prefix,
            ), None

        start = (env_state, timestep, jnp.ones_like(timestep.done), jnp.zeros((pairs, 2)),
                 jnp.full_like(z, -jnp.inf), jnp.full_like(z, jnp.inf), zeros)
        (_, _, _, total, lower, upper, prefix), _ = jax.lax.scan(
            step, start, jax.random.split(step_key, horizon)
        )
        scale = 1 / (2 * pairs * sigma)
        if normalize_fitness:
            scale /= jnp.maximum(total.std(), 1e-8)
        total = total[:, 0] - total[:, 1]
        pieces = []
        for layer in noise[:-1]:
            kernel = jnp.einsum("pi,po->io", layer.inputs[..., 0], total[:, None] * layer.outputs[:, 0])
            pieces.extend([kernel.ravel(), total @ layer.bias])
        bias = total @ noise[-1].bias
        head = (inputs.T @ (total[:, None] * z)).ravel()
        raw = jnp.concatenate([*pieces, bias, head]) * scale
        if not conditioned:
            return raw
        violation = jnp.maximum((lower - z).max(), (z - upper).max())
        terminal = (inputs.T @ (total[:, None] * truncated_mean(lower, upper))).ravel()
        return raw, terminal * scale, (inputs.T @ prefix).ravel() * scale, violation

    return batch


def measure(estimator, params, keys):
    jax.block_until_ready(estimator(params, keys[0]))
    start = time.monotonic()
    rows = [jax.device_get(estimator(params, key)) for key in keys]
    return jax.tree.map(lambda *xs: np.stack(xs), *rows), (time.monotonic() - start) / len(keys)


def report(raw, terminal, prefix, reference, head_width):
    head_size = terminal.shape[1]
    raw_head = raw[:, -head_size:]
    rest = raw[:, :-head_size].astype(np.float64)
    base_variance = raw.astype(np.float64).var(axis=0, ddof=1).sum()
    head_variance = raw_head.astype(np.float64).var(axis=0, ddof=1).sum()
    bias_variance = rest[:, -head_width:].var(axis=0, ddof=1).sum()
    ref = reference.mean(axis=0, dtype=np.float64)
    results = {}
    for name, head in (("raw", raw_head), ("terminal", terminal), ("prefix", prefix)):
        if head is None:
            continue
        head = head.astype(np.float64)
        delta = head_variance - head.var(axis=0, ddof=1).sum()
        dot = rest @ ref[:-head_size] + head @ ref[-head_size:]
        norm = np.sqrt((rest**2).sum(1) + (head**2).sum(1))
        cosine = dot / np.maximum(norm * np.linalg.norm(ref), 1e-30)
        single = raw_head[:, ::head_width].astype(np.float64).var(axis=0, ddof=1).sum()
        single -= head[:, ::head_width].var(axis=0, ddof=1).sum()
        centered_raw = raw_head - raw_head.mean(axis=0, dtype=np.float64)
        centered_new = head - head.mean(axis=0)
        differences = (centered_raw**2 - centered_new**2).sum(axis=1)
        error = 100 * differences.std(ddof=1) / np.sqrt(len(differences)) / base_variance
        results[name] = dict(whole_reduction_se_pct=error,
                             max_kernel_gain_pct=100 * head_variance / base_variance,
                             head_reduction_pct=100 * delta / (head_variance + bias_variance),
                             whole_reduction_pct=100 * delta / base_variance,
                             one_coordinate_whole_reduction_pct=100 * single / base_variance,
                             reference_cosine=cosine.mean())
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id")
    parser.add_argument("run", type=Path)
    parser.add_argument("steps", nargs="+", type=int)
    parser.add_argument("--batches", type=int, default=256)
    parser.add_argument("--reference-batches", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--normalize-fitness", action="store_true")
    args = parser.parse_args()
    single = environments.make("gymnax", args.env_id)
    space = single.action_space()
    env = SameStepAutoReset(single, 256)
    mlp = LowRankPerturbedMLP((64, 64, space.num_actions))
    algorithm = EGGROLL(EGGROLLConfig(), Categorical(mlp), optax.sgd(1))
    raw_batch = make_batch(env, algorithm, mlp, args.horizon, False, args.normalize_fitness)
    conditioned_batch = make_batch(env, algorithm, mlp, args.horizon, True, args.normalize_fitness)
    keys = jax.random.split(jax.random.key(771), args.batches)
    ref_keys = jax.random.split(jax.random.key(772), args.reference_batches)
    for step in args.steps:
        params = load_checkpoint(str(args.run / "checkpoints/latest" / str(step) / "algorithm_state"))["params"]
        reference, raw_time = measure(raw_batch, params, ref_keys)
        (raw, terminal, prefix, violation), cond_time = measure(conditioned_batch, params, keys)
        np.testing.assert_allclose(raw[0], raw_batch(params, keys[0]), rtol=1e-4, atol=1e-5)
        assert np.isfinite(terminal).all() and np.isfinite(prefix).all()
        assert violation.max() < 1e-4, violation.max()
        results = report(raw, terminal, None if args.normalize_fitness else prefix, reference, space.num_actions)
        print(json.dumps(dict(environment=args.env_id, step=step, normalized=args.normalize_fitness, raw_ms=raw_time * 1000,
                              conditioned_ms=cond_time * 1000, results=results)), flush=True)


if __name__ == "__main__":
    main()
