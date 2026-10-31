"""BB-8 (Batch-Balanced on-policy RL): measuring the gradient noise scale of an on-policy update.

Units. The sample is a **segment**: one environment's T consecutive steps in one rollout (one column of a (T, N)
batch). A batch of one rollout has B = N segments; a large batch of R rollouts stacked with `stack_rollouts` has
B = R · N segments. Noise scales are in segments ("environments" in the logs and the report, since one rollout is one
segment per environment), and in transitions after multiplying by T.

Estimator (`noise_and_signal`, the only implementation; the online probe, the E1 large probe and the bias test all go
through `probe`, which calls it). The batch is split into m groups of b = B / m whole segments (`group_columns`; with
`stack_rollouts` a group holds whole environments across rollouts, so groups stay independent). From the m group
means x_i of a quantity x (a gradient or an update) and a squared norm ‖·‖²_M:

    noise  = b · Σ_i ‖x_i − x̄‖²_M / (m − 1)      unbiased tr(M Σ) per segment
    signal = ‖x̄‖²_M − noise / B                   unbiased ‖E[x]‖²_M
    B_simple = noise / signal                       in segments

Signal and noise are smoothed separately by an exponential moving average before the ratio is taken.

Spaces (what x is), per loss:
- `gradient`: the raw group gradient, Euclidean metric.
- `update`: optimizer.update(g_i) with the optimizer state frozen. Contains the frozen first moment's step, which is
  the same for every group: it adds to the signal and not to the noise, so this ratio stays flat.
- `fresh` (the noise scale BB-8 is about): the update map linearized at g = 0 with the whole optimizer state frozen,
  J_u(0) g_i, by `jax.jvp`. For Adam: −lr (1 − β1) / c1 · g / (√(β2 ν̂) + ε), linear in g. Gradient clipping is the
  identity at g = 0, so it is excluded by construction. Coordinates whose second moment is exactly 0 (parameters
  that have never had a gradient, so g is 0 there too) have a 0/0 tangent; it is set to 0 and counted
  (`bb8/<loss>/fresh_undefined`). For Lion the tangent of sign at 0 is 0, so `fresh` is undefined; use `difference`.
- `difference`: u(g_i) − u(0). Lets the group's own g² into Adam's second moment, which damps noisy (small) groups
  and biases the noise scale low at small B. A diagnostic only.
- `trust_region` (PPO): the fresh updates in the Fisher metric of the policy, ‖d‖²_F = E_s[Var_{a~π}(J_logits d)],
  computed by a JVP of the logits along d on a sample of the batch's states (2 KL to second order).
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from boonta.utils.typing import Array, PyTree

SPACES = ("gradient", "update", "fresh", "difference", "trust_region")


@dataclass(frozen=True)
class BB8Config:
    noise_scale_probe: bool = False
    num_groups: int = 64
    decay: float = 0.9
    # States sampled for the trust-region (Fisher) metric.
    fisher_states: int = 1024
    # PPO: normalize advantages by running moments instead of the batch's (for batches smaller than the default).
    running_advantage_normalization: bool = False
    # Running moments forget at rate exp(-n / horizon) per n transitions.
    advantage_horizon: float = 524_288.0


def stack_rollouts(rollouts: list[PyTree]) -> PyTree:
    """R rollouts of (T, N, ...) leaves to (T, N · R, ...), environment-major: column e · R + k is environment e in
    rollout k, so a group of contiguous columns holds whole environments across all rollouts."""
    return jax.tree.map(
        lambda *leaves: jnp.stack(leaves, axis=2).reshape(leaves[0].shape[0], -1, *leaves[0].shape[2:]), *rollouts
    )


def group_columns(tree: PyTree, num_groups: int) -> PyTree:
    """(T, B, ...) leaves to (num_groups, T · B / num_groups, ...): contiguous segments (columns) per group."""

    def split(leaf):
        num_steps, num_columns, *rest = leaf.shape
        assert num_columns % num_groups == 0, f"{num_columns} segments do not split into {num_groups} groups"
        leaf = leaf.reshape(num_steps, num_groups, num_columns // num_groups, *rest)
        leaf = jnp.moveaxis(leaf, 1, 0)
        return leaf.reshape(num_groups, num_steps * (num_columns // num_groups), *rest)

    return jax.tree.map(split, tree)


def squared_norm(tree: PyTree) -> Array:
    return sum(jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in jax.tree.leaves(tree))


def noise_and_signal(group_means: PyTree, num_segments: float, metric: Callable | None = None) -> tuple[Array, Array]:
    """Noise (tr(M Σ) per segment) and signal (‖E[x]‖²_M) from m group means of num_segments / m segments each.

    `group_means` has a leading axis of m groups; `metric` maps a pytree (no group axis) to its squared norm
    (Euclidean by default).
    """
    metric = metric or squared_norm
    num_groups = jax.tree.leaves(group_means)[0].shape[0]
    segments_per_group = num_segments / num_groups
    mean = jax.tree.map(lambda leaf: jnp.mean(leaf.astype(jnp.float32), axis=0), group_means)
    deviations = jax.tree.map(lambda leaf, centre: leaf.astype(jnp.float32) - centre[None], group_means, mean)
    spread = jnp.sum(jax.vmap(metric)(deviations))
    noise = segments_per_group * spread / (num_groups - 1)
    signal = metric(mean) - noise / num_segments
    return noise, signal


def frozen_updates(
    optimizer: optax.GradientTransformation, stacked: PyTree, optimizer_state: optax.OptState, params: PyTree
) -> tuple[PyTree, PyTree, PyTree, Array]:
    """For each group gradient, with the optimizer state frozen: the would-be update u(g_i); the fresh update
    J_u(0) g_i (JVP at zero); the difference u(g_i) − u(0); and the fraction of undefined (0/0) fresh coordinates."""

    def update(grads):
        updates, _ = optimizer.update(grads, optimizer_state, params)
        return updates

    zeros = jax.tree.map(jnp.zeros_like, params)
    updates = jax.vmap(update)(stacked)
    baseline = update(zeros)
    difference = jax.tree.map(lambda u, u0: u - u0[None], updates, baseline)
    fresh = jax.vmap(lambda g: jax.jvp(update, (zeros,), (g,))[1])(stacked)
    undefined = sum(jnp.sum(~jnp.isfinite(leaf)) for leaf in jax.tree.leaves(fresh))
    total = sum(leaf.size for leaf in jax.tree.leaves(fresh))
    fresh = jax.tree.map(lambda leaf: jnp.where(jnp.isfinite(leaf), leaf, 0.0), fresh)
    return updates, fresh, difference, undefined / total


def init_state(cfg: BB8Config, names: tuple[str, ...]) -> PyTree:
    """The algorithm's `bb8` state: probe moving averages and running advantage moments, or None when both are off."""
    state = {}
    if cfg.noise_scale_probe:
        state |= init(names)
    if cfg.running_advantage_normalization:
        state["advantage"] = {name: jnp.zeros((), jnp.float32) for name in ("weight", "mean", "square")}
    return state or None


def normalize_running(cfg: BB8Config, statistics: PyTree, advantages: Array) -> tuple[PyTree, Array]:
    """Normalizes by moments over recent batches (this one included), weighted by transitions."""
    count = jnp.asarray(advantages.size, jnp.float32)
    keep = jnp.exp(-count / cfg.advantage_horizon) * statistics["weight"]
    weight = keep + count
    mean = (keep * statistics["mean"] + count * advantages.mean()) / weight
    square = (keep * statistics["square"] + count * jnp.mean(jnp.square(advantages))) / weight
    std = jnp.sqrt(jnp.maximum(square - mean**2, 0.0))
    return {"weight": weight, "mean": mean, "square": square}, (advantages - mean) / (std + 1e-8)


def init(names: tuple[str, ...]) -> PyTree:
    zeros = {name: {space: jnp.zeros((), jnp.float32) for space in SPACES} for name in names}
    return {"count": jnp.zeros((), jnp.int32), "noise": zeros, "signal": jax.tree.map(jnp.zeros_like, zeros)}


def fisher_metric(apply_logits: Callable, params: PyTree, states: PyTree) -> Callable:
    """‖d‖²_F = E_s[Var_{a~π(·|s)}((J_logits d)_a)]: twice the KL of a step d to second order (categorical policy)."""
    logits = apply_logits(params, states)
    probabilities = jax.nn.softmax(logits, axis=-1)

    def metric(direction):
        _, tangent = jax.jvp(lambda p: apply_logits(p, states), (params,), (direction,))
        centred = tangent - jnp.sum(probabilities * tangent, axis=-1, keepdims=True)
        return jnp.mean(jnp.sum(probabilities * jnp.square(centred), axis=-1))

    return metric


def probe(
    cfg: BB8Config,
    state: PyTree,
    gradients: dict[str, PyTree],
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    params: PyTree,
    num_segments: int,
    num_steps: int,
    fisher: Callable | None = None,
) -> tuple[PyTree, dict[str, Array]]:
    """Noise and signal of each loss's group gradients and would-be updates; returns the smoothed state and logs.

    `gradients` maps a loss name to its group gradients, stacked on a leading axis of `cfg.num_groups`;
    `num_segments` is B (columns of the batch); `fisher` is the policy's Fisher metric, or None.
    """
    count = state["count"] + 1
    correction = 1.0 - cfg.decay ** count.astype(jnp.float32)
    noise_state, signal_state, logs = {}, {}, {}
    for name, stacked in gradients.items():
        updates, fresh, difference, undefined = frozen_updates(optimizer, stacked, optimizer_state, params)
        logs[f"bb8/{name}/group_gradient_norm"] = jnp.mean(jax.vmap(lambda g: jnp.sqrt(squared_norm(g)))(stacked))
        logs[f"bb8/{name}/fresh_undefined"] = undefined
        noise_state[name], signal_state[name] = {}, {}
        for space in SPACES:
            if space == "trust_region":
                if fisher is None:
                    noise_state[name][space] = state["noise"][name][space]
                    signal_state[name][space] = state["signal"][name][space]
                    continue
                noise, signal = noise_and_signal(fresh, num_segments, fisher)
            else:
                values = {"gradient": stacked, "update": updates, "fresh": fresh, "difference": difference}[space]
                noise, signal = noise_and_signal(values, num_segments)
            smoothed_noise = cfg.decay * state["noise"][name][space] + (1.0 - cfg.decay) * noise
            smoothed_signal = cfg.decay * state["signal"][name][space] + (1.0 - cfg.decay) * signal
            noise_state[name][space], signal_state[name][space] = smoothed_noise, smoothed_signal
            scale = (smoothed_noise / correction) / (smoothed_signal / correction)
            prefix = f"bb8/{name}/{space}"
            logs |= {
                f"{prefix}/noise": noise,
                f"{prefix}/signal": signal,
                f"{prefix}/noise_scale_envs_raw": noise / signal,
                f"{prefix}/noise_scale_envs": scale,
                f"{prefix}/noise_scale_transitions": scale * num_steps,
            }
    return {"count": count, "noise": noise_state, "signal": signal_state}, logs
