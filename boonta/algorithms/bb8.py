"""BB-8 (Batch-Balanced on-policy RL): measuring the gradient noise scale of an on-policy update.

The probe splits a batch of `num_envs` environment columns into `num_groups` groups of whole T-step segments. The
segments inside a group are correlated, but the groups are independent. It takes each group's loss gradient at the
pre-update parameters, maps it through `optimizer.update` with the optimizer state frozen, and estimates, for the
raw gradients and for the would-be updates,

    noise  = b · Σ_i ‖x_i − x̄‖² / (m − 1)       an estimate of tr(Σ) per environment segment
    signal = ‖x̄‖² − noise / B                    an estimate of ‖E[x]‖²
    B_simple = noise / signal                      in environments; times T in transitions

with b = B / m environments per group. Signal and noise are smoothed separately by an exponential moving average
before the ratio is taken.

`update` is the full would-be update u(g_i), which includes the part of Adam's step that comes from its frozen
first moment. That part is the same for every group, so it adds to the signal and not to the noise. `fresh` is
u(g_i) − u(0), the part of the update that the group's gradient causes.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from boonta.utils.typing import Array, PyTree

SPACES = ("gradient", "update", "fresh")


@dataclass(frozen=True)
class BB8Config:
    noise_scale_probe: bool = False
    num_groups: int = 64
    decay: float = 0.9


def group_columns(tree: PyTree, num_groups: int) -> PyTree:
    """(T, N, ...) leaves to (num_groups, T · N / num_groups, ...): contiguous environment columns per group."""

    def split(leaf):
        num_steps, num_envs, *rest = leaf.shape
        assert num_envs % num_groups == 0, f"{num_envs} environments do not split into {num_groups} groups"
        leaf = leaf.reshape(num_steps, num_groups, num_envs // num_groups, *rest)
        leaf = jnp.moveaxis(leaf, 1, 0)
        return leaf.reshape(num_groups, num_steps * (num_envs // num_groups), *rest)

    return jax.tree.map(split, tree)


def squared_norm(tree: PyTree) -> Array:
    return sum(jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in jax.tree.leaves(tree))


def noise_and_signal(stacked: PyTree, batch_size: float) -> tuple[Array, Array]:
    """Unbiased tr(Σ) per sample and ‖E[x]‖² from `m` group means of `batch_size / m` samples each."""
    leaves = jax.tree.leaves(stacked)
    num_groups = leaves[0].shape[0]
    group_size = batch_size / num_groups
    mean = jax.tree.map(lambda leaf: jnp.mean(leaf.astype(jnp.float32), axis=0), stacked)
    spread = sum(
        jnp.sum(jnp.square(leaf.astype(jnp.float32) - centre))
        for leaf, centre in zip(leaves, jax.tree.leaves(mean))
    )
    noise = group_size * spread / (num_groups - 1)
    signal = squared_norm(mean) - noise / batch_size
    return noise, signal


def frozen_updates(
    optimizer: optax.GradientTransformation, stacked: PyTree, optimizer_state: optax.OptState, params: PyTree
) -> tuple[PyTree, PyTree]:
    """Each group's would-be update with the optimizer state held fixed, and that minus the update of a zero gradient."""

    def update(grads):
        updates, _ = optimizer.update(grads, optimizer_state, params)
        return updates

    updates = jax.vmap(update)(stacked)
    baseline = update(jax.tree.map(jnp.zeros_like, params))
    fresh = jax.tree.map(lambda u, u0: u - u0[None], updates, baseline)
    return updates, fresh


def init(names: tuple[str, ...]) -> PyTree:
    zeros = {name: {space: jnp.zeros((), jnp.float32) for space in SPACES} for name in names}
    return {"count": jnp.zeros((), jnp.int32), "noise": zeros, "signal": jax.tree.map(jnp.zeros_like, zeros)}


def probe(
    cfg: BB8Config,
    state: PyTree,
    gradients: dict[str, PyTree],
    optimizer: optax.GradientTransformation,
    optimizer_state: optax.OptState,
    params: PyTree,
    num_envs: int,
    num_steps: int,
) -> tuple[PyTree, dict[str, Array]]:
    """Noise and signal of each loss's group gradients and would-be updates; returns the smoothed state and logs.

    `gradients` maps a loss name to its group gradients, stacked on a leading axis of `cfg.num_groups`.
    """
    count = state["count"] + 1
    correction = 1.0 - cfg.decay ** count.astype(jnp.float32)
    noise_state, signal_state, logs = {}, {}, {}
    for name, stacked in gradients.items():
        updates, fresh = frozen_updates(optimizer, stacked, optimizer_state, params)
        logs[f"bb8/{name}/group_gradient_norm"] = jnp.mean(jax.vmap(lambda g: jnp.sqrt(squared_norm(g)))(stacked))
        noise_state[name], signal_state[name] = {}, {}
        for space, values in zip(SPACES, (stacked, updates, fresh)):
            noise, signal = noise_and_signal(values, num_envs)
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
