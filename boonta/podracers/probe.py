"""Signal accounting for one policy update, measured alongside training without touching it.

For the update pi_t -> pi_{t+1}, with u = log pi_{t+1} - log pi_t:

    captured signal   C = E_s[Cov_pi(u(s, .), A(s, .))]     the first-order change in J
    movement          delta = E_s[KL(pi_t || pi_{t+1})]  ~  1/2 E_s[Var_pi(u)]
    available signal  S = E_s[Var_pi(A(s, .))]
    bound             C <= sqrt(2 delta S),  efficiency rho = C / sqrt(2 delta S)

Every quantity is measured on from-start samples only (info["rho"]), never on restarted
segments, and on non-truncated steps. The probe keeps the previous update's parameters, so the
rollout collected by pi_t is scored under pi_{t-1} too: that gives the held-out captured signal
and the movement of update t-1, measured on data the update never saw. The in-batch captured
signal of update t is measured on rollout t. All keys are folded off the training keys, so the
probe draws no randomness from training.
"""

from dataclasses import dataclass
from collections.abc import Callable

import jax
import jax.numpy as jnp
import lox
from flax import struct

from boonta.algorithms.advantage_estimators import generalized_advantage_estimator
from boonta.algorithms.ppo import feature_rank_metrics
from boonta.utils import PyTree, Transition
from boonta.utils.typing import Array, Key


@struct.dataclass
class ProbeState:
    previous_params: PyTree
    ledger: PyTree
    reach: Array
    length: Array
    out_mean: Array
    in_mean: Array
    previous_in: Array
    updates: Array


@dataclass
class SignalProbe:
    network: object
    gamma: float
    gae_lambda: float
    archive_size: int
    num_actions: int
    rate: float = 0.2
    smoothing: float = 0.1
    rank_every: int = 50
    rank_samples: int = 4096
    bucket_fn: Callable | None = None
    num_buckets: int = 0

    def __post_init__(self):
        from boonta.curricula.psr import Advantage

        self.advantage = Advantage(gamma=self.gamma, gae_lambda=self.gae_lambda, rate=self.rate)

    def init(self, params: PyTree) -> ProbeState:
        zero = jnp.zeros((), jnp.float32)
        return ProbeState(
            previous_params=params,
            ledger=self.advantage.init(self.archive_size, self.num_actions),
            reach=jnp.zeros(self.archive_size, jnp.float32),
            length=zero,
            out_mean=zero,
            in_mean=zero,
            previous_in=zero,
            updates=jnp.zeros((), jnp.int32),
        )

    def policy(self, params, obs):
        dist, value = self.network.apply(params, obs, temperature=1.0)
        return jax.nn.log_softmax(dist.logits.astype(jnp.float32), axis=-1), value[..., 0]

    def update(self, state: ProbeState, key: Key, transitions: Transition, before: PyTree, after: PyTree) -> ProbeState:
        from boonta.curricula.psr import signal, wipe

        first, second = transitions.first, transitions.second
        actions = second.action.astype(jnp.int32)
        mask = first.info["rho"].astype(bool) & ~second.truncated.astype(bool)
        count = jnp.maximum(jnp.sum(mask, dtype=jnp.float32), 1.0)

        def mean(value):
            return jnp.sum(jnp.where(mask, value, 0.0)) / count

        # Raw GAE advantages of this rollout in the recipe's reward units, bootstrapped as PPO does.
        _, last_value = self.policy(before, jax.tree.map(lambda leaf: leaf[-1], second.obs))
        advantages, _ = generalized_advantage_estimator(
            transitions, transitions.aux["value"], last_value, self.gamma, self.gae_lambda
        )
        now = jax.nn.log_softmax(transitions.aux["logits"].astype(jnp.float32), axis=-1)
        taken_now = jnp.take_along_axis(now, actions[..., None], axis=-1)[..., 0]
        after_log, _ = self.policy(after, first.obs)
        previous_log, _ = self.policy(state.previous_params, first.obs)
        taken_after = jnp.take_along_axis(after_log, actions[..., None], axis=-1)[..., 0]
        taken_previous = jnp.take_along_axis(previous_log, actions[..., None], axis=-1)[..., 0]

        c_in = (jnp.exp(taken_after - taken_now) - 1.0) * advantages
        c_out = (1.0 - jnp.exp(taken_previous - taken_now)) * advantages
        previous_probs = jnp.exp(previous_log)
        kl = jnp.sum(previous_probs * (previous_log - now), axis=-1)
        step = now - previous_log
        spread = jnp.sum(previous_probs * step**2, axis=-1) - jnp.sum(previous_probs * step, axis=-1) ** 2

        # The cell ledger: the debiased between-action variance of each cell, weighted by how
        # often from-start play reaches the cell.
        cells = first.info["cell"]
        wiped = wipe(jnp.reshape(second.info["evicted"], (-1,)), self.archive_size)
        ledger = self.advantage.update(self.advantage.forget(state.ledger, wiped), transitions)
        rows = jnp.reshape(jnp.where(mask & (cells >= 0), cells, self.archive_size), (-1,))
        visits = jnp.zeros(self.archive_size).at[rows].add(1.0, mode="drop")
        reach = jnp.where(wiped, 0.0, (1.0 - self.rate) * state.reach) + self.rate * visits
        weights = reach / jnp.maximum(jnp.sum(reach), 1e-30)
        s_cell = jnp.sum(weights * jnp.maximum(signal(ledger), 0.0))
        s_raw = mean(advantages**2)

        # From-start episodes ending in this rollout give the running episode length L.
        ended = second.info["rho_done"].astype(bool) & first.info["rho"].astype(bool)
        lengths = first.info["age"].astype(jnp.float32) + 1.0
        num_ended = jnp.sum(ended, dtype=jnp.float32)
        batch_length = jnp.sum(jnp.where(ended, lengths, 0.0)) / jnp.maximum(num_ended, 1.0)
        length = jnp.where(
            num_ended > 0.0,
            jnp.where(state.length > 0.0, (1.0 - self.smoothing) * state.length + self.smoothing * batch_length, batch_length),
            state.length,
        )
        discount = self.gamma ** first.info["age"].astype(jnp.float32)
        episodes = count / jnp.maximum(length, 1.0)

        started = state.updates > 0
        c_out_mean, c_in_mean, delta = mean(c_out), mean(c_in), mean(kl)

        def smooth(old, new):
            return jnp.where(state.updates > 1, (1.0 - self.smoothing) * old + self.smoothing * new, new)

        out_mean = jnp.where(started, smooth(state.out_mean, c_out_mean), state.out_mean)
        in_mean = jnp.where(started, smooth(state.in_mean, state.previous_in), state.in_mean)
        held = lambda value: jnp.where(started, value, jnp.nan)  # noqa: E731

        logs = {
            "signal/c_out": held(c_out_mean),
            "signal/c_in": c_in_mean,
            "signal/holdout_ratio": held(c_out_mean / jnp.where(state.previous_in != 0.0, state.previous_in, jnp.nan)),
            "signal/holdout_ratio_smoothed": held(out_mean / jnp.where(in_mean != 0.0, in_mean, jnp.nan)),
            "signal/delta": held(delta),
            "signal/half_var_u": held(0.5 * mean(spread)),
            "signal/s_raw": s_raw,
            "signal/s_cell": s_cell,
            "signal/rho_low": held(c_out_mean / jnp.sqrt(jnp.maximum(2.0 * delta * s_raw, 1e-30))),
            "signal/rho_high": held(c_out_mean / jnp.sqrt(jnp.maximum(2.0 * delta * s_cell, 1e-30))),
            "signal/c_gamma": held(jnp.sum(jnp.where(mask, discount * c_out, 0.0)) / jnp.maximum(episodes, 1e-30)),
            "signal/c_gamma_in": jnp.sum(jnp.where(mask, discount * c_in, 0.0)) / jnp.maximum(episodes, 1e-30),
            "signal/episode_length": length,
            "signal/from_start_share": jnp.mean(mask.astype(jnp.float32)),
        }

        if self.bucket_fn is not None:
            restarted = second.info["restarted"].astype(bool)
            buckets = jnp.clip(self.bucket_fn(second.obs), 0, self.num_buckets - 1)
            placed = jnp.maximum(jnp.sum(restarted, dtype=jnp.float32), 1.0)
            for bucket in range(self.num_buckets):
                logs[f"restart/bucket_{bucket}"] = jnp.where(
                    jnp.any(restarted), jnp.sum(restarted & (buckets == bucket), dtype=jnp.float32) / placed, jnp.nan
                )

        rank_key = jax.random.fold_in(key, 0x52414E4B)

        def measure():
            flat_mask = jnp.reshape(mask, (-1,))
            weights = flat_mask / jnp.maximum(jnp.sum(flat_mask), 1)
            picked = jax.random.choice(rank_key, flat_mask.shape[0], (self.rank_samples,), p=weights)
            obs = jax.tree.map(lambda leaf: jnp.reshape(leaf, (-1, *leaf.shape[2:]))[picked], first.obs)
            return feature_rank_metrics(self.network, after, obs)

        due = state.updates % self.rank_every == 0
        shapes = jax.eval_shape(measure)
        ranks = jax.lax.cond(
            due, measure, lambda: jax.tree.map(lambda leaf: jnp.full(leaf.shape, jnp.nan, leaf.dtype), shapes)
        )
        logs.update({name: jnp.where(due, value, jnp.nan) for name, value in ranks.items()})
        lox.log(logs)
        return state.replace(
            previous_params=before,
            ledger=ledger,
            reach=reach,
            length=length,
            out_mean=out_mean,
            in_mean=in_mean,
            previous_in=c_in_mean,
            updates=state.updates + 1,
        )
