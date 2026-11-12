from collections.abc import Callable
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from boonta.utils import Timestep, Transition, canonicalize_dtype, flatten, remove_feature_axis
from boonta.utils.typing import Array, Key, PyTree

from .advantage_estimators import generalized_advantage_estimator


@struct.dataclass(frozen=True)
class PPOConfig:
    num_minibatches: int
    update_epochs: int
    clip_coefficient: float
    clip_value_loss: bool
    entropy_coefficient: float
    value_coefficient: float
    gamma: float
    gae_lambda: float
    normalize_advantage: bool = False
    # States sampled from each batch to log the rank of the encoder's features after the update;
    # 0 turns the measurement off. It only logs.
    feature_rank: int = 0


def rank_metrics(features: Array) -> dict:
    """Rank measures of a (samples, units) feature matrix, centred over the samples: the entropy
    rank exp(H(variance shares)), n99 (eigen-directions for 99% of the variance), srank (singular
    values for 99% of their sum, Kumar et al. 2021), and the share of units never above zero."""
    features = features.astype(jnp.float32)
    centred = features - jnp.mean(features, axis=0)
    singular = jnp.linalg.svd(centred, compute_uv=False)
    variance = singular**2
    share = variance / jnp.maximum(jnp.sum(variance), 1e-30)
    entropy = jnp.exp(-jnp.sum(jnp.where(share > 0.0, share * jnp.log(jnp.maximum(share, 1e-30)), 0.0)))
    n99 = 1 + jnp.argmax(jnp.cumsum(share) >= 0.99)
    srank = 1 + jnp.argmax(jnp.cumsum(singular / jnp.maximum(jnp.sum(singular), 1e-30)) >= 0.99)
    dead = jnp.mean(jnp.all(features <= 0.0, axis=0).astype(jnp.float32))
    return {"entropy": entropy, "n99": n99.astype(jnp.float32), "srank": srank.astype(jnp.float32), "dead": dead}


def log_feature_rank(network: nn.Module, params: PyTree, obs: PyTree) -> None:
    """Logs rank_metrics for the feature extractor's output (the penultimate features the heads
    read) and for every layer inside it."""
    lox.log(feature_rank_metrics(network, params, obs))


def feature_rank_metrics(network: nn.Module, params: PyTree, obs: PyTree) -> dict:
    """rank_metrics of the feature extractor's output and of every layer inside it, keyed
    rank/<layer>/<measure>."""

    def inside(module, _):
        return len(module.path) > 0 and module.path[0] == "feature_extractor"

    _, captured = network.apply(
        params, obs, temperature=1.0, capture_intermediates=inside, mutable=["intermediates"]
    )
    logs = {}
    for path, value in jax.tree_util.tree_flatten_with_path(captured["intermediates"])[0]:
        # Paths look like feature_extractor/observation_extractor/Dense_0/__call__/0.
        names = [part.key for part in path if isinstance(part, jax.tree_util.DictKey) and part.key != "__call__"]
        if jnp.ndim(value) < 2 or value.shape[-1] < 8:
            continue
        if names == ["feature_extractor"]:
            layer = "penultimate"
        elif len(names) == 2:
            continue  # the observation extractor's own output, which is the penultimate again
        else:
            layer = "/".join(names[2:])
        flat = jnp.reshape(value, (-1, value.shape[-1]))
        for name, metric in rank_metrics(flat).items():
            logs[f"rank/{layer}/{name}"] = metric
    return logs


@struct.dataclass(frozen=True)
class PPOState:
    step: Array
    params: PyTree
    optimizer_state: optax.OptState


@dataclass
class PPO:
    cfg: PPOConfig
    network: nn.Module
    optimizer: optax.GradientTransformation
    auxiliary_losses: tuple[Callable, ...] = ()
    advantage_estimator: Callable = generalized_advantage_estimator

    def init(self, key: Key, timestep: Timestep) -> tuple[PPOState, PyTree]:
        params = self.network.init(key, timestep.obs, temperature=1.0)
        optimizer_state = self.optimizer.init(params["params"])

        state = PPOState(
            step=jnp.array(0, dtype=canonicalize_dtype(jnp.int64)),
            params=params,
            optimizer_state=optimizer_state,
        )

        return state, None

    def step(
        self,
        state: PPOState,
        carry: PyTree,
        key: Key,
        timestep: Timestep,
        temperature: float = 1.0,
    ) -> tuple[PPOState, PyTree, Array, PyTree]:
        dist, value = self.network.apply(
            state.params, timestep.obs, temperature=temperature
        )
        action, log_prob = dist.sample_and_log_prob(seed=key)
        return (
            state,
            carry,
            action,
            {
                "log_prob": log_prob,
                "value": remove_feature_axis(value),
                "logits": getattr(dist, "logits", None),
            },
        )

    def update(self, state: PPOState, key: Key, transitions: Transition) -> PPOState:
        def loss_fn(params: PyTree, transitions: Transition) -> tuple[Array, tuple]:
            advantages = transitions.aux["advantages"]
            returns = transitions.aux["returns"]

            (dist, value), variables = self.network.apply(
                params,
                transitions.first.obs,
                temperature=1.0,
                mutable=True,
            )

            log_probs = dist.log_prob(transitions.second.action)
            entropy = dist.entropy().mean()
            ratio = jnp.exp(log_probs - transitions.aux["log_prob"])
            approximate_kl = jnp.mean(transitions.aux["log_prob"] - log_probs)
            clip_fraction = jnp.mean(
                (jnp.abs(ratio - 1.0) > self.cfg.clip_coefficient).astype(jnp.float32)
            )
            info = transitions.second.info if isinstance(transitions.second.info, dict) else {}
            actor_loss = -(
                transitions.aux.get("weight", info.get("sample_weight", 1.0))
                * jnp.minimum(
                    ratio * advantages,
                    jnp.clip(
                        ratio,
                        1.0 - self.cfg.clip_coefficient,
                        1.0 + self.cfg.clip_coefficient,
                    )
                    * advantages,
                )
            ).mean()

            value = remove_feature_axis(value)
            critic_loss = 0.5 * ((value - returns) ** 2)
            if self.cfg.clip_value_loss:
                clipped_value = transitions.aux["value"] + jnp.clip(
                    value - transitions.aux["value"],
                    -self.cfg.clip_coefficient,
                    self.cfg.clip_coefficient,
                )
                clipped_critic_loss = 0.5 * ((clipped_value - returns) ** 2)
                critic_loss = jnp.maximum(critic_loss, clipped_critic_loss)
            critic_loss = critic_loss.mean()

            loss = (
                actor_loss
                - self.cfg.entropy_coefficient * entropy
                + self.cfg.value_coefficient * critic_loss
            )

            def apply(params: PyTree) -> PyTree:
                dist, _ = self.network.apply(
                    params, transitions.first.obs, temperature=1.0
                )
                return dist

            for auxiliary_loss in self.auxiliary_losses:
                loss = loss + auxiliary_loss(
                    params=params,
                    apply=apply,
                    transitions=transitions,
                    dist=dist,
                    value=value,
                    variables=variables,
                )
            return loss, (
                variables,
                actor_loss,
                critic_loss,
                entropy,
                approximate_kl,
                clip_fraction,
                value,
            )

        def minibatch_fn(state: PPOState, indices: Array) -> tuple[PPOState, None]:
            minibatch = jax.tree.map(
                lambda leaf: jnp.take(leaf, indices, axis=0), transitions
            )
            returns = minibatch.aux["returns"]
            advantages = minibatch.aux["advantages"]

            (_, aux), grads = jax.value_and_grad(
                loss_fn, has_aux=True, allow_int=True
            )(state.params, minibatch)
            (
                variables,
                actor_loss,
                critic_loss,
                entropy,
                approximate_kl,
                clip_fraction,
                value,
            ) = aux
            explained_variance = 1 - jnp.var(returns - value) / (
                jnp.var(returns) + 1e-8
            )
            logs = {
                "actor/loss": actor_loss,
                "actor/entropy": entropy,
                "actor/approximate_kl": approximate_kl,
                "actor/clip_fraction": clip_fraction,
                "actor/advantage": advantages.mean(),
                "critic/loss": critic_loss,
                "critic/explained_variance": explained_variance,
                "critic/value": value.mean(),
            }
            lox.log(logs)

            updates, optimizer_state = self.optimizer.update(
                grads["params"], state.optimizer_state, state.params["params"]
            )
            variables = {
                name: variables.get(name, value) for name, value in state.params.items()
            }
            params = {
                **variables,
                "params": optax.apply_updates(state.params["params"], updates),
            }

            state = state.replace(
                params=params,
                optimizer_state=optimizer_state,
            )
            return state, None

        def epoch_fn(state: PPOState, key: Key) -> tuple[PPOState, None]:
            permutation = jax.random.permutation(key, batch_size)
            minibatch_indices = permutation.reshape(self.cfg.num_minibatches, -1)
            state, _ = jax.lax.scan(minibatch_fn, state, minibatch_indices)
            return state, None

        num_steps, num_envs, *_ = transitions.second.reward.shape
        batch_size = num_steps * num_envs

        obs = jax.tree.map(
            lambda leaf: jnp.take(leaf, -1, axis=0), transitions.second.obs
        )
        _, value = self.network.apply(state.params, obs, temperature=1.0)
        value = remove_feature_axis(value)
        advantages, returns = self.advantage_estimator(
            transitions,
            transitions.aux["value"],
            value,
            self.cfg.gamma,
            self.cfg.gae_lambda,
        )
        if self.cfg.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        transitions = transitions.replace(
            aux={**transitions.aux, "advantages": advantages, "returns": returns}
        )

        transitions = jax.tree.map(
            lambda x: flatten(x, start_dim=0, end_dim=1), transitions
        )

        # A key folded off the update's key, so logging never shifts the training randomness.
        rank_key = jax.random.fold_in(key, 0x52414E4B)
        keys = jax.random.split(key, self.cfg.update_epochs)
        state, _ = jax.lax.scan(epoch_fn, state, keys)

        if self.cfg.feature_rank:
            sampled = jax.random.choice(
                rank_key, batch_size, (min(self.cfg.feature_rank, batch_size),), replace=False
            )
            log_feature_rank(
                self.network,
                state.params,
                jax.tree.map(lambda leaf: jnp.take(leaf, sampled, axis=0), transitions.first.obs),
            )

        return state
