import jax.numpy as jnp
import lox
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.ppo import PPO
from boonta.environments.kaggriculture.wrappers import DeltaMoney, Dispatch
from boonta.environments.wrappers import RecordEpisodeStatistics, TransformReward
from boonta.networks import Pretrained
from hangar.kaggriculture.distribution import SEATS
from hangar.kaggriculture.network import Farmer, clone

from . import schedules

FLOW_SCALE = 8.0


def market_flow(coefficient):
    def auxiliary_loss(transitions, dist, **kwargs):
        before = transitions.first.obs["market"].astype(jnp.float32)
        after = transitions.second.obs["market"].astype(jnp.float32)
        target = jnp.repeat((after - before) / FLOW_SCALE, SEATS, axis=0)
        loss = jnp.mean((dist.logits["flow"] - target) ** 2)
        lox.log({"auxiliary/market_flow": loss})
        return coefficient * loss

    return auxiliary_loss


def behaviour_anchor(network, reference, coefficient):
    def auxiliary_loss(transitions, dist, **kwargs):
        prior, _ = network.apply(reference, transitions.first.obs, temperature=1.0)
        action = transitions.second.action
        log_ratio = jnp.clip(prior.log_prob(action) - dist.log_prob(action), -20.0, 20.0)
        divergence = jnp.mean(jnp.exp(log_ratio) - 1.0 - log_ratio)
        lox.log({"auxiliary/behaviour_kl": divergence})
        return coefficient * divergence

    return auxiliary_loss


def make(cfg):
    scale = float(cfg.environment.get("reward_scale", 1.0))
    env = environments.make(
        namespace=cfg.environment.namespace,
        env_id=cfg.environment.env_id,
        kwargs=dict(cfg.environment.get("kwargs") or {}),
    )
    env = RecordEpisodeStatistics(DeltaMoney(Dispatch(env)), gamma=cfg.algorithm.gamma)
    env = TransformReward(env, lambda reward: reward / scale)

    network = Farmer(
        features=cfg.network.features,
        blocks=cfg.network.blocks,
        market_none_bias=cfg.network.get("market_none_bias", 0.0),
    )
    auxiliary_losses = [market_flow(cfg.network.get("auxiliary_coefficient", 0.5))]

    pretrained = cfg.network.get("pretrained")
    if pretrained:
        weights = clone(pretrained)
        network = Pretrained(module=network, weights=weights)
        anchor = float(cfg.network.get("anchor_coefficient", 0.0))
        if anchor:
            auxiliary_losses.append(
                behaviour_anchor(network, {"params": {"pretrained": weights}}, anchor)
            )

    learning_rate = schedules.learning_rate(
        cfg, batch_size=cfg.environment.num_envs * cfg.rollout.num_steps
    )

    return {
        "algorithm": PPO(
            cfg=instantiate(cfg.algorithm),
            network=network,
            optimizer=optax.chain(
                optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
                optax.adam(learning_rate),
            ),
            auxiliary_losses=tuple(auxiliary_losses),
        ),
        "environment": env,
    }
