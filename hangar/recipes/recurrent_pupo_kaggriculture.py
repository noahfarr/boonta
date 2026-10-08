import jax
import jax.numpy as jnp
import lox
import optax
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.recurrent_pupo import RecurrentPuPO
from boonta.environments.kaggriculture.wrappers import DeltaMoney, Dispatch
from boonta.environments.wrappers import RecordEpisodeStatistics, TransformReward
from boonta.networks import Network, Pretrained
from hangar.kaggriculture.distribution import SEATS
from hangar.kaggriculture.network import Heads, Seated, Split, Trunk, Value, clone

from . import schedules

FLOW_SCALE = 8.0


def market_flow(coefficient):
    def auxiliary_loss(transitions, dist, **kwargs):
        before = transitions.first.obs["market"].astype(jnp.float32)
        after = transitions.second.obs["market"].astype(jnp.float32)
        target = jnp.repeat(((after - before) / FLOW_SCALE)[:, None], SEATS, axis=1)
        flow = dist.logits["flow"]
        loss = jnp.mean((flow - target.reshape(-1, target.shape[-1])) ** 2)
        lox.log({"auxiliary/market_flow": loss})
        return coefficient * loss

    return auxiliary_loss


def behaviour_anchor(weights, coefficient):
    def auxiliary_loss(transitions, dist, apply, params, **kwargs):
        frozen = jax.lax.stop_gradient(
            {**params, "params": {**params["params"], "actor": {"pretrained": weights}}}
        )
        action = transitions.second.action
        log_ratio = jnp.clip(apply(frozen).log_prob(action) - dist.log_prob(action), -20.0, 20.0)
        divergence = jnp.mean(jnp.exp(log_ratio) - 1.0 - log_ratio)
        lox.log({"auxiliary/behaviour_kl": divergence})
        return coefficient * divergence

    return auxiliary_loss


def side(tree):
    def label(path, _):
        head = [getattr(part, "key", None) for part in path[:2]]
        return "actor" if "actor" in head else "critic"

    return jax.tree_util.tree_map_with_path(label, tree)


def make(cfg):
    scale = float(cfg.environment.get("reward_scale", 1.0))
    env = environments.make(
        namespace=cfg.environment.namespace,
        env_id=cfg.environment.env_id,
        kwargs=dict(cfg.environment.get("kwargs") or {}),
    )
    env = RecordEpisodeStatistics(DeltaMoney(Dispatch(env)), gamma=cfg.algorithm.gamma)
    env = TransformReward(env, lambda reward: reward / scale)

    dtype = cfg.network.get("dtype")
    actor = Network(
        feature_extractor=Trunk(
            features=cfg.network.features, blocks=cfg.network.blocks, dtype=dtype
        ),
        torso=Seated(torso=instantiate(cfg.stack)),
        head=Heads(dtype=dtype, market_none_bias=cfg.network.get("market_none_bias", 0.0)),
    )
    auxiliary_losses = [market_flow(cfg.network.get("auxiliary_coefficient", 0.5))]

    pretrained = cfg.network.get("pretrained")
    if pretrained:
        weights = clone(pretrained)
        actor = Pretrained(module=actor, weights=weights)
        anchor = float(cfg.network.get("anchor_coefficient", 0.0))
        if anchor:
            auxiliary_losses.append(behaviour_anchor(weights, anchor))

    network = Split(
        actor=actor,
        critic=Network(
            feature_extractor=Trunk(
                features=cfg.network.features, blocks=cfg.network.blocks, dtype=dtype
            ),
            torso=Seated(torso=instantiate(cfg.stack)),
            head=Value(dtype=dtype),
        ),
    )

    batch = cfg.environment.num_envs * cfg.rollout.num_steps
    updates = max(int(cfg.total_timesteps) // batch, 1)
    steps = updates * cfg.algorithm.update_epochs * cfg.algorithm.num_minibatches
    warmup = int(cfg.optimizer.get("warmup_steps", 0))
    learning_rate = schedules.warmup_cosine(
        cfg.optimizer.lr, steps, warmup, alpha=cfg.optimizer.get("min_lr_ratio", 0.0)
    )

    settle = int(cfg.optimizer.get("critic_warmup_steps", 0))
    release = max(settle // 4, 1)
    actor_rate = (
        learning_rate
        if settle <= 0
        else optax.join_schedules(
            [
                optax.constant_schedule(0.0),
                optax.linear_schedule(0.0, cfg.optimizer.lr, release),
                learning_rate,
            ],
            [settle, settle + release],
        )
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
        optax.multi_transform(
            {"actor": optax.adam(actor_rate), "critic": optax.adam(learning_rate)}, side
        ),
    )
    accumulate = int(cfg.optimizer.get("accumulate", 1))
    if accumulate > 1:
        optimizer = optax.MultiSteps(optimizer, every_k_schedule=accumulate)

    return {
        "algorithm": RecurrentPuPO(
            cfg=instantiate(cfg.algorithm),
            importance_exponent=instantiate(cfg.importance_exponent),
            network=network,
            optimizer=optimizer,
            auxiliary_losses=tuple(auxiliary_losses),
        ),
        "environment": env,
    }
