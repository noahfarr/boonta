import jax
import jax.numpy as jnp
import lox
import optax
from hydra.utils import instantiate
from optax.contrib._muon import MuonDimensionNumbers

from boonta import datasets, environments
from boonta.algorithms.recurrent_bc import RecurrentBC
from boonta.environments.kaggriculture.wrappers import DeltaMoney, Dispatch
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.networks import Network
from hangar.kaggriculture.network import Heads, Seated, Trunk

from . import schedules


def weighted(transitions, term):
    weight = transitions.aux["weight"].astype(term.dtype)
    return jnp.sum(term * weight) / jnp.maximum(jnp.sum(weight), 1.0)


def reweigh(coefficients):
    def auxiliary_loss(transitions, dist, **kwargs):
        heads = {
            name: -weighted(transitions, term)
            for name, term in dist.parts(transitions.second.action).items()
        }
        lox.log({f"actor/{name}": term for name, term in heads.items()})
        return sum((coefficients.get(name, 1.0) - 1.0) * term for name, term in heads.items())

    return auxiliary_loss


def regress(coefficient):
    def auxiliary_loss(transitions, intermediates, **kwargs):
        value = intermediates["intermediates"]["head"]["value"][..., 0]
        target = transitions.aux["value"].astype(value.dtype)[..., None]
        weight = jnp.broadcast_to(transitions.aux["weight"].astype(value.dtype), value.shape)
        loss = jnp.sum(((value - target) ** 2) * weight) / jnp.maximum(jnp.sum(weight), 1.0)
        lox.log({"critic/loss": loss})
        return coefficient * loss

    return auxiliary_loss


def dimensions(params):
    return jax.tree.map(lambda p: MuonDimensionNumbers(-2, -1) if p.ndim >= 2 else None, params)


def pace(cfg, peak):
    if not cfg.optimizer.get("anneal", False):
        return peak
    steps = int(cfg.total_timesteps) // int(cfg.algorithm.batch_size)
    alpha = float(cfg.optimizer.get("min_lr_ratio", 0.0))
    warmup = int(steps * float(cfg.optimizer.get("warmup", 0.0)))
    return schedules.warmup_cosine(peak, steps, warmup, start=peak * alpha, alpha=alpha)


def tune(cfg):
    clip = optax.clip_by_global_norm(cfg.optimizer.max_grad_norm)
    if cfg.optimizer.get("name") != "muon":
        return optax.chain(clip, optax.adam(pace(cfg, cfg.optimizer.lr)))
    return optax.chain(
        clip,
        optax.contrib.muon(
            pace(cfg, cfg.optimizer.lr),
            beta=cfg.optimizer.get("beta", 0.95),
            adam_learning_rate=pace(cfg, cfg.optimizer.get("adam_lr", cfg.optimizer.lr)),
            weight_decay=cfg.optimizer.get("weight_decay", 0.0),
            muon_weight_dimension_numbers=dimensions,
        ),
    )


def make(cfg):
    env = environments.make(
        namespace=cfg.environment.namespace,
        env_id=cfg.environment.env_id,
        kwargs=dict(cfg.environment.get("kwargs") or {}),
    )
    env = RecordEpisodeStatistics(
        DeltaMoney(Dispatch(env)), gamma=cfg.evaluation.get("gamma", 1.0)
    )

    dtype = cfg.network.get("dtype")
    network = Network(
        feature_extractor=Trunk(
            features=cfg.network.features, blocks=cfg.network.blocks, dtype=dtype
        ),
        torso=Seated(torso=instantiate(cfg.stack)),
        head=Heads(dtype=dtype, market_none_bias=cfg.network.get("market_none_bias", 0.0)),
    )

    auxiliary = cfg.get("auxiliary") or {}
    auxiliary_losses = []
    coefficients = auxiliary.get("head_coefficients")
    if coefficients:
        auxiliary_losses.append(
            reweigh({str(name): float(weight) for name, weight in coefficients.items()})
        )
    value_coefficient = float(auxiliary.get("value_coefficient", 0.0))
    if value_coefficient > 0.0:
        auxiliary_losses.append(regress(value_coefficient))

    return {
        "algorithm": RecurrentBC(
            cfg=instantiate(cfg.algorithm),
            network=network,
            optimizer=tune(cfg),
            auxiliary_losses=tuple(auxiliary_losses),
        ),
        "environment": env,
        "dataset": datasets.make(**cfg.dataset),
    }
