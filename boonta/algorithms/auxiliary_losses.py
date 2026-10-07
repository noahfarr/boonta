from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import lox

from boonta.utils import PyTree, Transition
from boonta.utils.typing import Array


@dataclass
class DR3:
    coefficient: float

    def __call__(
        self, intermediates: PyTree, transitions: Transition, **kwargs
    ) -> Array:
        features = intermediates["intermediates"]["features"].astype(jnp.float32)
        assert features.ndim == 3, (
            f"DR3 pairs each state's features with the next state's, so it needs "
            f"them as (batch, time, width) trajectories; got shape {features.shape}. "
            f"An algorithm that shuffles single transitions into minibatches has "
            f"no next state to pair with."
        )
        done = transitions.second.terminated | transitions.second.truncated
        done = done[:, :-1].astype(jnp.float32)
        similarity = jnp.sum(features[:, :-1] * features[:, 1:], axis=-1)
        dr3 = jnp.sum(similarity * (1.0 - done)) / jnp.maximum(
            jnp.sum(1.0 - done), 1.0
        )
        lox.log({"auxiliary_loss/dr3": dr3})
        return self.coefficient * dr3


@dataclass
class Anchor:
    params: PyTree
    coefficient: float

    def __call__(self, dist: PyTree, apply: Callable, **kwargs) -> Array:
        log_p = dist.logits
        log_q = apply(self.params).logits
        divergence = (jnp.exp(log_p) * (log_p - log_q)).sum(-1).mean()
        lox.log({"auxiliary_loss/anchor_divergence": divergence})
        return self.coefficient * divergence
