import flax.linen as nn
import jax.numpy as jnp

from boonta.utils.typing import Array


class ActorCritic(nn.Module):
    actor: nn.Module
    critic: nn.Module

    @nn.compact
    def __call__(self, x: Array, temperature: float | Array) -> tuple:
        distribution = self.actor(x, temperature=temperature)
        return distribution, self.critic(x).astype(jnp.float32)
