import flax.linen as nn
import jax.numpy as jnp

from boonta.utils.typing import Array


class GoalConditioned(nn.Module):
    encoder: nn.Module
    decoder: nn.Module
    goal_size: int

    def __call__(self, x: Array) -> Array:
        latent = jnp.tanh(self.encoder(x[..., -self.goal_size :]))
        return self.decoder(jnp.concatenate([latent, x[..., : -self.goal_size]], axis=-1))
