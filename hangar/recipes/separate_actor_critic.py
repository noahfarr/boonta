import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

ACTIVATIONS = {"relu": nn.relu, "tanh": nn.tanh}


def mlp(width: int, activation: str, outputs: int, scale: float) -> nn.Sequential:
    layers = []
    for _ in range(2):
        layers += [
            nn.Dense(width, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            ACTIVATIONS[activation],
        ]
    return nn.Sequential(
        [*layers, nn.Dense(outputs, kernel_init=orthogonal(scale), bias_init=constant(0.0))]
    )


class StateIndependentScale(nn.Module):
    mean: nn.Module
    action_dim: int

    @nn.compact
    def __call__(self, x):
        mean = self.mean(x)
        scale = self.param("scale", constant(float(np.log(np.expm1(1.0 - 1e-3)))), (self.action_dim,))
        return jnp.concatenate([mean, jnp.broadcast_to(scale, mean.shape)], axis=-1)
