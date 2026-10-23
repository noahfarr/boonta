import flax.linen as nn
import jax
import jax.numpy as jnp

from boonta.utils.typing import Array


class Normalize(nn.Module):
    epsilon: float = 1e-2

    @nn.compact
    def __call__(self, x: Array) -> Array:
        *_, width = jnp.shape(x)
        mean = self.variable("statistics", "mean", jnp.zeros, (width,))
        variance = self.variable("statistics", "variance", jnp.ones, (width,))
        count = self.variable("statistics", "count", jnp.zeros, ())
        if self.is_mutable_collection("statistics") and not self.is_initializing():
            batch = jax.lax.stop_gradient(jnp.reshape(x, (-1, width)))
            size, _ = jnp.shape(batch)
            total = count.value + size
            delta = jnp.mean(batch, axis=0) - mean.value
            variance.value = (
                variance.value * count.value
                + jnp.var(batch, axis=0) * size
                + delta**2 * count.value * size / total
            ) / total
            mean.value = mean.value + delta * size / total
            count.value = total
        return (x - mean.value) / jnp.sqrt(variance.value + self.epsilon**2)
