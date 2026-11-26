import flax.linen as nn
import jax.numpy as jnp
from flax import struct

from boonta.utils import Array


@struct.dataclass(frozen=True)
class MirroredDensePerturbation:
    inputs: Array
    outputs: Array
    bias: Array


class LowRankPerturbedDense(nn.Dense):
    @nn.compact
    def __call__(self, x, noise, sigma):
        y = super().__call__(x)
        if noise is not None:
            grouped = x.reshape(noise.inputs.shape[0], 2, -1, x.shape[-1])
            projected = jnp.einsum("psbi,pir->psbr", grouped, noise.inputs)
            delta = jnp.einsum("psbr,pro->psbo", projected, noise.outputs)
            delta = delta + noise.bias[:, None, None, :]
            delta = delta * jnp.array([1, -1], dtype=x.dtype)[None, :, None, None]
            y = y + sigma * delta.reshape(y.shape)
        return y


class LowRankPerturbedMLP(nn.Module):
    widths: tuple[int, ...]

    @nn.compact
    def __call__(self, observations, noise=None, *, sigma=0.0):
        if not self.widths or any(width < 1 for width in self.widths):
            raise ValueError("Policy widths must be positive")
        x = observations.reshape(observations.shape[0], -1).astype(jnp.float32)
        for index, width in enumerate(self.widths):
            perturbation = None if noise is None else noise[index]
            x = LowRankPerturbedDense(width, name=f"dense_{index}")(
                x, perturbation, sigma
            )
            if index < len(self.widths) - 1:
                x = jnp.tanh(x)
        return x
