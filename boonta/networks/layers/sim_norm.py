import flax.linen as nn

from boonta.utils.typing import Array


class SimNorm(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.dim)
        x = nn.softmax(x, axis=-1)
        return x.reshape(shape)
