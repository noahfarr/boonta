from math import prod

import flax.linen as nn

from boonta.utils.typing import Array


class Flatten(nn.Module):
    start_dim: int = 1
    end_dim: int = -1

    @nn.compact
    def __call__(self, x: Array) -> Array:
        shape = x.shape
        ndim = len(shape)

        start = self.start_dim if self.start_dim >= 0 else ndim + self.start_dim
        end = self.end_dim if self.end_dim >= 0 else ndim + self.end_dim

        new_shape = shape[:start] + (prod(shape[start : end + 1]),) + shape[end + 1 :]
        return x.reshape(new_shape)
