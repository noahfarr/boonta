import flax.linen as nn

from boonta.utils.typing import Array


class Identity(nn.Module):
    def __call__(self, x: Array, *args, **kwargs) -> Array:
        return x
