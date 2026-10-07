import flax.linen as nn

from boonta.utils.typing import Array


class Parameter(nn.Module):
    value: float = 0.0

    @nn.compact
    def __call__(self) -> Array:
        return self.param("value", nn.initializers.constant(self.value), ())
