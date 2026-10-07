import flax.linen as nn

from boonta.utils.typing import Array


class Unembed(nn.Module):
    embed: nn.Embed

    @nn.compact
    def __call__(self, x: Array) -> Array:
        return self.embed.attend(x)
