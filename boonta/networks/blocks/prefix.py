import flax.linen as nn
import jax
import jax.numpy as jnp

from boonta.utils.typing import Array, Carry, Key

from .block import Block


class Prefix(Block):
    block: Block
    encoder: nn.Module
    tokens: tuple[int, ...]

    @nn.compact
    def __call__(self, carry: Carry, x: Array, done: Array, **kwargs) -> tuple[Carry, Array]:
        length = len(self.tokens)
        features = self.encoder(jnp.asarray(self.tokens, jnp.int32)[None])
        prefix, _ = self.block(
            jax.tree.map(lambda leaf: jnp.zeros_like(leaf[:1]), carry),
            features,
            jnp.zeros((1, length), bool),
        )
        slots = length + length % 2
        prefix = jax.tree.map(lambda leaf: leaf[:, :slots], prefix)
        return self.block(carry, x, done, prefix=prefix, **kwargs)

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return self.block.initialize_carry(key, input_shape)
