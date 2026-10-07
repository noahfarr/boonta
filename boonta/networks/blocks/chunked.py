import flax.linen as nn
import jax.numpy as jnp

from boonta.utils.typing import Array, Carry, Key

from .block import Block


class Chunked(Block):
    block: Block

    @nn.compact
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        batch_size, sequence_length, num_tokens, features = x.shape
        x = x.reshape(batch_size, sequence_length * num_tokens, features)
        done = (
            jnp.zeros((batch_size, sequence_length, num_tokens), done.dtype)
            .at[:, :, 0]
            .set(done)
        )
        carry, x = self.block(
            carry, x, done.reshape(batch_size, sequence_length * num_tokens)
        )
        return (
            carry,
            x.reshape(batch_size, sequence_length, num_tokens, features)[:, :, -1],
        )

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return self.block.initialize_carry(key, input_shape)
