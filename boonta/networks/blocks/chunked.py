import flax.linen as nn
import jax.numpy as jnp
from flax import struct

from boonta.utils.typing import Array, Carry, Key

from .block import Block


@struct.dataclass
class Chunk:
    features: Array
    valid: Array


class Chunked(Block):
    block: Block

    @nn.compact
    def __call__(
        self, carry: Carry, x: Array | Chunk, done: Array
    ) -> tuple[Carry, Array]:
        x, valid = (x.features, x.valid) if isinstance(x, Chunk) else (x, None)
        batch_size, sequence_length, num_tokens, features = x.shape
        x = x.reshape(batch_size, sequence_length * num_tokens, features)
        done = (
            jnp.zeros((batch_size, sequence_length, num_tokens), done.dtype)
            .at[:, :, 0]
            .set(done)
        )
        done = done.reshape(batch_size, sequence_length * num_tokens)
        if valid is None:
            carry, x = self.block(carry, x, done)
            last = jnp.full((batch_size, sequence_length), num_tokens - 1)
        else:
            carry, x = self.block(
                carry,
                x,
                done,
                valid=valid.reshape(batch_size, sequence_length * num_tokens),
            )
            last = num_tokens - 1 - jnp.argmax(valid[..., ::-1], axis=-1)
        x = x.reshape(batch_size, sequence_length, num_tokens, features)
        return carry, jnp.take_along_axis(x, last[..., None, None], axis=2)[:, :, 0]

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return self.block.initialize_carry(key, input_shape)
