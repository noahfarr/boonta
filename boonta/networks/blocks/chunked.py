import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils.typing import Array, Carry, Key

from .block import Block


@struct.dataclass
class Chunk:
    features: Array
    valid: Array


def _check_budget(count, budget):
    if int(count) > budget:
        raise ValueError(
            f"a sequence holds {int(count)} tokens, more than Chunked's budget of {budget}"
        )


class Chunked(Block):
    block: Block
    budget: int | None = None

    @nn.compact
    def __call__(self, carry: Carry, x: Array | Chunk, done: Array) -> tuple[Carry, Array]:
        if not isinstance(x, Chunk):
            batch_size, sequence_length, num_tokens, features = x.shape
            starts = jnp.zeros((batch_size, sequence_length, num_tokens), done.dtype)
            starts = starts.at[:, :, 0].set(done)
            carry, x = self.block(
                carry,
                x.reshape(batch_size, sequence_length * num_tokens, features),
                starts.reshape(batch_size, sequence_length * num_tokens),
            )
            x = x.reshape(batch_size, sequence_length, num_tokens, features)
            return carry, x[:, :, -1]

        features, valid = x.features, x.valid.astype(bool)
        batch_size, sequence_length, num_tokens, width = features.shape
        starts = done.astype(bool)[..., None] & (jnp.cumsum(valid, axis=-1) == 1)
        length = sequence_length * num_tokens
        budget = min(length, self.budget or length)

        valid = valid.reshape(batch_size, length)
        order = jnp.argsort(~valid, axis=-1, stable=True)[:, :budget]
        count = valid.sum(axis=-1)
        jax.debug.callback(_check_budget, count.max(), budget)

        def pack(leaf):
            leaf = leaf.reshape(batch_size, length, *leaf.shape[3:])
            return jnp.take_along_axis(
                leaf, order.reshape(batch_size, budget, *(1,) * (leaf.ndim - 2)), axis=1
            )

        starts = pack(starts & valid.reshape(starts.shape))
        segments = jnp.where(
            jnp.arange(budget) < count[:, None], 1 + jnp.cumsum(starts, axis=1), 0
        )
        carry, y = self.block(carry, pack(features), starts, segments=segments)
        last = jnp.cumsum(valid.reshape(batch_size, sequence_length, num_tokens).sum(-1), axis=-1) - 1
        return carry, jnp.take_along_axis(y, jnp.clip(last, 0, budget - 1)[..., None], axis=1)

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return self.block.initialize_carry(key, input_shape)
