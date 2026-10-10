from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from boonta.utils import get_attention_implementation
from boonta.utils.typing import Array, Dtype, Key

from .block import Block


@struct.dataclass
class SelfAttentionCarry:
    key: Array
    value: Array
    position: Array
    segment: Array
    position_offset: Array
    segment_offset: Array


def causal_attention_mask(
    query_position: Array,
    key_position: Array,
    query_segment: Array,
    key_segment: Array,
    valid: Array,
) -> Array:
    causal = query_position[:, :, None] >= key_position[:, None, :]
    same_segment = query_segment[:, :, None] == key_segment[:, None, :]
    return (valid[:, None, :] & causal & same_segment)[:, None]


def joint_attention_mask(input_mask: Array, segment_starts: Array) -> Array:
    segment_starts = jnp.broadcast_to(segment_starts, input_mask.shape)
    cumsum = jnp.cumsum(segment_starts, axis=-1)
    attention_mask = cumsum[..., None, :] <= cumsum[..., :, None]
    valid_mask = input_mask[..., None, :] & input_mask[..., :, None]
    return attention_mask & valid_mask


class OutputGate(nn.Module):
    num_heads: int
    head_dim: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, attention: Array, x: Array) -> Array:
        gate = nn.DenseGeneral(
            (self.num_heads, self.head_dim),
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="projection",
        )(x)
        return attention * nn.sigmoid(gate)


class SelfAttention(Block):
    features: int
    num_heads: int
    attention_mask: Callable[..., Array | None] | None = None
    context_length: int = 0
    num_groups: int | None = None
    head_dim: int | None = None
    use_bias: bool = True
    positional_embedding: Callable[
        [Array, Array, Array, Array], tuple[Array, Array, Array | None]
    ] = lambda query, key, query_positions, key_positions: (query, key, None)
    output_gate: Callable[[Array, Array], Array] = lambda attention, x: attention
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(
        self, carry: SelfAttentionCarry | None, x: Array, done: Array
    ) -> tuple[SelfAttentionCarry, Array]:
        batch_size, sequence_length, *_ = x.shape
        context_length = self.context_length or sequence_length
        assert (
            sequence_length <= context_length
        ), f"sequence_length must be less than or equal to context_length, but was sequence_length: {sequence_length}, context_length: {context_length}"
        assert (
            self.head_dim is not None or self.features % self.num_heads == 0
        ), f"features must be divisible by num_heads, but was features: {self.features}, num_heads: {self.num_heads}"

        num_groups = self.num_groups or self.num_heads
        assert (
            self.num_heads % num_groups == 0
        ), f"num_heads must be divisible by num_groups, but was num_heads: {self.num_heads}, num_groups: {num_groups}"

        head_dim = self.head_dim or self.features // self.num_heads

        if carry is None:
            carry = self.initialize_carry(None, x.shape)

        projection = partial(
            nn.DenseGeneral,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
        )

        query = projection(features=(self.num_heads, head_dim), name="query")(x)
        new_key = projection(features=(num_groups, head_dim), name="key")(x)
        new_value = projection(features=(num_groups, head_dim), name="value")(x)

        positions = carry.position_offset + jnp.arange(sequence_length, dtype=jnp.int32)
        slots = jnp.mod(positions, context_length)
        batch_index = jnp.arange(batch_size)[:, None]
        increments = jnp.cumsum(done.astype(jnp.int32), axis=1)
        query_position = positions
        query_segment = carry.segment_offset + increments

        filled = (jnp.arange(context_length) < carry.position_offset) | (
            carry.position_offset >= context_length
        )
        keys = jnp.concatenate([carry.key, new_key.astype(carry.key.dtype)], axis=1)
        values = jnp.concatenate(
            [carry.value, new_value.astype(carry.value.dtype)], axis=1
        )
        key_position = jnp.concatenate([carry.position, query_position], axis=1)
        key_segment = jnp.concatenate([carry.segment, query_segment], axis=1)
        valid = jnp.concatenate([filled, jnp.ones_like(query_position, bool)], axis=1)
        window = key_position[:, None, :] > query_position[:, :, None] - context_length

        attention_query, attention_key, bias = self.positional_embedding(
            query, keys, query_position, key_position
        )

        mask = (valid[:, None, :] & window)[:, None]
        if self.attention_mask is not None:
            mask = mask & self.attention_mask(
                query_position, key_position, query_segment, key_segment, valid
            )
        mask = jnp.broadcast_to(
            mask,
            (batch_size, self.num_heads, sequence_length, context_length + sequence_length),
        )

        key = carry.key.at[batch_index, slots].set(new_key.astype(carry.key.dtype))
        value = carry.value.at[batch_index, slots].set(
            new_value.astype(carry.value.dtype)
        )
        position = carry.position.at[batch_index, slots].set(query_position)
        segment = carry.segment.at[batch_index, slots].set(query_segment)
        position_offset = carry.position_offset + sequence_length
        segment_offset = carry.segment_offset + increments[:, -1:]

        implementation, attention_dtype = get_attention_implementation(
            attention_query.shape[-1], sequence_length, context_length + sequence_length
        )
        attention = jax.nn.dot_product_attention(
            attention_query.astype(attention_dtype),
            attention_key.astype(attention_dtype),
            values.astype(attention_dtype),
            bias=bias.astype(attention_dtype) if bias is not None else None,
            mask=mask,
            implementation=implementation,
        ).astype(query.dtype)
        attention = self.output_gate(attention, x)

        y = nn.DenseGeneral(
            self.features,
            axis=(-2, -1),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            name="output_projection",
        )(attention)

        carry = SelfAttentionCarry(
            key=key,
            value=value,
            position=position,
            segment=segment,
            position_offset=position_offset,
            segment_offset=segment_offset,
        )
        return carry, y

    @nn.nowrap
    def initialize_carry(
        self, key: Key, input_shape: tuple[int, ...]
    ) -> SelfAttentionCarry:
        batch_size, sequence_length, *_ = input_shape
        context_length = self.context_length or sequence_length
        num_groups = self.num_groups or self.num_heads
        head_dim = self.head_dim or self.features // self.num_heads
        return SelfAttentionCarry(
            key=jnp.zeros(
                (batch_size, context_length, num_groups, head_dim), dtype=self.dtype
            ),
            value=jnp.zeros(
                (batch_size, context_length, num_groups, head_dim), dtype=self.dtype
            ),
            position=jnp.zeros((batch_size, context_length), dtype=jnp.int32),
            segment=jnp.zeros((batch_size, context_length), dtype=jnp.int32),
            position_offset=jnp.zeros((batch_size, 1), dtype=jnp.int32),
            segment_offset=jnp.zeros((batch_size, 1), dtype=jnp.int32),
        )
