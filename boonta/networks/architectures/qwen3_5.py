from functools import partial, reduce

import flax.linen as nn
import jax
import jax.numpy as jnp

from boonta.utils import get_attention_implementation
from boonta.utils.typing import Array, Carry, Dtype, Key

from ..blocks import (GLU, Block, GatedDeltaNet, LinearAttention,
                      QueryKeyNorm, SelfAttentionCarry, broadcast_carry,
                      causal_attention_mask)
from ..blocks.positional_embeddings import rotate


def partial_rotary_embedding(
    query: Array,
    key: Array,
    query_positions: Array,
    key_positions: Array,
    max_wavelength: float,
    rotary_dim: int,
) -> tuple[Array, Array, None]:
    def embed(x: Array, positions: Array) -> Array:
        rotated = rotate(x[..., :rotary_dim], positions, max_wavelength)
        return jnp.concatenate([rotated, x[..., rotary_dim:]], axis=-1)

    return embed(query, query_positions), embed(key, key_positions), None


class GatedAttention(Block):
    features: int
    num_heads: int
    num_groups: int
    head_dim: int
    rotary_dim: int
    max_wavelength: float
    context_length: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self, carry: SelfAttentionCarry | None, x: Array, done: Array
    ) -> tuple[SelfAttentionCarry, Array]:
        batch_size, sequence_length, _ = x.shape
        context_length = self.context_length
        assert (
            sequence_length <= context_length
        ), f"sequence_length must be less than or equal to context_length, but was sequence_length: {sequence_length}, context_length: {context_length}"
        if carry is None:
            carry = self.initialize_carry(None, x.shape)

        projection = partial(
            nn.DenseGeneral,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        query, gate = jnp.split(
            projection(features=(self.num_heads, 2 * self.head_dim), name="query")(x),
            2,
            axis=-1,
        )
        new_key = projection(features=(self.num_groups, self.head_dim), name="key")(x)
        new_value = projection(features=(self.num_groups, self.head_dim), name="value")(x)

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
        values = jnp.concatenate([carry.value, new_value.astype(carry.value.dtype)], axis=1)
        key_position = jnp.concatenate([carry.position, query_position], axis=1)
        key_segment = jnp.concatenate([carry.segment, query_segment], axis=1)
        valid = jnp.concatenate([filled, jnp.ones_like(query_position, bool)], axis=1)
        window = key_position[:, None, :] > query_position[:, :, None] - context_length

        attention_query, attention_key, _ = QueryKeyNorm(
            partial(
                partial_rotary_embedding,
                max_wavelength=self.max_wavelength,
                rotary_dim=self.rotary_dim,
            ),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="positional_embedding",
        )(query, keys, query_position, key_position)

        mask = (valid[:, None, :] & window)[:, None] & causal_attention_mask(
            query_position, key_position, query_segment, key_segment, valid
        )
        mask = jnp.broadcast_to(
            mask,
            (batch_size, self.num_heads, sequence_length, context_length + sequence_length),
        )

        implementation, attention_dtype = get_attention_implementation(
            self.head_dim, sequence_length, context_length + sequence_length
        )
        attention = jax.nn.dot_product_attention(
            attention_query.astype(attention_dtype),
            attention_key.astype(attention_dtype),
            values.astype(attention_dtype),
            mask=mask,
            implementation=implementation,
        ).astype(query.dtype)
        attention = attention * nn.sigmoid(gate)

        y = nn.DenseGeneral(
            self.features,
            axis=(-2, -1),
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="output_projection",
        )(attention)

        carry = SelfAttentionCarry(
            key=carry.key.at[batch_index, slots].set(new_key.astype(carry.key.dtype)),
            value=carry.value.at[batch_index, slots].set(new_value.astype(carry.value.dtype)),
            position=carry.position.at[batch_index, slots].set(query_position),
            segment=carry.segment.at[batch_index, slots].set(query_segment),
            position_offset=carry.position_offset + sequence_length,
            segment_offset=carry.segment_offset + increments[:, -1:],
        )
        return carry, y

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> SelfAttentionCarry:
        batch_size, *_ = input_shape
        cache = (batch_size, self.context_length, self.num_groups, self.head_dim)
        return SelfAttentionCarry(
            key=jnp.zeros(cache, dtype=self.dtype),
            value=jnp.zeros(cache, dtype=self.dtype),
            position=jnp.zeros((batch_size, self.context_length), dtype=jnp.int32),
            segment=jnp.zeros((batch_size, self.context_length), dtype=jnp.int32),
            position_offset=jnp.zeros((batch_size, 1), dtype=jnp.int32),
            segment_offset=jnp.zeros((batch_size, 1), dtype=jnp.int32),
        )


class Qwen3_5Layer(Block):
    features: int
    layer_type: str
    num_heads: int
    num_groups: int
    head_dim: int
    rotary_dim: int
    max_wavelength: float
    context_length: int
    linear_num_heads: int
    linear_num_value_heads: int
    linear_head_dim: int
    linear_value_head_dim: int
    kernel_size: int = 4
    chunk_size: int = 64
    hidden_dim: int | None = None
    epsilon: float = 1e-6
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.nowrap
    def mixer(self) -> Block:
        if self.layer_type == "full_attention":
            return GatedAttention(
                features=self.features,
                num_heads=self.num_heads,
                num_groups=self.num_groups,
                head_dim=self.head_dim,
                rotary_dim=self.rotary_dim,
                max_wavelength=self.max_wavelength,
                context_length=self.context_length,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name="attention",
            )
        if self.layer_type == "linear_attention":
            return LinearAttention(
                cell=GatedDeltaNet(
                    features=self.features,
                    num_heads=self.linear_num_heads,
                    num_value_heads=self.linear_num_value_heads,
                    head_dim=self.linear_head_dim,
                    value_head_dim=self.linear_value_head_dim,
                    kernel_size=self.kernel_size,
                    epsilon=self.epsilon,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    parent=None,
                ),
                chunk_size=self.chunk_size,
                name="linear_attention",
            )
        raise ValueError(f"unknown layer type {self.layer_type!r}")

    @nn.compact
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        rms_norm = partial(
            nn.RMSNorm, epsilon=self.epsilon, dtype=self.dtype, param_dtype=self.param_dtype
        )
        carry, y = self.mixer()(carry, rms_norm(name="attention_norm")(x), done)
        x = x + y
        _, y = GLU(
            self.features,
            hidden_dim=self.hidden_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="swi_glu",
        )(None, rms_norm(name="swi_glu_norm")(x), done)
        return carry, x + y

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return self.mixer().initialize_carry(key, input_shape)


class Qwen3_5(Block):
    features: int
    layer_types: tuple[str, ...]
    num_heads: int
    num_groups: int
    head_dim: int
    rotary_dim: int
    max_wavelength: float
    context_length: int
    linear_num_heads: int
    linear_num_value_heads: int
    linear_head_dim: int
    linear_value_head_dim: int
    kernel_size: int = 4
    chunk_size: int = 64
    hidden_dim: int | None = None
    epsilon: float = 1e-6
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @property
    def num_layers(self) -> int:
        return len(self.layer_types)

    @nn.nowrap
    def layer(self, index: int, cls=Qwen3_5Layer, name: str | None = None) -> Qwen3_5Layer:
        return cls(
            features=self.features,
            layer_type=self.layer_types[index],
            num_heads=self.num_heads,
            num_groups=self.num_groups,
            head_dim=self.head_dim,
            rotary_dim=self.rotary_dim,
            max_wavelength=self.max_wavelength,
            context_length=self.context_length,
            linear_num_heads=self.linear_num_heads,
            linear_num_value_heads=self.linear_num_value_heads,
            linear_head_dim=self.linear_head_dim,
            linear_value_head_dim=self.linear_value_head_dim,
            kernel_size=self.kernel_size,
            chunk_size=self.chunk_size,
            hidden_dim=self.hidden_dim,
            epsilon=self.epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name=name,
        )

    @nn.compact
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        layer_cls = nn.remat(Qwen3_5Layer)

        def step(state: tuple[Carry, Array], inputs) -> tuple[Carry, Array]:
            carry, x = state
            index, carry_i = inputs
            carry_i, x = self.layer(index, layer_cls, name=f"layers_{index}")(
                carry_i, x, done
            )
            return (*carry, carry_i), x

        carries = broadcast_carry(carry, self.num_layers)
        carry, x = reduce(
            step, zip(range(self.num_layers), carries, strict=True), ((), x)
        )
        x = nn.RMSNorm(
            epsilon=self.epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="output_norm",
        )(x)
        return carry, x

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return tuple(
            self.layer(index).initialize_carry(key, input_shape)
            for index in range(self.num_layers)
        )
