import math

import flax.linen as nn
import jax.numpy as jnp

from boonta.utils.typing import Array, Carry, Dtype, Key

from ..blocks import Block, SelfAttention, causal_attention_mask

INITIALIZATION_STD = 0.02
LAYER_NORM_EPSILON = 1e-5


class GPT2Layer(Block):
    features: int
    num_heads: int
    num_layers: int
    context_length: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.nowrap
    def attention(self) -> SelfAttention:
        return SelfAttention(
            features=self.features,
            num_heads=self.num_heads,
            attention_mask=causal_attention_mask,
            context_length=self.context_length,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(INITIALIZATION_STD),
            output_kernel_init=self.residual_init(),
            name="attention",
        )

    @nn.nowrap
    def residual_init(self) -> nn.initializers.Initializer:
        return nn.initializers.normal(
            INITIALIZATION_STD / math.sqrt(2 * self.num_layers)
        )

    @nn.compact
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        norm = nn.LayerNorm(
            epsilon=LAYER_NORM_EPSILON,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="attention_norm",
        )
        carry, y = self.attention()(carry, norm(x), done)
        x = x + y
        y = nn.LayerNorm(
            epsilon=LAYER_NORM_EPSILON,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="mlp_norm",
        )(x)
        y = nn.Dense(
            4 * self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(INITIALIZATION_STD),
            name="mlp_hidden",
        )(y)
        y = nn.gelu(y, approximate=True)
        y = nn.Dense(
            self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.residual_init(),
            name="mlp_output",
        )(y)
        return carry, x + y

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return self.attention().initialize_carry(key, input_shape)


class GPT2(Block):
    features: int
    num_layers: int
    num_heads: int
    context_length: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.nowrap
    def layer(self, cls=GPT2Layer, name: str | None = None) -> GPT2Layer:
        return cls(
            features=self.features,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            context_length=self.context_length,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name=name,
        )

    @nn.compact
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        _, sequence_length, *_ = x.shape
        positions = jnp.mod(
            carry[0].position_offset + jnp.arange(sequence_length, dtype=jnp.int32),
            self.context_length,
        )
        x = x + nn.Embed(
            self.context_length,
            self.features,
            embedding_init=nn.initializers.normal(INITIALIZATION_STD),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="position_embedding",
        )(positions)
        layer_cls = nn.remat(GPT2Layer)
        next_carry = []
        for index, layer_carry in enumerate(carry):
            layer_carry, x = self.layer(layer_cls, name=f"layers_{index}")(
                layer_carry, x, done
            )
            next_carry.append(layer_carry)
        x = nn.LayerNorm(
            epsilon=LAYER_NORM_EPSILON,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="output_norm",
        )(x)
        return tuple(next_carry), x

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return tuple(
            self.layer().initialize_carry(key, input_shape)
            for _ in range(self.num_layers)
        )
