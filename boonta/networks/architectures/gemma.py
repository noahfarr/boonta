from functools import partial, reduce

import flax.linen as nn
import jax.numpy as jnp

from boonta.utils.typing import Array, Carry, Dtype, Key

from ..blocks import (GLU, Block, SelfAttention, broadcast_carry,
                      causal_attention_mask, rotary_positional_embedding)


class GemmaLayer(Block):
    features: int
    num_heads: int
    num_groups: int
    head_dim: int
    max_wavelength: float
    context_length: int
    expansion_factor: int = 4
    hidden_dim: int | None = None
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.nowrap
    def attention(self) -> SelfAttention:
        return SelfAttention(
            features=self.features,
            num_heads=self.num_heads,
            attention_mask=causal_attention_mask,
            num_groups=self.num_groups,
            head_dim=self.head_dim,
            use_bias=False,
            context_length=self.context_length,
            positional_embedding=partial(
                rotary_positional_embedding, max_wavelength=self.max_wavelength
            ),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="attention",
        )

    @nn.compact
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        rms_norm = partial(nn.RMSNorm, dtype=self.dtype, param_dtype=self.param_dtype)
        carry, y = self.attention()(carry, rms_norm(name="attention_norm")(x), done)
        x = x + y
        _, y = GLU(
            self.features,
            self.expansion_factor,
            hidden_dim=self.hidden_dim,
            activation=partial(nn.gelu, approximate=True),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="ge_glu",
        )(None, rms_norm(name="ge_glu_norm")(x), done)
        return carry, x + y

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return self.attention().initialize_carry(key, input_shape)


class Gemma(Block):
    features: int
    num_layers: int
    num_heads: int
    num_groups: int
    head_dim: int
    max_wavelength: float
    context_length: int
    expansion_factor: int = 4
    hidden_dim: int | None = None
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.nowrap
    def layer(self, name: str | None = None) -> GemmaLayer:
        return GemmaLayer(
            features=self.features,
            num_heads=self.num_heads,
            num_groups=self.num_groups,
            head_dim=self.head_dim,
            expansion_factor=self.expansion_factor,
            hidden_dim=self.hidden_dim,
            max_wavelength=self.max_wavelength,
            context_length=self.context_length,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name=name,
        )

    @nn.compact
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        def step(state: tuple[Carry, Array], inputs) -> tuple[Carry, Array]:
            carry, x = state
            index, carry_i = inputs
            carry_i, x = self.layer(name=f"layers_{index}")(carry_i, x, done)
            return (*carry, carry_i), x

        carries = broadcast_carry(carry, self.num_layers)
        carry, x = reduce(
            step, zip(range(self.num_layers), carries, strict=True), ((), x)
        )
        x = nn.RMSNorm(
            dtype=self.dtype, param_dtype=self.param_dtype, name="output_norm"
        )(x)
        return carry, x

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return tuple(
            self.layer().initialize_carry(key, input_shape)
            for _ in range(self.num_layers)
        )
