from functools import partial, reduce

import flax.linen as nn
import jax.numpy as jnp

from boonta.utils.typing import Array, Carry, Dtype, Key

from ..blocks import (GLU, Block, GatedDeltaNet, LinearAttention, OutputGate,
                      QueryKeyNorm, SelfAttention, broadcast_carry,
                      causal_attention_mask, partial_rotary_embedding)


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
            return SelfAttention(
                features=self.features,
                num_heads=self.num_heads,
                attention_mask=causal_attention_mask,
                num_groups=self.num_groups,
                head_dim=self.head_dim,
                use_bias=False,
                context_length=self.context_length,
                positional_embedding=QueryKeyNorm(
                    partial(
                        partial_rotary_embedding,
                        max_wavelength=self.max_wavelength,
                        rotary_dim=self.rotary_dim,
                    ),
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    parent=None,
                ),
                output_gate=OutputGate(
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    parent=None,
                ),
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
