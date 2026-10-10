from collections.abc import Callable
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils.typing import Array, Dtype, Key

from .block import Block
from .linear_attention import (LinearAttentionCarry, LinearAttentionCellBase,
                               LinearAttentionInputs)


@struct.dataclass
class GatedDeltaNetCarry:
    query: Array
    key: Array
    value: Array


def l2_normalize(x: Array, epsilon: float = 1e-6) -> Array:
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + epsilon)


def inverse_softplus_init(minimum: float = 1e-3, maximum: float = 1e-1, floor: float = 1e-4):
    def init(key: Key, shape: tuple[int, ...], dtype: Dtype = jnp.float32) -> Array:
        log_step = jax.random.uniform(
            key, shape, jnp.float32, jnp.log(minimum), jnp.log(maximum)
        )
        step = jnp.maximum(jnp.exp(log_step), floor)
        return (step + jnp.log(-jnp.expm1(-step))).astype(dtype)

    return init


def log_uniform_init(minimum: float = 0.0, maximum: float = 16.0):
    def init(key: Key, shape: tuple[int, ...], dtype: Dtype = jnp.float32) -> Array:
        rate = jax.random.uniform(key, shape, jnp.float32, minimum, maximum)
        return jnp.log(rate).astype(dtype)

    return init


class ShortConvolution(Block):
    features: int
    kernel_size: int = 4
    activation: Callable[[Array], Array] = nn.silu
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: nn.initializers.Initializer = nn.initializers.variance_scaling(
        1 / 3, "fan_in", "uniform", in_axis=0, out_axis=1
    )

    @nn.compact
    def __call__(self, carry: Array, x: Array, done: Array) -> tuple[Array, Array]:
        batch_size, sequence_length, _ = x.shape
        kernel = self.param(
            "kernel", self.kernel_init, (self.kernel_size, self.features), self.param_dtype
        )
        history = self.kernel_size - 1
        inputs = jnp.concatenate([carry, x.astype(carry.dtype)], axis=1)
        segments = jnp.concatenate(
            [
                jnp.zeros((batch_size, history), jnp.int32),
                jnp.cumsum(done.astype(jnp.int32), axis=1),
            ],
            axis=1,
        )
        current = segments[:, history:, None]
        output = sum(
            jnp.where(
                segments[:, offset : offset + sequence_length, None] == current,
                inputs[:, offset : offset + sequence_length].astype(jnp.float32),
                0.0,
            )
            * kernel[offset].astype(jnp.float32)
            for offset in range(self.kernel_size)
        )
        window = jnp.where(
            segments[:, sequence_length:, None] == segments[:, -1:, None],
            inputs[:, sequence_length:],
            0,
        ).astype(carry.dtype)
        return window, self.activation(output).astype(self.dtype or x.dtype)

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Array:
        batch_size, *_ = input_shape
        return jnp.zeros(
            (batch_size, self.kernel_size - 1, self.features), dtype=self.dtype
        )


class GatedDeltaNet(LinearAttentionCellBase):
    features: int
    num_key_heads: int = 6
    num_value_heads: int = 6
    key_dim: int = 256
    value_dim: int = 512
    kernel_size: int = 4
    epsilon: float = 1e-5
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()

    @nn.nowrap
    def widths(self) -> tuple[int, int]:
        return self.num_key_heads * self.key_dim, self.num_value_heads * self.value_dim

    def setup(self):
        assert (
            self.num_value_heads % self.num_key_heads == 0
        ), f"num_value_heads must be divisible by num_key_heads, but was num_value_heads: {self.num_value_heads}, num_key_heads: {self.num_key_heads}"
        key_width, value_width = self.widths()
        dense = partial(
            nn.Dense,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
        )
        convolution = partial(
            ShortConvolution,
            kernel_size=self.kernel_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        self.query = dense(key_width)
        self.key = dense(key_width)
        self.value = dense(value_width)
        self.query_convolution = convolution(key_width)
        self.key_convolution = convolution(key_width)
        self.value_convolution = convolution(value_width)
        self.alpha = dense(self.num_value_heads)
        self.beta = dense(self.num_value_heads)
        self.log_rate = self.param(
            "log_rate", log_uniform_init(), (self.num_value_heads,), self.param_dtype
        )
        self.step_bias = self.param(
            "step_bias", inverse_softplus_init(), (self.num_value_heads,), self.param_dtype
        )
        self.gate = dense(value_width)
        self.norm = nn.RMSNorm(
            epsilon=self.epsilon, dtype=self.dtype, param_dtype=self.param_dtype
        )
        self.output_projection = dense(self.features)

    def __call__(
        self, carry: GatedDeltaNetCarry, x: Array, done: Array
    ) -> tuple[GatedDeltaNetCarry, LinearAttentionInputs]:
        batch_size, sequence_length, _ = x.shape
        query_window, query = self.query_convolution(carry.query, self.query(x), done)
        key_window, key = self.key_convolution(carry.key, self.key(x), done)
        value_window, value = self.value_convolution(carry.value, self.value(x), done)

        alpha = self.alpha(x).astype(jnp.float32)
        beta = self.beta(x).astype(jnp.float32)
        log_decay = -jnp.exp(self.log_rate.astype(jnp.float32)) * nn.softplus(
            alpha + self.step_bias.astype(jnp.float32)
        )

        def heads(x: Array) -> Array:
            x = x.reshape(batch_size, sequence_length, self.num_key_heads, self.key_dim)
            return jnp.repeat(
                l2_normalize(x.astype(jnp.float32)),
                self.num_value_heads // self.num_key_heads,
                axis=2,
            )

        beta = nn.sigmoid(beta)
        inputs = LinearAttentionInputs(
            query=heads(query) * self.key_dim**-0.5,
            key=heads(key),
            value=value.reshape(batch_size, sequence_length, self.num_value_heads, -1),
            log_decay=log_decay,
            erase=beta,
            write=beta,
        )
        return GatedDeltaNetCarry(query_window, key_window, value_window), inputs

    def output(self, outputs: Array, x: Array) -> Array:
        batch_size, sequence_length, _ = x.shape
        _, value_width = self.widths()
        gate = self.gate(x)
        outputs = self.norm(outputs).astype(jnp.float32) * nn.silu(
            gate.reshape(outputs.shape).astype(jnp.float32)
        )
        return self.output_projection(
            outputs.reshape(batch_size, sequence_length, value_width).astype(gate.dtype)
        )

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> LinearAttentionCarry:
        batch_size, *_ = input_shape
        key_width, value_width = self.widths()
        window = partial(jnp.zeros, dtype=self.dtype)
        history = self.kernel_size - 1
        return LinearAttentionCarry(
            state=jnp.zeros(
                (batch_size, self.num_value_heads, self.key_dim, self.value_dim), jnp.float32
            ),
            cell=GatedDeltaNetCarry(
                query=window((batch_size, history, key_width)),
                key=window((batch_size, history, key_width)),
                value=window((batch_size, history, value_width)),
            ),
        )
