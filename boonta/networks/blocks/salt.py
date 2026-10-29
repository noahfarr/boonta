from functools import partial

import flax.linen as nn
import jax.numpy as jnp

from boonta.utils.typing import Array, Dtype, Key

from .delta_net import inverse_softplus_init, l2_normalize, log_uniform_init
from .linear_attention import (
    LinearAttentionCarry,
    LinearAttentionCellBase,
    LinearAttentionInputs,
)


def interleave(a: Array, b: Array) -> Array:
    batch_size, sequence_length, *rest = a.shape
    return jnp.stack([a, b], axis=2).reshape(batch_size, 2 * sequence_length, *rest)


class SALT(LinearAttentionCellBase):
    features: int
    action_encoder: nn.Module
    num_heads: int = 2
    key_dim: int = 16
    value_dim: int = 16
    epsilon: float = 1e-5
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()

    def setup(self):
        key_width = self.num_heads * self.key_dim
        value_width = self.num_heads * self.value_dim
        dense = partial(
            nn.Dense,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
        )
        self.query = dense(key_width)
        self.state_key = dense(key_width)
        self.state_value = dense(value_width)
        self.action_key = dense(key_width)
        self.action_value = dense(value_width)
        self.alpha = dense(self.num_heads)
        self.beta = dense(self.num_heads)
        self.gamma = dense(
            self.num_heads, use_bias=True, bias_init=nn.initializers.constant(-4.0)
        )
        self.log_rate = self.param(
            "log_rate", log_uniform_init(), (self.num_heads,), self.param_dtype
        )
        self.step_bias = self.param(
            "step_bias", inverse_softplus_init(), (self.num_heads,), self.param_dtype
        )
        self.norm = nn.RMSNorm(
            epsilon=self.epsilon, dtype=self.dtype, param_dtype=self.param_dtype
        )
        self.output_projection = nn.Dense(
            self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
        )

    def __call__(
        self, carry: None, x: Array, done: Array, action: Array, **kwargs: Array
    ) -> tuple[None, LinearAttentionInputs]:
        batch_size, sequence_length, _ = x.shape
        u = self.action_encoder(action).astype(x.dtype)
        joint = jnp.concatenate([x, u], axis=-1)

        def heads(x: Array) -> Array:
            return x.reshape(batch_size, sequence_length, self.num_heads, -1)

        def keys(x: Array) -> Array:
            return l2_normalize(nn.silu(heads(x)).astype(jnp.float32)).astype(x.dtype)

        query = keys(self.query(joint))
        log_decay = -jnp.exp(self.log_rate.astype(jnp.float32)) * nn.softplus(
            self.alpha(joint).astype(jnp.float32) + self.step_bias.astype(jnp.float32)
        )
        beta = 2.0 * nn.sigmoid(self.beta(joint).astype(jnp.float32))
        gamma = nn.sigmoid(self.gamma(joint).astype(jnp.float32))
        inputs = LinearAttentionInputs(
            query=interleave(query, query),
            key=interleave(keys(self.action_key(u)), keys(self.state_key(x))),
            value=interleave(heads(self.action_value(u)), heads(self.state_value(x))),
            log_decay=interleave(log_decay, jnp.zeros_like(log_decay)),
            erase_gate=interleave(jnp.zeros_like(beta), beta),
            write_gate=interleave(gamma, beta),
        )
        return carry, inputs

    def output(self, outputs: Array, x: Array) -> Array:
        batch_size, sequence_length, _ = x.shape
        outputs = self.norm(outputs.astype(self.dtype or outputs.dtype))
        return self.output_projection(outputs.reshape(batch_size, sequence_length, -1))

    @nn.nowrap
    def initialize_carry(
        self, key: Key, input_shape: tuple[int, ...]
    ) -> LinearAttentionCarry:
        batch_size, *_ = input_shape
        return LinearAttentionCarry(
            state=jnp.zeros(
                (batch_size, self.num_heads, self.key_dim, self.value_dim), jnp.float32
            ),
            cell=None,
        )
