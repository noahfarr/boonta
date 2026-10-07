from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct

from boonta.utils.typing import Array, Dtype, Key

from .rnn import RNNCellBase


@struct.dataclass
class RTUCarry:
    real: Array
    imaginary: Array


def init_nu(key, shape, dtype=jnp.float32, r_min=0.0, r_max=1.0):
    u = jax.random.uniform(key, shape, dtype)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def init_theta(key, shape, dtype=jnp.float32, max_phase=6.28):
    u = jax.random.uniform(key, shape, dtype)
    return jnp.log(max_phase * u)


class RTUCell(RNNCellBase):
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28
    eps: float = 1e-8
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, carry: RTUCarry, x: Array) -> tuple[RTUCarry, Array]:
        dense = partial(
            nn.Dense,
            self.features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
        )
        nu = self.param(
            "nu_log",
            partial(init_nu, r_min=self.r_min, r_max=self.r_max),
            (self.features,),
            self.param_dtype,
        )
        theta = self.param(
            "theta_log",
            partial(init_theta, max_phase=self.max_phase),
            (self.features,),
            self.param_dtype,
        )

        r = jnp.exp(-jnp.exp(nu))
        g, phi = r * jnp.cos(jnp.exp(theta)), r * jnp.sin(jnp.exp(theta))
        norm = jnp.sqrt(1 - r**2) + self.eps

        carry = RTUCarry(
            real=jnp.tanh(
                g * carry.real
                - phi * carry.imaginary
                + norm * dense(name="input_real")(x)
            ),
            imaginary=jnp.tanh(
                g * carry.imaginary
                + phi * carry.real
                + norm * dense(name="input_imaginary")(x)
            ),
        )
        return carry, self.output(carry)

    def output(self, carry: RTUCarry) -> Array:
        return jnp.concatenate([carry.real, carry.imaginary], axis=-1)

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> RTUCarry:
        *batch_size, _ = input_shape
        zeros = jnp.zeros((*batch_size, self.features))
        return RTUCarry(real=zeros, imaginary=zeros)
