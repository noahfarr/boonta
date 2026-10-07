import flax.linen as nn
import jax.numpy as jnp

from boonta.utils import add_time_axis
from boonta.utils.typing import Array, Dtype


class AdaptiveRMSNorm(nn.Module):
    features: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, condition: Array | None = None) -> tuple[Array, Array]:
        x = nn.RMSNorm(use_scale=False, dtype=self.dtype, param_dtype=self.param_dtype)(
            x
        )

        if condition is None:
            scale = self.param(
                "scale",
                nn.initializers.zeros_init(),
                (self.features,),
                self.param_dtype,
            )
            x, scale = nn.dtypes.promote_dtype(x, scale, dtype=self.dtype)
            return x * (1 + scale), 1.0

        modulation = nn.Dense(
            self.features * 3,
            kernel_init=nn.initializers.zeros_init(),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="dense",
        )(condition)
        scale, shift, gate = jnp.split(add_time_axis(modulation), 3, axis=-1)
        return x * (1 + scale) + shift, gate
