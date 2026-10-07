import flax.linen as nn
import jax.numpy as jnp

from boonta.utils.typing import Array, Dtype


class PatchEmbedding(nn.Module):
    features: int
    patch_size: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, image: Array) -> Array:
        x = nn.Conv(
            self.features,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="projection",
        )(image)

        batch_size, height, width, features = x.shape
        x = x.reshape(batch_size, height * width, features)

        position_embedding = self.param(
            "position_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, height * width, features),
            self.param_dtype,
        )
        x, position_embedding = nn.dtypes.promote_dtype(
            x, position_embedding, dtype=self.dtype
        )
        return x + position_embedding
