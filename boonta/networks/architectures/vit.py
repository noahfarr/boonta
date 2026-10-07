from functools import partial

import flax.linen as nn
import jax.numpy as jnp

from boonta.utils import add_time_axis
from boonta.utils.typing import Array, Carry, Dtype, Key

from ..blocks import FFN, Block, PatchEmbedding, SelfAttention


class ViTLayer(Block):
    features: int
    num_heads: int
    expansion_factor: int = 4
    hidden_dim: int | None = None
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        layer_norm = partial(
            nn.LayerNorm, dtype=self.dtype, param_dtype=self.param_dtype
        )
        _, y = SelfAttention(
            features=self.features,
            num_heads=self.num_heads,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="attention",
        )(None, layer_norm(name="attention_norm")(x), done)
        x = x + y
        _, y = FFN(
            self.features,
            self.expansion_factor,
            hidden_dim=self.hidden_dim,
            activation=partial(nn.gelu, approximate=True),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="ffn",
        )(None, layer_norm(name="ffn_norm")(x), done)
        return carry, x + y

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return None


class ViT(Block):
    patch_size: int
    features: int
    num_layers: int
    num_heads: int
    expansion_factor: int = 4
    hidden_dim: int | None = None
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, carry: Carry, image: Array, done: Array) -> tuple[Carry, Array]:
        x = PatchEmbedding(
            self.features,
            self.patch_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="patch_embedding",
        )(image)

        batch_size, num_patches, *_ = x.shape
        done = jnp.broadcast_to(add_time_axis(done), (batch_size, num_patches))

        for i in range(self.num_layers):
            _, x = ViTLayer(
                self.features,
                self.num_heads,
                self.expansion_factor,
                hidden_dim=self.hidden_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"layers_{i}",
            )(None, x, done)

        x = nn.LayerNorm(
            dtype=self.dtype, param_dtype=self.param_dtype, name="output_norm"
        )(x)
        return carry, x

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return None
