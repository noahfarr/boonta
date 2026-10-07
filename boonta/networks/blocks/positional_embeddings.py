import jax
import jax.numpy as jnp
from flax import linen as nn

from boonta.utils import (add_batch_axis, add_feature_axis, add_time_axis,
                          remove_time_axis)
from boonta.utils.typing import Array, Dtype, Key

from .block import Block


def episode_positions(position: Array, done: Array) -> tuple[Array, Array]:
    _, sequence_length = done.shape
    index = add_batch_axis(jnp.arange(sequence_length))
    last_start = jax.lax.cummax(jnp.where(done.astype(bool), index, -1), axis=1)
    positions = jnp.where(
        last_start >= 0, index - last_start, add_time_axis(position) + index
    )
    return remove_time_axis(positions[:, -1:]) + 1, positions


def rotate(x: Array, positions: Array, max_wavelength: float) -> Array:
    assert (
        x.shape[-1] % 2 == 0
    ), f"features must be even, but was features: {x.shape[-1]}"

    half = x.shape[-1] // 2
    frequencies = max_wavelength ** (-jnp.arange(half) / half)
    angles = add_feature_axis(positions) * frequencies
    angles = angles[..., None, :]

    sin, cos = jnp.sin(angles), jnp.cos(angles)
    first, second = x[..., :half], x[..., half:]
    return jnp.concatenate(
        [first * cos - second * sin, second * cos + first * sin], axis=-1
    ).astype(x.dtype)


def rotary_positional_embedding(
    query: Array,
    key: Array,
    query_positions: Array,
    key_positions: Array,
    max_wavelength: float = 10_000.0,
) -> tuple[Array, Array, None]:
    return (
        rotate(query, query_positions, max_wavelength),
        rotate(key, key_positions, max_wavelength),
        None,
    )


def sinusoidal_time_embedding(
    time: Array, features: int, min_period: float = 4e-3, max_period: float = 4.0
) -> Array:
    fraction = jnp.linspace(0.0, 1.0, features // 2)
    period = min_period * (max_period / min_period) ** fraction
    phase = 2 * jnp.pi * time[:, None] / period[None, :]
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)


class SinusoidalPositionalEmbedding(Block):
    features: int
    max_wavelength: float = 10_000.0

    @nn.compact
    def __call__(self, carry: Array, x: Array, done: Array) -> tuple[Array, Array]:
        assert (
            self.features % 2 == 0
        ), f"features must be even, but was features: {self.features}"

        carry, positions = episode_positions(carry, done)

        half = self.features // 2
        frequencies = self.max_wavelength ** (-jnp.arange(half) / half)
        angles = add_feature_axis(positions) * frequencies
        embedding = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        return carry, x + embedding.astype(x.dtype)

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Array:
        batch_size, *_ = input_shape
        return jnp.zeros((batch_size,), dtype=jnp.int32)


class LearnedPositionalEmbedding(Block):
    features: int
    max_length: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, carry: Array, x: Array, done: Array) -> tuple[Array, Array]:
        carry, positions = episode_positions(carry, done)
        embedding = nn.Embed(
            self.max_length,
            self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )(jnp.minimum(positions, self.max_length - 1))
        return carry, x + embedding.astype(x.dtype)

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Array:
        batch_size, *_ = input_shape
        return jnp.zeros((batch_size,), dtype=jnp.int32)
