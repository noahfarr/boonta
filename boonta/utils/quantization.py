from collections.abc import Callable
from contextlib import contextmanager
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct
from jax import lax

from .typing import Array, Dtype, PyTree


@struct.dataclass
class QuantizedArray:
    values: Array
    scale: Array

    def dequantize(self) -> Array:
        return self.values.astype(self.scale.dtype) * self.scale


def quantize(
    params: PyTree,
    bits: int = 8,
    axis: int = -1,
    dtype: Dtype = jnp.bfloat16,
    min_ndim: int = 2,
) -> PyTree:
    qmax = 2 ** (bits - 1) - 1
    storage = jnp.int4 if bits <= 4 else jnp.int8

    def leaf(weight: Array | QuantizedArray) -> Array | QuantizedArray:
        if isinstance(weight, QuantizedArray) or jnp.ndim(weight) < min_ndim:
            return weight
        weight = weight.astype(jnp.float32)
        channel = axis % weight.ndim
        reduce_axes = tuple(i for i in range(weight.ndim) if i != channel)
        scale = jnp.max(jnp.abs(weight), axis=reduce_axes, keepdims=True) / qmax
        scale = jnp.where(scale == 0, 1, scale).astype(dtype)
        values = jnp.round(weight / scale).clip(-qmax, qmax).astype(storage)
        return QuantizedArray(values=values, scale=scale)

    return jax.tree_util.tree_map(
        leaf, params, is_leaf=lambda x: isinstance(x, QuantizedArray)
    )


def dequantize(params: PyTree) -> PyTree:
    def leaf(value: Array | QuantizedArray) -> Array:
        return value.dequantize() if isinstance(value, QuantizedArray) else value

    return jax.tree_util.tree_map(
        leaf, params, is_leaf=lambda x: isinstance(x, QuantizedArray)
    )


def materialize_gradients() -> optax.GradientTransformation:
    def update(updates: PyTree, state, params: PyTree | None = None):
        def leaf(gradient: Array) -> Array:
            if gradient.dtype == jax.dtypes.float0:
                return jnp.zeros(gradient.shape, jnp.float32)
            return gradient

        return jax.tree.map(leaf, updates), state

    return optax.GradientTransformation(lambda params: optax.EmptyState(), update)


def quantized_dot_general(weight_bits: int = 8, activation_bits: int = 8) -> Callable:
    def quant(x: Array, bits: int, contract: tuple[int, ...]) -> tuple[Array, Array]:
        qmax = 2 ** (bits - 1) - 1
        scale = jnp.max(jnp.abs(x), axis=contract, keepdims=True) / qmax
        scale = jnp.where(scale == 0, 1, scale)
        values = jnp.round(x / scale).clip(-qmax, qmax).astype(jnp.int8)
        return values, scale

    def rescale(
        acc: Array,
        lhs_scale: Array,
        rhs_scale: Array,
        dimension_numbers,
        lhs_ndim: int,
        rhs_ndim: int,
    ) -> Array:
        (lhs_c, rhs_c), (lhs_b, rhs_b) = dimension_numbers
        lhs_c, rhs_c = tuple(lhs_c), tuple(rhs_c)
        lhs_b, rhs_b = tuple(lhs_b), tuple(rhs_b)
        lhs_free = tuple(
            i for i in range(lhs_ndim) if i not in lhs_c and i not in lhs_b
        )
        rhs_free = tuple(
            i for i in range(rhs_ndim) if i not in rhs_c and i not in rhs_b
        )
        n_batch = len(lhs_b)

        lhs_s = jnp.transpose(lhs_scale, lhs_b + lhs_free + lhs_c)
        lhs_s = lhs_s.reshape(lhs_s.shape[: n_batch + len(lhs_free)])
        lhs_s = lhs_s.reshape(lhs_s.shape + (1,) * len(rhs_free))

        rhs_s = jnp.transpose(rhs_scale, rhs_b + rhs_free + rhs_c)
        rhs_s = rhs_s.reshape(rhs_s.shape[: n_batch + len(rhs_free)])
        rhs_s = rhs_s.reshape(
            rhs_s.shape[:n_batch] + (1,) * len(lhs_free) + rhs_s.shape[n_batch:]
        )

        return acc * lhs_s * rhs_s

    def dot_general(
        lhs: Array,
        rhs: Array,
        dimension_numbers,
        precision=None,
        preferred_element_type: Dtype | None = None,
        **kwargs,
    ) -> Array:
        (lhs_c, rhs_c), _ = dimension_numbers
        lhs_q, lhs_s = quant(lhs, activation_bits, tuple(lhs_c))
        rhs_q, rhs_s = quant(rhs, weight_bits, tuple(rhs_c))
        acc = lax.dot_general(
            lhs_q, rhs_q, dimension_numbers, preferred_element_type=jnp.int32
        ).astype(jnp.float32)
        out = rescale(acc, lhs_s, rhs_s, dimension_numbers, lhs.ndim, rhs.ndim)
        return out.astype(preferred_element_type or lhs.dtype)

    return dot_general


@contextmanager
def quantizing(weights: int = 8, activations: int = 8):
    dot_general = quantized_dot_general(weights, activations)
    saved = nn.Dense, nn.DenseGeneral
    nn.Dense, nn.DenseGeneral = (partial(cls, dot_general=dot_general) for cls in saved)
    try:
        yield
    finally:
        nn.Dense, nn.DenseGeneral = saved
