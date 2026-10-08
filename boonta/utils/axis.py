import jax
import jax.numpy as jnp

from .typing import Array, PyTree


def add_feature_axis(x: Array) -> Array:
    return jnp.expand_dims(x, axis=-1)


def remove_feature_axis(x: Array) -> Array:
    return jnp.squeeze(x, axis=-1)


def add_time_axis(x: Array) -> Array:
    return jnp.expand_dims(x, axis=1)


def remove_time_axis(x: PyTree) -> PyTree:
    return jax.tree.map(lambda leaf: jnp.squeeze(leaf, axis=1), x)


def add_batch_axis(x: Array) -> Array:
    return x[None]


def remove_batch_axis(x: Array) -> Array:
    return x[0]


def broadcast(x: Array, like: Array) -> Array:
    assert x.ndim <= like.ndim, f"cannot broadcast {x.ndim} dims against {like.ndim}"
    return x.reshape(x.shape + (1,) * (like.ndim - x.ndim))


def flatten(x: Array, start_dim: int = 0, end_dim: int = -1) -> Array:
    if x.ndim == 0:
        return x

    start_dim = start_dim % x.ndim
    end_dim = end_dim % x.ndim

    if start_dim >= end_dim:
        return x

    return x.reshape(*x.shape[:start_dim], -1, *x.shape[end_dim + 1 :])


def take(tree: PyTree, index: int, count: int, axis: int = 0) -> PyTree:
    def cut(leaf):
        width = leaf.shape[axis] // count
        return jax.lax.slice_in_dim(leaf, index * width, (index + 1) * width, axis=axis)

    return jax.tree.map(cut, tree)


def place(values: tuple, index: int, value) -> tuple:
    return tuple(value if slot == index else old for slot, old in enumerate(values))


def concatenate(pieces: list, axis: int = 0) -> PyTree:
    return jax.tree.map(lambda *leaves: jnp.concatenate(leaves, axis=axis), *pieces)
