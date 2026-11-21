import dataclasses

import jax
import jax.numpy as jnp
from jax.sharding import Mesh


def mesh(count: int, start: int = 0) -> Mesh:
    devices = jax.devices()[start : start + count]
    assert len(devices) == count, (
        f"requested devices [{start}:{start + count}) but only "
        f"{jax.device_count()} exist"
    )
    return Mesh(devices, ("data",))


def children(node):
    paths, _ = jax.tree_util.tree_flatten_with_path(node, is_leaf=lambda v: v is not node)
    names = {
        key.name
        for (key, *_), _ in paths
        if isinstance(key, jax.tree_util.GetAttrKey)
    }
    return names or {f.name for f in dataclasses.fields(node)}


def sharded(node, axis=False):
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        values = vars(node)
        names = children(node)
        return type(node)(**{
            f.name: sharded(
                values[f.name], f.metadata["axis"] if "axis" in f.metadata else axis
            )
            if f.name in names
            else values[f.name]
            for f in dataclasses.fields(node)
        })
    return jax.tree.map(
        lambda v: sharded(v, axis)
        if dataclasses.is_dataclass(v)
        else (axis if jnp.ndim(v) else False),
        node,
        is_leaf=dataclasses.is_dataclass,
    )


def vary(node):
    def ensure(leaf, axis):
        if not axis:
            return leaf
        if axis in jax.typeof(leaf).manual_axis_type.varying:
            return leaf
        return jax.lax.pcast(leaf, axis, to="varying")

    return jax.tree.map(ensure, node, sharded(node))




def minibatches(key, tree, num_minibatches: int):
    size = jax.tree.leaves(tree)[0].shape[0]
    assert size % num_minibatches == 0, (
        f"{size} entries cannot be split into {num_minibatches} minibatches"
    )
    blocks = size // num_minibatches
    order = jnp.argsort(jax.random.uniform(key, (blocks, num_minibatches)), axis=-1)

    def split(leaf):
        grouped = leaf.reshape(blocks, num_minibatches, *leaf.shape[1:])
        index = order.reshape(order.shape + (1,) * (leaf.ndim - 1))
        return jnp.moveaxis(jnp.take_along_axis(grouped, index, axis=1), 1, 0)

    return jax.tree.map(split, tree)
