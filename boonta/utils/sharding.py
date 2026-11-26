import dataclasses
import math
import re
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P

def mesh(count: int, start: int = 0, model: int = 1) -> Mesh:
    devices = jax.devices()[start : start + count]
    assert len(devices) == count, (
        f"requested devices [{start}:{start + count}) but only "
        f"{jax.device_count()} exist"
    )
    assert count % model == 0, f"{count} devices do not split into model={model}"
    return jax.make_mesh(
        (count // model, model),
        ("data", "model"),
        axis_types=(AxisType.Auto,) * 2,
        devices=devices,
    )


def _normalize(spec: P, mesh: Mesh) -> P:
    def keep(entry):
        names = entry if isinstance(entry, tuple) else (entry,)
        names = tuple(
            name
            for name in names
            if name is not None and (name == "data" or mesh.shape.get(name, 1) > 1)
        )
        if not names:
            return None
        return names[0] if len(names) == 1 else names

    entries = [keep(entry) for entry in spec]
    while entries and entries[-1] is None:
        entries.pop()
    return P(*entries)


def _name(path) -> str:
    def part(key):
        for attribute in ("name", "key", "idx"):
            if hasattr(key, attribute):
                return str(getattr(key, attribute))
        return str(key)

    return "/".join(part(key) for key in path)


def _fit(name: str, shape: tuple[int, ...], spec: P, mesh: Mesh) -> P:
    if len(spec) > len(shape):
        raise ValueError(f"{name} has shape {shape}, too few dimensions for {spec}")
    for size, entry in zip(shape, spec):
        names = entry if isinstance(entry, tuple) else (entry,)
        ways = math.prod(mesh.shape[name] for name in names if name is not None)
        if size % ways:
            raise ValueError(f"{name} has shape {shape}, which {spec} cannot split evenly")
    return spec


def _resolve(name, leaf, axis, mesh, rules) -> P | None:
    if leaf is None or axis is None:
        return None
    shape = tuple(jnp.shape(leaf))
    leading = "data" if axis == "batch" else axis
    for pattern, spec in rules:
        if re.search(pattern, name):
            spec = P(*("data" if entry == "batch" else entry for entry in spec))
            if axis:
                spec = P(leading, *spec[1:])
            return _fit(name, shape, _normalize(spec, mesh), mesh)
    if axis:
        return _normalize(P(leading), mesh)
    if not rules or math.prod(shape) <= 1:
        return P()
    raise ValueError(f"no partition rule matches {name} with shape {shape}")


def pin(tree, shardings):
    if shardings is None:
        return tree
    is_none = lambda value: value is None
    leaves, treedef = jax.tree_util.tree_flatten(tree, is_leaf=is_none)
    targets = jax.tree_util.tree_leaves(shardings, is_leaf=is_none)
    pinned = [
        leaf if leaf is None or target is None else jax.lax.with_sharding_constraint(leaf, target)
        for leaf, target in zip(leaves, targets, strict=True)
    ]
    return jax.tree_util.tree_unflatten(treedef, pinned)


def partition(node, mesh: Mesh, rules: Sequence[tuple[str, P]] = ()):
    model = mesh.shape.get("model", 1)
    assert rules or model == 1, (
        f"a mesh with model={model} needs partition rules to split the model across its "
        f"devices, and none were given"
    )
    is_none = lambda value: value is None
    paths, treedef = jax.tree_util.tree_flatten_with_path(node, is_leaf=is_none)
    axes = jax.tree_util.tree_leaves(sharded(node), is_leaf=is_none)
    specs = [
        _resolve(_name(path), leaf, axis, mesh, rules)
        for (path, leaf), axis in zip(paths, axes, strict=True)
    ]
    return jax.tree_util.tree_unflatten(treedef, specs)


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
        axis = "data" if axis == "batch" else axis
        if axis in jax.typeof(leaf).manual_axis_type.varying:
            return leaf
        return jax.lax.pcast(leaf, axis, to="varying")

    return jax.tree.map(ensure, node, sharded(node))


