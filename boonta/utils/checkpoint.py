import os
from pathlib import Path

import jax
from flax import serialization


def hollow(tree):
    return jax.tree.map(lambda leaf: None if getattr(leaf, "size", 1) == 0 else leaf, tree)


def refill(template, restored):
    kept = iter(jax.tree.leaves(restored))
    return jax.tree.unflatten(
        jax.tree.structure(template),
        [leaf if getattr(leaf, "size", 1) == 0 else next(kept) for leaf in jax.tree.leaves(template)],
    )


def newest(path, which="latest"):
    if path is None:
        return None
    held = Path(path)
    if (held / "checkpoints").is_dir():
        held = held / "checkpoints" / which
    if not held.is_dir():
        return str(held)
    steps = [item for item in held.iterdir() if item.name.isdigit()]
    if not steps:
        return str(held)
    return str(max(steps, key=lambda item: int(item.name)))


def load_checkpoint(path, target=None):
    if path is None:
        return target
    path = Path(path)
    if not path.is_dir():
        data = path.read_bytes()
        if target is None:
            return serialization.msgpack_restore(data)
        return serialization.from_bytes(target, data)

    import orbax.checkpoint as ocp

    leaf = (path / "_METADATA").exists()
    if leaf and target is None:
        return ocp.StandardCheckpointer().restore(os.path.abspath(str(path)))
    items = [path]
    if not leaf:
        items = sorted(
            item for item in path.iterdir() if (item / "_METADATA").exists()
        )
    if not items:
        raise FileNotFoundError(f"no checkpoint items under {path}")
    return target.replace(
        **{
            item.name: refill(
                getattr(target, item.name),
                ocp.StandardCheckpointer().restore(
                    os.path.abspath(str(item)), target=hollow(getattr(target, item.name))
                ),
            )
            for item in items
        }
    )
