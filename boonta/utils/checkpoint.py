import os
from pathlib import Path

import jax
import numpy as np
from flax import serialization


def empty(leaf) -> bool:
    return np.size(leaf) == 0


def prune(tree, cond_fn=empty):
    return jax.tree.map(lambda leaf: None if cond_fn(leaf) else leaf, tree)


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
            item.name: jax.tree.map(
                lambda kept, saved: kept if saved is None else saved,
                getattr(target, item.name),
                ocp.StandardCheckpointer().restore(
                    os.path.abspath(str(item)), target=prune(getattr(target, item.name))
                ),
            )
            for item in items
        }
    )
