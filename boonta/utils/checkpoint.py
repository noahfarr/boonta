import os
from pathlib import Path

from flax import serialization


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


def save_checkpoint(path, items):
    import orbax.checkpoint as ocp

    path = Path(path).absolute()
    with ocp.StandardCheckpointer() as checkpointer:
        for name, tree in items.items():
            checkpointer.save(path / name, tree)


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
            item.name: ocp.StandardCheckpointer().restore(
                os.path.abspath(str(item)), target=getattr(target, item.name)
            )
            for item in items
        }
    )
