import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import jax
from flax import serialization


REMOTE = re.compile(r"^(?P<host>[A-Za-z0-9_.@-]{2,}):(?P<path>(?!//).+)$")
LATEST = (
    'd={path}; [ -d "$d/checkpoints" ] && d="$d/checkpoints/latest"; '
    'if [ -d "$d" ]; then s=$(ls "$d" | grep -E "^[0-9]+$" | sort -n | tail -1); [ -n "$s" ] && d="$d/$s"; fi; '
    'cd "$d" 2>/dev/null && pwd -P || echo "$d"'
)


def remote_parts(path):
    match = REMOTE.match(str(path))
    if match is None or Path(str(path)).exists():
        return None
    return match["host"], match["path"]


def cache_directory():
    return (
        Path(os.environ.get("BOONTA_CACHE", "~/.cache/boonta")).expanduser()
        / "checkpoints"
    )


def quoted(path):
    return (
        '"$HOME"/' + shlex.quote(path[2:])
        if path.startswith("~/")
        else shlex.quote(path)
    )


def pull_checkpoint(host, path):
    ssh = ["ssh", "-o", "BatchMode=yes", host]
    found = subprocess.run(
        [*ssh, LATEST.format(path=quoted(path))], capture_output=True, text=True
    )
    remote = (
        found.stdout.strip().splitlines()[-1]
        if found.returncode == 0 and found.stdout.strip()
        else None
    )
    if remote is None:
        raise RuntimeError(
            f"could not reach {host}:{path} over ssh: {found.stderr.strip()}"
        )
    target = cache_directory() / host / remote.lstrip("/")
    if (target / ".complete").exists():
        return str(target)
    target.mkdir(parents=True, exist_ok=True)
    print(f"pulling {host}:{remote} to {target}", file=sys.stderr)
    if shutil.which("rsync"):
        copy = [
            "rsync",
            "-a",
            "-e",
            "ssh -o BatchMode=yes",
            f"{host}:{remote}/",
            f"{target}/",
        ]
    else:
        copy = ["scp", "-r", "-o", "BatchMode=yes", f"{host}:{remote}/.", str(target)]
    done = subprocess.run(copy, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"could not copy {host}:{remote}: {done.stderr.strip()}")
    (target / ".complete").touch()
    return str(target)


def localize(path):
    parts = remote_parts(path) if path is not None else None
    return pull_checkpoint(*parts) if parts else path


def resolve_checkpoint(path):
    if path is None:
        return None
    path = localize(path)
    held = Path(path)
    if (held / "checkpoints").is_dir():
        held = held / "checkpoints" / "latest"
    if not held.is_dir():
        return str(held)
    steps = [item for item in held.iterdir() if item.name.isdigit()]
    if not steps:
        return str(held)
    return str(max(steps, key=lambda item: int(item.name)))


def restore_item(directory, node):
    import orbax.checkpoint as ocp

    checkpoint = ocp.StandardCheckpointer().restore(
        directory,
        target=jax.tree.map(
            lambda leaf: None if getattr(leaf, "size", 1) == 0 else leaf, node
        ),
    )
    return jax.tree.map(
        lambda kept, leaf: leaf if kept is None else kept,
        checkpoint,
        node,
        is_leaf=lambda leaf: leaf is None,
    )


def load_checkpoint(path, target=None):
    if path is None:
        return target
    path = Path(localize(path))
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
        items = sorted(item for item in path.iterdir() if (item / "_METADATA").exists())
    if not items:
        raise FileNotFoundError(f"no checkpoint items under {path}")
    return target.replace(
        **{item.name: restore_item(os.path.abspath(str(item)), getattr(target, item.name)) for item in items}
    )
