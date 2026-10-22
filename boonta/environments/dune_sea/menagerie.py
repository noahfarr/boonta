import os
import subprocess
from pathlib import Path

REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT = "cda5236f98383f6fbaa4820c6b18dba892cd087f"


def cache() -> Path:
    root = os.environ.get("BOONTA_MENAGERIE")
    if root:
        return Path(root)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "boonta" / "mujoco_menagerie" / COMMIT[:12]


def fetch(robot: str) -> Path:
    root = cache()
    if (root / robot).is_dir():
        return root / robot
    root.mkdir(parents=True, exist_ok=True)
    git = lambda *args: subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    if not (root / ".git").is_dir():
        git("init", "-q")
        git("remote", "add", "origin", REPOSITORY)
    git("sparse-checkout", "set", robot)
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", COMMIT)
    git("checkout", "-q", COMMIT)
    return root / robot
