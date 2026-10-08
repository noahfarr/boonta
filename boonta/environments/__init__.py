import ctypes
import os
import subprocess
import sys
from pathlib import Path

import jax


def capsule(address):
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return PyCapsule_New(address, None, None)


def build(directory, name, sources):
    lib_path = directory / name
    sources = [directory / source for source in sources]
    if not lib_path.exists() or any(
        lib_path.stat().st_mtime < source.stat().st_mtime for source in sources
    ):
        env = dict(
            os.environ,
            JAXLIB_INCLUDE=jax.ffi.include_dir(),
            PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        )
        subprocess.run(
            [str(directory / "build.sh")], check=True, capture_output=True, env=env
        )
    return lib_path


def load(directory, name):
    return ctypes.cdll.LoadLibrary(str(directory / name))


def register_targets(lib, targets):
    for target, getter in targets.items():
        for platform, suffix in (("cpu", "_cpu"), ("CUDA", "_cuda")):
            handler = getattr(lib, getter + suffix)
            handler.restype = ctypes.c_void_p
            jax.ffi.register_ffi_target(target, capsule(handler()), platform=platform)


from . import (ale, brax, connectx, craftax, gymnasium, gymnax, isaaclab,
               isaaclab_arena, jaxmarl, jumanji, kaggriculture, kinetix, libero,
               mapox, mujoco_playground, peanut_gb, wordle, xland_minigrid)

registry = {
    "ale": ale.make,
    "brax": brax.make,
    "connectx": connectx.make,
    "craftax": craftax.make,
    "gymnasium": gymnasium.make,
    "gymnax": gymnax.make,
    "isaaclab": isaaclab.make,
    "isaaclab_arena": isaaclab_arena.make,
    "jaxmarl": jaxmarl.make,
    "jumanji": jumanji.make,
    "kaggriculture": kaggriculture.make,
    "kinetix": kinetix.make,
    "libero": libero.make,
    "mapox": mapox.make,
    "mujoco_playground": mujoco_playground.make,
    "peanut_gb": peanut_gb.make,
    "wordle": wordle.make,
    "xland_minigrid": xland_minigrid.make,
}


def make(namespace, env_id, **kwargs):
    env_kwargs = kwargs.get("kwargs") or {}
    return registry[namespace](env_id, **env_kwargs)
