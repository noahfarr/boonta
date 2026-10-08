from ..store import place


def environment(node, name):
    place(node, f"environment/{name}", package="environment")


from . import (  # noqa: E402, F401
    ale,
    brax,
    connectx,
    craftax,
    gymnasium,
    gymnax,
    isaaclab,
    jaxmarl,
    jumanji,
    kinetix,
    libero,
    mapox,
    mujoco_playground,
    peanut_gb,
    wordle,
    xland_minigrid,
)
