"""Train the agents in the README's banner and tile their renders into docs/agents.webp.

    uv sync --extra hangar --extra kinetix --extra craftax
    uv run python docs/agents.py           # train the agents that have no render yet, then tile
    uv run python docs/agents.py --tile    # only tile the renders already in runs/agents
    uv run python docs/agents.py kinetix   # only train the named agents, for one job each
    uv run python docs/agents.py --render  # re-render each run from its checkpoint, losslessly, then tile

Any other argument is a Hydra override passed to every run, such as total_timesteps=1_000_000.
Pokémon Red needs your own ROM at boonta/environments/peanut_gb/roms/pokemon_red.gb.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageSequence

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "agents"
OUTPUT = ROOT / "docs" / "agents.webp"

TILE = 320
GAP = 8
FPS = 20
QUALITY = 92


@dataclass(frozen=True)
class Agent:
    name: str
    overrides: tuple[str, ...]
    seed: int = 0
    box: tuple[int, int, int] | None = None
    length: int | None = None


AGENTS = [
    Agent(
        "breakout",
        (
            "algorithm=ppo",
            "environment=gymnax/minatar/breakout",
            "+artisans.render.kwargs.block=32",
        ),
    ),
    Agent(
        "kinetix",
        (
            "algorithm=ppo",
            "environment=kinetix/kinetix",
            "environment.env_id=[m/h1_car_left,m/h2_car_right,m/h3_car_thrust,m/h7_car_flip]",
            "total_timesteps=200_000_000",
        ),
        seed=2,
        box=(160, 165, 335),
        length=83,
    ),
    Agent(
        "pokemon_red",
        (
            "algorithm=recurrent_ppo",
            "environment=peanut_gb/pokemon_red",
            "+artisans.render.kwargs.scale=4",
        ),
    ),
    Agent(
        "craftax_classic",
        (
            "algorithm=ppo",
            "environment=craftax/craftax_classic/symbolic",
            "+artisans.render.kwargs.block=64",
        ),
    ),
]

RENDER = (
    "logger=[file,orbax]",
    "loggers.orbax.max_to_keep=1",
    "+artisan=[render,checkpointer]",
    "artisans.render.max_frames=256",
    "evaluation.num_steps=256",
)


def renders(agent: Agent) -> list[Path]:
    return sorted(
        (RUNS / agent.name).glob("render-*.gif"),
        key=lambda path: int(path.stem.removeprefix("render-")),
    )


def train(agent: Agent, overrides: list[str]) -> None:
    command = ["boonta", *RENDER, *agent.overrides, *overrides, f"hydra.run.dir={RUNS / agent.name}"]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def render(agent: Agent) -> None:
    import jax
    import numpy as np
    import yaml
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import instantiate

    from boonta.utils import load_checkpoint, newest
    from hangar import recipes

    run = RUNS / agent.name
    overrides = yaml.safe_load((run / ".hydra" / "overrides.yaml").read_text())
    with initialize_config_dir(str(ROOT / "hangar" / "config"), version_base=None):
        cfg = compose(
            "config",
            overrides=[*overrides, f"hydra.run.dir={run}"],
            return_hydra_config=True,
        )
    HydraConfig.instance().set_config(cfg)
    podracer = recipes.make(cfg)
    init_key, evaluate_key = jax.random.split(jax.random.key(agent.seed))
    state = load_checkpoint(newest(run), podracer.init(init_key))
    _, logs = podracer.evaluate(state, evaluate_key, cfg.evaluation.num_steps)
    video = instantiate(cfg.artisans.render).craft(podracer.algorithm, podracer.environment, state, logs)
    np.save(run / "frames.npy", video.data)
    print(f"{agent.name}: {video.data.shape} frames from {newest(run)}", flush=True)


def clip(agent: Agent) -> list[Image.Image]:
    frames = RUNS / agent.name / "frames.npy"
    if frames.exists():
        import numpy as np

        frames = [Image.fromarray(frame) for frame in np.load(frames)]
    else:
        *_, latest = renders(agent)
        with Image.open(latest) as gif:
            frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    return [square(frame, agent.box) for frame in frames[: agent.length]]


def square(frame: Image.Image, box: tuple[int, int, int] | None = None) -> Image.Image:
    frame = frame.convert("RGB")
    side = min(frame.size)
    left, top = (frame.width - side) // 2, (frame.height - side) // 2
    if box is not None:
        left, top, side = box
    frame = frame.crop((left, top, left + side, top + side))
    resample = Image.NEAREST if side <= TILE else Image.LANCZOS
    return frame.resize((TILE, TILE), resample)


def tile(agents: list[Agent]) -> None:
    clips = [clip(agent) for agent in agents]

    length = max(len(frames) for frames in clips)
    width = len(clips) * TILE + (len(clips) - 1) * GAP
    canvas = []
    for step in range(length):
        frame = Image.new("RGBA", (width, TILE), (0, 0, 0, 0))
        for index, frames in enumerate(clips):
            frame.paste(frames[step % len(frames)], (index * (TILE + GAP), 0))
        canvas.append(frame)

    first, *rest = canvas
    first.save(
        OUTPUT,
        save_all=True,
        append_images=rest,
        duration=1000 // FPS,
        loop=0,
        quality=QUALITY,
        method=6,
    )
    print(f"{OUTPUT.relative_to(ROOT)}: {OUTPUT.stat().st_size / 1e6:.1f} MB, {length} frames")


def main(arguments: list[str]) -> None:
    names = {agent.name for agent in AGENTS}
    chosen = [agent for agent in AGENTS if agent.name in arguments]
    flags = {"--tile", "--render"}
    overrides = [argument for argument in arguments if argument not in names | flags]
    if "--render" in arguments:
        for agent in chosen or AGENTS:
            render(agent)
    elif "--tile" not in arguments:
        for agent in chosen or AGENTS:
            if not renders(agent):
                train(agent, overrides)
    if not chosen:
        tile(AGENTS)


if __name__ == "__main__":
    main(sys.argv[1:])
