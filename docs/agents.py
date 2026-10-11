"""Train the agents in the README's banner and tile their renders into docs/agents.webp.

    uv sync --extra hangar --extra brax --extra mujoco-playground --extra craftax
    uv run python docs/agents.py           # train the agents that have no render yet, then tile
    uv run python docs/agents.py --tile    # only tile the renders already in runs/agents
    uv run python docs/agents.py ant       # only train the named agents, for one job each

Any other argument is a Hydra override passed to every run, such as total_timesteps=1_000_000.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageSequence

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "agents"
OUTPUT = ROOT / "docs" / "agents.webp"

CELL = (320, 240)
CAPTION = 32
COLUMNS = 2
FPS = 20
BACKGROUND = (13, 17, 23)
TEXT = (201, 209, 217)


@dataclass(frozen=True)
class Agent:
    name: str
    caption: str
    overrides: tuple[str, ...]
    stride: int = 1


AGENTS = [
    Agent("breakout", "Breakout · MinAtar", ("algorithm=ppo", "environment=gymnax/minatar/breakout")),
    Agent("ant", "Ant · Brax", ("algorithm=ppo", "environment=brax/mujoco/ant")),
    Agent("halfcheetah", "HalfCheetah · Brax", ("algorithm=ppo", "environment=brax/mujoco/halfcheetah")),
    Agent("craftax_classic", "Craftax-Classic", ("algorithm=ppo", "environment=craftax/craftax_classic/symbolic")),
]

RENDER = (
    "logger=file",
    "+artisan=[render]",
    "artisans.render.max_frames=256",
    "evaluation.num_steps=256",
)


def renders(agent: Agent) -> list[Path]:
    return sorted(
        (RUNS / agent.name).glob("render-*.gif"),
        key=lambda path: int(path.stem.removeprefix("render-")),
    )


def train(agent: Agent, overrides: list[str]) -> None:
    command = ["boonta", *agent.overrides, *RENDER, *overrides, f"hydra.run.dir={RUNS / agent.name}"]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def cell(frame: Image.Image, caption: str, font) -> Image.Image:
    width, height = CELL
    tile = Image.new("RGB", (width, height + CAPTION), BACKGROUND)
    tile.paste(ImageOps.pad(frame.convert("RGB"), CELL, Image.NEAREST, color=BACKGROUND))
    ImageDraw.Draw(tile).text(
        (width / 2, height + CAPTION / 2), caption, fill=TEXT, font=font, anchor="mm"
    )
    return tile


def tile(agents: list[Agent]) -> None:
    font = ImageFont.load_default(size=15)
    clips = []
    for agent in agents:
        *_, latest = renders(agent)
        with Image.open(latest) as gif:
            frames = [frame.copy() for frame in ImageSequence.Iterator(gif)][:: agent.stride]
        clips.append([cell(frame, agent.caption, font) for frame in frames])

    width, height = CELL[0], CELL[1] + CAPTION
    rows = -(-len(clips) // COLUMNS)
    length = max(len(clip) for clip in clips)
    canvas = []
    for step in range(length):
        frame = Image.new("RGB", (COLUMNS * width, rows * height), BACKGROUND)
        for index, clip in enumerate(clips):
            row, column = divmod(index, COLUMNS)
            frame.paste(clip[step % len(clip)], (column * width, row * height))
        canvas.append(frame)

    first, *rest = canvas
    first.save(
        OUTPUT, save_all=True, append_images=rest, duration=1000 // FPS, loop=0, quality=80
    )
    print(f"{OUTPUT.relative_to(ROOT)}: {OUTPUT.stat().st_size / 1e6:.1f} MB, {length} frames")


def main(arguments: list[str]) -> None:
    names = {agent.name for agent in AGENTS}
    chosen = [agent for agent in AGENTS if agent.name in arguments]
    overrides = [argument for argument in arguments if argument not in names | {"--tile"}]
    if "--tile" not in arguments:
        for agent in chosen or AGENTS:
            if not renders(agent):
                train(agent, overrides)
    if not chosen:
        tile(AGENTS)


if __name__ == "__main__":
    main(sys.argv[1:])
