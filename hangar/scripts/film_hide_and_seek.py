import argparse
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from boonta.environments.dune_sea import HideAndSeek, Layout
from boonta.environments.dune_sea.hide_and_seek import (
    EMPTY,
    GRAB,
    HIDER,
    JUMP,
    LEFT,
    LOCK,
    NOOP,
    REACH,
    RIGHT,
    UNLOCKED,
)
from boonta.environments.dune_sea.hide_and_seek_render import HIDER_BLUE, SEEKER_RED

FPS = 30
FOOTER_HEIGHT = 48
SHIELD, RAMP = 0, 1
SCENE = Layout(
    wall_x=jnp.array([8.2]),
    wall_height=jnp.array([1.0]),
    object_x=jnp.array([12.6, 4.0]),
    mirrored=jnp.array([False]),
    agent_x=jnp.array([10.0, 11.6, 1.0, 2.2]),
)
PLANS = (
    [("lock", RAMP), ("go", 15.0), ("wait",)],
    [
        ("grab", SHIELD),
        ("go", 13.0),
        ("release",),
        ("lock", SHIELD),
        ("go", 14.3),
        ("wait",),
    ],
    [("chase",)],
    [("grab", RAMP), ("carry", 11.6), ("release",), ("chase",)],
)
STUCK = 0.3
RISING = 0.8


class Scripted:
    def __init__(self, index, plan, team):
        self.index, self.plan, self.team = index, list(plan), team
        self.pushed = False
        self.climbing = False

    def act(self, view):
        while self.plan:
            kind, *args = self.plan[0]
            action = getattr(self, kind)(view, *args)
            if action is not None:
                return action
            self.plan.pop(0)
        return NOOP

    def walk(self, view, target, tolerance=0.15):
        offset = target - view["position"][self.index, 0]
        if abs(offset) < tolerance:
            self.pushed = False
            return None
        direction = 1.0 if offset > 0 else -1.0
        if self.climbing and view["velocity"][self.index, 1] > RISING:
            return NOOP
        self.climbing = False
        speed = view["velocity"][self.index, 0] * direction
        blocked = self.pushed and view["grounded"][self.index] and speed < STUCK
        self.pushed = not blocked
        if blocked:
            self.climbing = True
            return JUMP
        return RIGHT if direction > 0 else LEFT

    def reach(self, view, target, press, finished):
        if finished:
            return None
        if view["gap"][self.index, target] < REACH * 0.8:
            self.pushed = False
            return press
        return self.walk(view, view["objects"][target, 0], tolerance=0.0)

    def grab(self, view, target):
        holding = view["holding"][self.index]
        return self.reach(
            view,
            target,
            GRAB,
            holding == target or view["locked_by"][target] != UNLOCKED,
        )

    def lock(self, view, target):
        return self.reach(view, target, LOCK, view["locked_by"][target] == self.team)

    def go(self, view, target):
        return self.walk(view, target)

    def carry(self, view, target):
        if view["holding"][self.index] == EMPTY:
            return None
        return self.walk(view, target, tolerance=0.3)

    def release(self, view):
        return GRAB if view["holding"][self.index] != EMPTY else None

    def wait(self, view):
        return NOOP

    def chase(self, view):
        hiders = np.flatnonzero(view["team"] == HIDER)
        if view["sight"][self.index, hiders].any():
            return NOOP
        here = view["position"][self.index, 0]
        target = view["position"][
            hiders[np.argmin(np.abs(view["position"][hiders, 0] - here))], 0
        ]
        return self.walk(view, target, tolerance=0.0)


def snapshot(env, state, sight, gap):
    return {
        "position": np.asarray(state.sim.circle.position),
        "velocity": np.asarray(state.sim.circle.velocity),
        "grounded": np.asarray(state.grounded),
        "holding": np.asarray(state.holding),
        "locked_by": np.asarray(state.locked_by),
        "objects": np.asarray(state.sim.polygon.position[env.first_object :]),
        "team": np.asarray(env.team),
        "sight": np.asarray(sight),
        "gap": np.asarray(gap),
    }


def annotate(frame, step, steps, preparation_steps, info, totals, font):
    scene = Image.fromarray(frame)
    width, height = scene.size
    footer_width = max(width, 440)
    footer = Image.new("RGB", (footer_width, FOOTER_HEIGHT), (239, 243, 248))
    draw = ImageDraw.Draw(footer)
    ink = (34, 45, 60)
    left, right = 14, footer_width - 14
    status = (
        "Preparation"
        if info["preparing"]
        else ("Hiders hidden" if info["hidden"] else "Hiders seen")
    )
    status_colour = SEEKER_RED if status == "Hiders seen" else ink
    draw.text((left, 12), f"{step / FPS:04.1f} s", font=font, fill=ink)
    draw.text(
        (footer_width / 2, 12), status, font=font, fill=status_colour, anchor="mt"
    )
    scores = f"Seekers {totals[1]:+.0f}"
    draw.text((right, 12), scores, font=font, fill=SEEKER_RED, anchor="rt")
    draw.text(
        (right - draw.textlength(scores, font=font) - 16, 12),
        f"Hiders {totals[0]:+.0f}",
        font=font,
        fill=HIDER_BLUE,
        anchor="rt",
    )

    split = left + round((right - left) * preparation_steps / steps)
    cursor = left + round((right - left) * step / steps)
    draw.line((left, 35, split, 35), fill=(153, 180, 214), width=3)
    draw.line((split, 35, right, 35), fill=(192, 201, 213), width=3)
    draw.line((left, 35, cursor, 35), fill=ink, width=3)
    draw.line((split, 31, split, 39), fill=ink)
    draw.ellipse((cursor - 3, 32, cursor + 3, 38), fill=ink)

    if width < footer_width:
        footer = footer.resize(
            (width, max(1, round(FOOTER_HEIGHT * width / footer_width))),
            Image.Resampling.LANCZOS,
        )
    image = Image.new("RGB", (width, height + footer.height))
    image.paste(scene)
    image.paste(footer, (0, height))
    return np.asarray(image)


def ffmpeg():
    try:
        import imageio_ffmpeg
    except ImportError:
        return "ffmpeg"
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/hide_and_seek/scripted.mp4")
    parser.add_argument("--scale", type=int, default=64)
    parser.add_argument("--time-limit", type=int, default=360)
    args = parser.parse_args()

    if args.scale < 1 or args.time_limit < 1:
        parser.error("--scale and --time-limit must be positive")

    env = HideAndSeek(
        num_hiders=2,
        num_seekers=2,
        num_boxes=1,
        num_ramps=1,
        num_walls=1,
        time_limit=args.time_limit,
        generator=lambda *_: SCENE,
    )

    step, sight, gaps = jax.jit(env.step), jax.jit(env.sight), jax.jit(env.gaps)
    key = jax.random.key(0)
    state, timestep = env.init(key)
    agents = [
        Scripted(index, plan, int(env.team[index])) for index, plan in enumerate(PLANS)
    ]
    font = ImageFont.load_default(size=14)
    totals = np.zeros(2)
    hiders = np.asarray(env.team) == HIDER

    def frame(index, state, timestep):
        arena = np.asarray(env.render(state, scale=args.scale, height=2.65, sight=True))
        return annotate(
            arena,
            index,
            args.time_limit,
            env.preparation_steps,
            timestep.info,
            totals,
            font,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, columns, _ = frame(0, state, timestep).shape
    command = [
        ffmpeg(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{columns}x{rows}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=0xeff3f8",
        "-c:v",
        "libx264",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for index in range(1, args.time_limit + 1):
            view = snapshot(env, state, sight(state), gaps(state.sim))
            actions = jnp.array([agent.act(view) for agent in agents], jnp.int32)
            state, timestep = step(key, state, actions)
            reward = np.asarray(timestep.reward)
            totals += [reward[hiders].mean(), reward[~hiders].mean()]
            encoder.stdin.write(frame(index, state, timestep).tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise subprocess.CalledProcessError(encoder.returncode, command)
    print(
        f"wrote {output} ({args.time_limit} frames), returns: hiders {totals[0]:+.0f}, seekers {totals[1]:+.0f}"
    )


if __name__ == "__main__":
    main()
