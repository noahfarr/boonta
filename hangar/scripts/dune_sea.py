import argparse

import jax
import jax.numpy as jnp
import numpy as np
import pyray as pr

from boonta.environments.dune_sea import DuneSea
from boonta.environments.dune_sea.dune_sea import (AGENT_COLOR, BACKWARD, FORWARD, GROUND_COLOR,
                                                   FLORA_COLOR, HIGHLIGHT_COLOR, NOOP, NUM_ACTIONS, PANEL_COLOR, ROCK_COLOR,
                                                   STEP_LEFT, STEP_RIGHT, TURN_LEFT, TURN_RIGHT)

TILE_SIZE = 16
PANEL_WIDTH = 240


def raylib_color(rgb) -> pr.Color:
    return pr.Color(int(rgb[0]), int(rgb[1]), int(rgb[2]), 255)


PANEL = raylib_color(PANEL_COLOR)
GROUND = raylib_color(GROUND_COLOR)
ROCK = raylib_color(ROCK_COLOR)
FLORA = raylib_color(FLORA_COLOR)
AGENT = raylib_color(AGENT_COLOR)
HIGHLIGHT = raylib_color(HIGHLIGHT_COLOR)
TITLE_TEXT = raylib_color(GROUND_COLOR)
BODY_TEXT = pr.Color(196, 180, 150, 255)
HINT_TEXT = pr.Color(140, 126, 106, 255)
CONTROL_KEYS = {
    pr.KeyboardKey.KEY_W: FORWARD,
    pr.KeyboardKey.KEY_S: BACKWARD,
    pr.KeyboardKey.KEY_A: STEP_LEFT,
    pr.KeyboardKey.KEY_D: STEP_RIGHT,
    pr.KeyboardKey.KEY_Q: TURN_LEFT,
    pr.KeyboardKey.KEY_E: TURN_RIGHT,
}
HELP_LINES = (
    "w/s  forward, back",
    "a/d  step left, right",
    "q/e  turn left, right",
    "n  wait one step",
    "space  autoplay",
    "+/-  speed   r  reset",
    "esc  quit",
)


class Viewer:
    def __init__(self, args):
        self.env = DuneSea(
            width=args.width,
            height=args.height,
            num_agents=args.agents,
            time_limit=args.time_limit,
            rock_fill=args.rock_fill,
            smooth_passes=args.smooth_passes,
            flora_fill=args.flora_fill,
        )
        self.scale = args.scale
        self.steps_per_second = args.fps
        self.autoplay = False
        self.clock = 0.0
        self.key = jax.random.key(args.seed)
        self.init_env = jax.jit(self.env.init)
        self.step_env = jax.jit(self.env.step)
        self.paint_state = jax.jit(lambda state: self.env.render(state, TILE_SIZE))
        self.reset_world()

    def next_key(self) -> jax.Array:
        self.key, used_key = jax.random.split(self.key)
        return used_key

    def reset_world(self):
        self.state, self.timestep = self.init_env(self.next_key())
        self.episode_steps = 0

    def advance_world(self, controlled_action: int | None):
        actions = jax.random.randint(self.next_key(), (self.env.num_agents,), 0, NUM_ACTIONS)
        if controlled_action is not None:
            actions = actions.at[0].set(controlled_action)
        self.state, self.timestep = self.step_env(self.next_key(), self.state, actions)
        self.episode_steps += 1
        if bool(self.timestep.truncated.all()):
            self.reset_world()

    def handle_keys(self):
        for key, action in CONTROL_KEYS.items():
            if pr.is_key_pressed(key):
                self.advance_world(action)
        if pr.is_key_pressed(pr.KeyboardKey.KEY_N):
            self.advance_world(NOOP)
        if pr.is_key_pressed(pr.KeyboardKey.KEY_SPACE):
            self.autoplay = not self.autoplay
        if pr.is_key_pressed(pr.KeyboardKey.KEY_R):
            self.reset_world()
        if pr.is_key_pressed(pr.KeyboardKey.KEY_EQUAL) or pr.is_key_pressed(pr.KeyboardKey.KEY_KP_ADD):
            self.steps_per_second = min(self.steps_per_second * 2, 240)
        if pr.is_key_pressed(pr.KeyboardKey.KEY_MINUS) or pr.is_key_pressed(pr.KeyboardKey.KEY_KP_SUBTRACT):
            self.steps_per_second = max(self.steps_per_second / 2, 0.5)

    def tick_clock(self):
        if not self.autoplay:
            return
        self.clock += pr.get_frame_time()
        interval = 1.0 / self.steps_per_second
        while self.clock >= interval:
            self.clock -= interval
            self.advance_world(None)

    def draw_view(self, left: int, top: int, size: int):
        view = np.asarray(self.timestep.obs[0])
        cell = size // view.shape[0]
        for row in range(view.shape[0]):
            for col in range(view.shape[1]):
                occupied, rocky, leafy = view[row, col] > 0
                color = AGENT if occupied else ROCK if rocky else FLORA if leafy else GROUND
                pr.draw_rectangle(left + col * cell, top + row * cell, cell - 1, cell - 1, color)
        centre = view.shape[0] // 2
        pr.draw_rectangle_lines(left + centre * cell, top + centre * cell, cell - 1, cell - 1, HIGHLIGHT)

    def draw_frame(self, texture):
        frame = np.asarray(self.paint_state(self.state))
        alpha = np.full(frame.shape[:2] + (1,), 255, np.uint8)
        pixels = np.ascontiguousarray(np.concatenate([frame, alpha], axis=-1))
        pr.update_texture(texture, pr.ffi.cast("void *", pixels.ctypes.data))

        world_width = frame.shape[1] * self.scale
        pr.begin_drawing()
        pr.clear_background(PANEL)
        pr.draw_texture_ex(texture, pr.Vector2(0, 0), 0.0, float(self.scale), pr.WHITE)
        controlled = self.state.position[0]
        pixel = TILE_SIZE * self.scale
        pr.draw_rectangle_lines_ex(
            pr.Rectangle(int(controlled[0]) * pixel, int(controlled[1]) * pixel, pixel, pixel),
            2,
            HIGHLIGHT,
        )
        left = world_width + 16
        mode = f"autoplay {self.steps_per_second:g}/s" if self.autoplay else "manual"
        pr.draw_text("dune_sea", left, 12, 22, TITLE_TEXT)
        pr.draw_text(f"step {self.episode_steps}/{self.env.time_limit()}", left, 42, 16, BODY_TEXT)
        pr.draw_text(mode, left, 62, 16, BODY_TEXT)
        pr.draw_text("agent 0 view", left, 92, 14, HINT_TEXT)
        self.draw_view(left, 110, PANEL_WIDTH - 32)
        for index, line in enumerate(HELP_LINES):
            pr.draw_text(line, left, 340 + 18 * index, 14, HINT_TEXT)
        pr.end_drawing()

    def start_viewer(self):
        frame = np.asarray(self.paint_state(self.state))
        width = frame.shape[1] * self.scale + PANEL_WIDTH
        height = max(frame.shape[0] * self.scale, 480)
        pr.set_trace_log_level(pr.TraceLogLevel.LOG_WARNING)
        pr.init_window(width, height, "dune_sea")
        pr.set_target_fps(60)
        image = pr.gen_image_color(frame.shape[1], frame.shape[0], pr.BLACK)
        texture = pr.load_texture_from_image(image)
        pr.unload_image(image)
        pr.set_texture_filter(texture, pr.TextureFilter.TEXTURE_FILTER_POINT)
        while not pr.window_should_close():
            self.handle_keys()
            self.tick_clock()
            self.draw_frame(texture)
        pr.unload_texture(texture)
        pr.close_window()


def main():
    parser = argparse.ArgumentParser(description="Walk agent 0 around the dune_sea grid world.")
    parser.add_argument("--width", type=int, default=56)
    parser.add_argument("--height", type=int, default=36)
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--time-limit", type=int, default=256)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--rock-fill", type=float, default=0.45)
    parser.add_argument("--smooth-passes", type=int, default=4)
    parser.add_argument("--flora-fill", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    Viewer(parser.parse_args()).start_viewer()


if __name__ == "__main__":
    main()
