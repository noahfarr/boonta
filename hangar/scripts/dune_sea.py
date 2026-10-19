import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pyray as pr  # noqa: E402

from boonta.environments.dune_sea import DuneSea  # noqa: E402
from boonta.environments.dune_sea.dune_sea import (  # noqa: E402
    BACKWARD,
    FORWARD,
    HIGHLIGHT_COLOR,
    NOOP,
    NUM_ACTIONS,
    STEP_LEFT,
    STEP_RIGHT,
    TURN_LEFT,
    TURN_RIGHT,
)

TILE_SIZE = 16
MIN_ZOOM = 0.25
MAX_ZOOM = 8.0
ZOOM_STEP = 1.15
PAN_BUTTONS = (
    pr.MouseButton.MOUSE_BUTTON_LEFT,
    pr.MouseButton.MOUSE_BUTTON_RIGHT,
    pr.MouseButton.MOUSE_BUTTON_MIDDLE,
)
HIGHLIGHT = pr.Color(*(int(channel) for channel in HIGHLIGHT_COLOR), 255)
CONTROL_KEYS = {
    pr.KeyboardKey.KEY_W: FORWARD,
    pr.KeyboardKey.KEY_S: BACKWARD,
    pr.KeyboardKey.KEY_A: STEP_LEFT,
    pr.KeyboardKey.KEY_D: STEP_RIGHT,
    pr.KeyboardKey.KEY_Q: TURN_LEFT,
    pr.KeyboardKey.KEY_E: TURN_RIGHT,
}
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
            oasis_density=args.oasis_density,
        )
        self.window_size = (args.window_width, args.window_height)
        self.start_zoom = args.zoom
        self.frame_dirty = True
        self.steps_per_second = args.fps
        self.autoplay = False
        self.clock = 0.0
        self.key = jax.random.key(args.seed)
        self.init_env = jax.jit(self.env.init)
        self.step_env = jax.jit(self.env.step)
        self.paint_state = jax.jit(lambda state: self.env.render(state, TILE_SIZE))
        self.camera = pr.Camera2D(pr.Vector2(0, 0), pr.Vector2(0, 0), 0.0, float(args.zoom))
        self.reset_world()

    def next_key(self) -> jax.Array:
        self.key, used_key = jax.random.split(self.key)
        return used_key

    def reset_world(self):
        self.state, self.timestep = self.init_env(self.next_key())
        self.episode_steps = 0
        self.frame_dirty = True

    def advance_world(self, controlled_action: int | None):
        actions = jax.random.randint(self.next_key(), (self.env.num_agents,), 0, NUM_ACTIONS)
        if controlled_action is not None:
            actions = actions.at[0].set(controlled_action)
        self.state, self.timestep = self.step_env(self.next_key(), self.state, actions)
        self.episode_steps += 1
        self.frame_dirty = True
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
        if pr.is_key_pressed(pr.KeyboardKey.KEY_F):
            self.fit_camera()
        if pr.is_key_pressed(pr.KeyboardKey.KEY_C):
            self.follow_agent()
        if pr.is_key_pressed(pr.KeyboardKey.KEY_EQUAL) or pr.is_key_pressed(pr.KeyboardKey.KEY_KP_ADD):
            self.steps_per_second = min(self.steps_per_second * 2, 240)
        if pr.is_key_pressed(pr.KeyboardKey.KEY_MINUS) or pr.is_key_pressed(pr.KeyboardKey.KEY_KP_SUBTRACT):
            self.steps_per_second = max(self.steps_per_second / 2, 0.5)

    def fit_camera(self):
        world_width = self.env.width * TILE_SIZE
        world_height = self.env.height * TILE_SIZE
        screen_width, screen_height = pr.get_screen_width(), pr.get_screen_height()
        self.camera.zoom = min(screen_width / world_width, screen_height / world_height)
        self.camera.offset.x, self.camera.offset.y = screen_width / 2, screen_height / 2
        self.camera.target.x, self.camera.target.y = world_width / 2, world_height / 2

    def follow_agent(self):
        self.camera.zoom = self.start_zoom
        self.camera.offset.x, self.camera.offset.y = pr.get_screen_width() / 2, pr.get_screen_height() / 2
        x, y = (int(value) for value in self.state.position[0])
        self.camera.target.x, self.camera.target.y = (x + 0.5) * TILE_SIZE, (y + 0.5) * TILE_SIZE

    def handle_mouse(self):
        if any(pr.is_mouse_button_down(button) for button in PAN_BUTTONS):
            delta = pr.get_mouse_delta()
            self.camera.target.x -= delta.x / self.camera.zoom
            self.camera.target.y -= delta.y / self.camera.zoom
        wheel = pr.get_mouse_wheel_move()
        if wheel:
            mouse = pr.get_mouse_position()
            anchor = pr.get_screen_to_world_2d(mouse, self.camera)
            self.camera.offset.x, self.camera.offset.y = mouse.x, mouse.y
            self.camera.target.x, self.camera.target.y = anchor.x, anchor.y
            zoom = self.camera.zoom * ZOOM_STEP**wheel
            self.camera.zoom = min(max(zoom, MIN_ZOOM), MAX_ZOOM)

    def tick_clock(self):
        if not self.autoplay:
            return
        self.clock += pr.get_frame_time()
        interval = 1.0 / self.steps_per_second
        while self.clock >= interval:
            self.clock -= interval
            self.advance_world(None)

    def visible_copies(self) -> list[tuple[int, int]]:
        world_width = self.env.width * TILE_SIZE
        world_height = self.env.height * TILE_SIZE
        corner = pr.get_screen_to_world_2d(pr.Vector2(0, 0), self.camera)
        far = pr.get_screen_to_world_2d(
            pr.Vector2(pr.get_screen_width(), pr.get_screen_height()), self.camera
        )
        columns = range(int(np.floor(corner.x / world_width)), int(np.floor(far.x / world_width)) + 1)
        rows = range(int(np.floor(corner.y / world_height)), int(np.floor(far.y / world_height)) + 1)
        return [(column * world_width, row * world_height) for row in rows for column in columns]

    def upload_frame(self, texture):
        frame = np.asarray(self.paint_state(self.state))
        alpha = np.full(frame.shape[:2] + (1,), 255, np.uint8)
        pixels = np.ascontiguousarray(np.concatenate([frame, alpha], axis=-1))
        pr.update_texture(texture, pr.ffi.cast("void *", pixels.ctypes.data))
        self.frame_dirty = False

    def draw_frame(self, texture):
        if self.frame_dirty:
            self.upload_frame(texture)

        pr.begin_drawing()
        pr.clear_background(pr.BLACK)
        pr.begin_mode_2d(self.camera)
        controlled = self.state.position[0]
        for left, top in self.visible_copies():
            pr.draw_texture_ex(texture, pr.Vector2(left, top), 0.0, 1.0, pr.WHITE)
            pr.draw_rectangle_lines_ex(
                pr.Rectangle(
                    left + int(controlled[0]) * TILE_SIZE,
                    top + int(controlled[1]) * TILE_SIZE,
                    TILE_SIZE,
                    TILE_SIZE,
                ),
                max(2.0 / self.camera.zoom, 1.0),
                HIGHLIGHT,
            )
        pr.end_mode_2d()
        pr.end_drawing()

    def start_viewer(self):
        frame = np.asarray(self.paint_state(self.state))
        pr.set_trace_log_level(pr.TraceLogLevel.LOG_WARNING)
        pr.set_config_flags(pr.ConfigFlags.FLAG_WINDOW_RESIZABLE)
        pr.init_window(*self.window_size, "dune_sea")
        pr.set_target_fps(60)
        image = pr.gen_image_color(frame.shape[1], frame.shape[0], pr.BLACK)
        texture = pr.load_texture_from_image(image)
        pr.unload_image(image)
        pr.set_texture_filter(texture, pr.TextureFilter.TEXTURE_FILTER_POINT)
        self.follow_agent()
        while not pr.window_should_close():
            self.handle_mouse()
            self.handle_keys()
            self.tick_clock()
            self.draw_frame(texture)
        pr.unload_texture(texture)
        pr.close_window()


def main():
    parser = argparse.ArgumentParser(
        description="Walk agent 0 around the dune_sea grid world. "
        "w/s forward/back, a/d step left/right, q/e turn, n wait, space autoplay, "
        "+/- speed, r reset, esc quit. Drag to pan, scroll to zoom, f to fit the map, c to centre on agent 0."
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--time-limit", type=int, default=1024)
    parser.add_argument("--zoom", type=float, default=2.0)
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=800)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--rock-fill", type=float, default=0.39)
    parser.add_argument("--smooth-passes", type=int, default=4)
    parser.add_argument("--flora-fill", type=float, default=0.03)
    parser.add_argument("--oasis-density", type=float, default=0.0012)
    parser.add_argument("--seed", type=int, default=0)
    Viewer(parser.parse_args()).start_viewer()


if __name__ == "__main__":
    main()
