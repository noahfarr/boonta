import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pyray as pr  # noqa: E402

from boonta import environments  # noqa: E402
from boonta.environments.dune_sea.dune_sea import (  # noqa: E402
    BACKWARD,
    BROADCAST,
    FORWARD,
    INTERACT,
    LEFT,
    NOOP,
    PLACE,
    NUM_ACTIONS,
    RIGHT,
    TURN_LEFT,
    TURN_RIGHT,
)

TILE_SIZE = 16
STEP_ANIMATION_SECONDS = 0.12
MIN_ZOOM = 0.25
MAX_ZOOM = 8.0
ZOOM_STEP = 1.15
PAN_BUTTONS = (
    pr.MouseButton.MOUSE_BUTTON_LEFT,
    pr.MouseButton.MOUSE_BUTTON_RIGHT,
    pr.MouseButton.MOUSE_BUTTON_MIDDLE,
)
HIGHLIGHT = pr.Color(231, 111, 81, 255)
CONTROL_KEYS = {
    pr.KeyboardKey.KEY_W: FORWARD,
    pr.KeyboardKey.KEY_S: BACKWARD,
    pr.KeyboardKey.KEY_A: LEFT,
    pr.KeyboardKey.KEY_D: RIGHT,
    pr.KeyboardKey.KEY_Q: TURN_LEFT,
    pr.KeyboardKey.KEY_E: TURN_RIGHT,
    pr.KeyboardKey.KEY_X: INTERACT,
    pr.KeyboardKey.KEY_B: BROADCAST,
    pr.KeyboardKey.KEY_N: NOOP,
    pr.KeyboardKey.KEY_P: PLACE,
}


class Viewer:
    def __init__(self, args):
        self.env = environments.make(
            "dune_sea",
            args.env,
            kwargs={
                "width": args.width,
                "height": args.height,
                "time_limit": args.time_limit,
                **(
                    {"num_hiders": args.agents // 2, "num_seekers": args.agents - args.agents // 2}
                    if args.env == "hide_and_seek"
                    else {"num_agents": args.agents}
                ),
            },
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
        self.show_fov = False
        self.previous = None
        self.moved_at = 0.0
        self.paint_plain = jax.jit(lambda state, previous, blend: self.env.render(state, 1, previous=previous, blend=blend))
        self.paint_fov = jax.jit(
            lambda state, previous, blend: self.env.render(state, 1, show_fov=True, previous=previous, blend=blend)
        )
        self.camera = pr.Camera2D(pr.Vector2(0, 0), pr.Vector2(0, 0), 0.0, float(args.zoom))
        self.reset_world()

    def blend(self) -> float:
        if self.previous is None:
            return 1.0
        return min((pr.get_time() - self.moved_at) / STEP_ANIMATION_SECONDS, 1.0)

    def paint_state(self, state):
        previous = state if self.previous is None else self.previous
        return (self.paint_fov if self.show_fov else self.paint_plain)(state, previous, self.blend())

    def next_key(self) -> jax.Array:
        self.key, used_key = jax.random.split(self.key)
        return used_key

    def reset_world(self):
        self.previous = None
        self.state, self.timestep = self.init_env(self.next_key())
        self.episode_steps = 0
        self.frame_dirty = True

    def advance_world(self, controlled_action: int | None):
        self.previous = self.state
        self.moved_at = pr.get_time()
        actions = jax.random.randint(self.next_key(), (self.env.num_agents,), 0, NUM_ACTIONS)
        if controlled_action is not None:
            actions = actions.at[0].set(controlled_action)
        self.state, self.timestep = self.step_env(self.next_key(), self.state, actions)
        self.episode_steps += 1
        self.frame_dirty = True
        if bool(self.timestep.truncated.all() | self.timestep.terminated.all()):
            self.reset_world()

    def handle_keys(self):
        for key, action in CONTROL_KEYS.items():
            if pr.is_key_pressed(key):
                self.advance_world(action)
        if pr.is_key_pressed(pr.KeyboardKey.KEY_SPACE):
            self.autoplay = not self.autoplay
        if pr.is_key_pressed(pr.KeyboardKey.KEY_R):
            self.reset_world()
        if pr.is_key_pressed(pr.KeyboardKey.KEY_V):
            self.show_fov = not self.show_fov
            self.frame_dirty = True
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
        y, x = (int(value) for value in self.state.agent.position[0])
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

    def upload_frame(self, texture):
        frame = np.asarray(self.paint_state(self.state))
        alpha = np.full(frame.shape[:2] + (1,), 255, np.uint8)
        pixels = np.ascontiguousarray(np.concatenate([frame, alpha], axis=-1))
        pr.update_texture(texture, pr.ffi.cast("void *", pixels.ctypes.data))
        self.frame_dirty = False

    def draw_frame(self, texture):
        if self.frame_dirty or self.blend() < 1.0:
            self.upload_frame(texture)

        pr.begin_drawing()
        pr.clear_background(pr.BLACK)
        pr.begin_mode_2d(self.camera)
        y, x = (int(value) for value in self.state.agent.position[0])
        pr.draw_texture_ex(texture, pr.Vector2(0, 0), 0.0, 1.0, pr.WHITE)
        pr.draw_rectangle_lines_ex(
            pr.Rectangle(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE),
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
        description="Walk agent 0 around a dune_sea grid world. "
        "w/s forward/back, a/d step left/right, q/e turn, x interact, b broadcast, p place, n wait, v show field of view, "
        "space autoplay, +/- speed, r reset, esc quit. "
        "Drag to pan, scroll to zoom, f to fit the map, c to centre on agent 0."
    )
    parser.add_argument("--env", choices=("dune_sea", "hide_and_seek", "tag", "oasis"), default="hide_and_seek")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--time-limit", type=int, default=1024)
    parser.add_argument("--zoom", type=float, default=2.0)
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=800)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    Viewer(parser.parse_args()).start_viewer()


if __name__ == "__main__":
    main()
