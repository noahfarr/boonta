import argparse

import jax
import jax.numpy as jnp
import numpy as np
import pyray as pr

from boonta.environments.dune_sea import NCA, symbol_palette

HELP_TEXT = "space pause  right step  n/p rule  r reseed  +/- speed  q quit"
PANEL_HEIGHT = 48


class Viewer:
    def __init__(self, args):
        self.nca = NCA(
            num_symbols=args.symbols,
            hidden_size=args.hidden_size,
            temperature=args.temperature,
            init_gain=args.init_gain,
        )
        self.size = args.size
        self.scale = args.scale
        self.rule_seed = args.rule
        self.field_seed = 0
        self.steps_per_second = args.fps
        self.paused = False
        self.clock = 0.0
        palette = symbol_palette(args.symbols)
        alpha = np.full((args.symbols, 1), 255, np.uint8)
        self.palette = jnp.asarray(np.concatenate([palette, alpha], axis=1))
        self.evolve_step = jax.jit(self.nca.evolve_field)
        self.paint_field = jax.jit(lambda field: self.palette[field])
        self.load_rule()

    def load_rule(self):
        rule_key, self.field_key, self.first_step_key = jax.random.split(
            jax.random.key(self.rule_seed), 3
        )
        self.rule = self.nca.draw_rule(rule_key)
        self.field_seed = 0
        self.reseed_field()

    def reseed_field(self):
        field_key, self.step_key = self.field_key, self.first_step_key
        if self.field_seed:
            field_key = jax.random.fold_in(field_key, self.field_seed)
            self.step_key = jax.random.fold_in(self.step_key, self.field_seed)
        self.field = self.nca.seed_field(field_key, self.size, self.size)
        self.step_count = 0
        self.changed = 1.0

    def advance_field(self):
        self.step_key, evolve_key = jax.random.split(self.step_key)
        field = self.evolve_step(self.rule, self.field, evolve_key)
        self.changed = float(jnp.mean(field != self.field))
        self.field = field
        self.step_count += 1

    def handle_keys(self):
        if pr.is_key_pressed(pr.KeyboardKey.KEY_SPACE):
            self.paused = not self.paused
        if pr.is_key_pressed(pr.KeyboardKey.KEY_RIGHT) and self.paused:
            self.advance_field()
        if pr.is_key_pressed(pr.KeyboardKey.KEY_N):
            self.rule_seed += 1
            self.load_rule()
        if pr.is_key_pressed(pr.KeyboardKey.KEY_P):
            self.rule_seed = max(self.rule_seed - 1, 0)
            self.load_rule()
        if pr.is_key_pressed(pr.KeyboardKey.KEY_R):
            self.field_seed += 1
            self.reseed_field()
        if pr.is_key_pressed(pr.KeyboardKey.KEY_EQUAL) or pr.is_key_pressed(pr.KeyboardKey.KEY_KP_ADD):
            self.steps_per_second = min(self.steps_per_second * 2, 960)
        if pr.is_key_pressed(pr.KeyboardKey.KEY_MINUS) or pr.is_key_pressed(pr.KeyboardKey.KEY_KP_SUBTRACT):
            self.steps_per_second = max(self.steps_per_second / 2, 0.5)

    def tick_clock(self):
        if self.paused:
            return
        self.clock += pr.get_frame_time()
        interval = 1.0 / self.steps_per_second
        while self.clock >= interval:
            self.clock -= interval
            self.advance_field()

    def draw_frame(self, texture):
        pixels = np.ascontiguousarray(np.asarray(self.paint_field(self.field)))
        pr.update_texture(texture, pr.ffi.cast("void *", pixels.ctypes.data))
        pr.begin_drawing()
        pr.clear_background(pr.BLACK)
        pr.draw_texture_ex(texture, pr.Vector2(0, 0), 0.0, float(self.scale), pr.WHITE)
        state = "paused" if self.paused else f"{self.steps_per_second:g} steps/s"
        status = (
            f"rule {self.rule_seed}   step {self.step_count}   "
            f"changed {100 * self.changed:5.1f}%   {state}"
        )
        top = self.size * self.scale
        pr.draw_text(status, 8, top + 6, 18, pr.RAYWHITE)
        pr.draw_text(HELP_TEXT, 8, top + 28, 14, pr.GRAY)
        pr.end_drawing()

    def start_viewer(self):
        width = self.size * self.scale
        pr.set_trace_log_level(pr.TraceLogLevel.LOG_WARNING)
        pr.init_window(width, width + PANEL_HEIGHT, "dune_sea")
        pr.set_target_fps(60)
        image = pr.gen_image_color(self.size, self.size, pr.BLACK)
        texture = pr.load_texture_from_image(image)
        pr.unload_image(image)
        pr.set_texture_filter(texture, pr.TextureFilter.TEXTURE_FILTER_POINT)
        while not pr.window_should_close() and not pr.is_key_pressed(pr.KeyboardKey.KEY_Q):
            self.handle_keys()
            self.tick_clock()
            self.draw_frame(texture)
        pr.unload_texture(texture)
        pr.close_window()


def main():
    parser = argparse.ArgumentParser(description="Watch a dune_sea NCA rule evolve, one state per frame.")
    parser.add_argument("--rule", type=int, default=0)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=5)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--symbols", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--init-gain", type=float, default=3.0)
    Viewer(parser.parse_args()).start_viewer()


if __name__ == "__main__":
    main()
