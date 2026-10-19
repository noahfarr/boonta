import colorsys

import jax
import jax.numpy as jnp
import numpy as np

from boonta.utils import Array, Key

NEIGHBOURHOOD_SIZE = 9
EPSILON = 1e-6


def gather_neighbourhood(field: Array, num_symbols: int) -> Array:
    one_hot = jax.nn.one_hot(field, num_symbols)
    shifted = [
        jnp.roll(one_hot, (-dy, -dx), axis=(0, 1)) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
    ]
    return jnp.concatenate(shifted, axis=-1)


class NCA:
    def __init__(
        self,
        num_symbols: int = 4,
        hidden_size: int = 64,
        temperature: float = 0.0,
        init_gain: float = 3.0,
    ):
        if num_symbols < 2:
            raise ValueError("dune_sea: num_symbols must be at least 2")
        if temperature < 0.0:
            raise ValueError("dune_sea: temperature must be non-negative")
        self.num_symbols = num_symbols
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.init_gain = init_gain
        self.num_inputs = NEIGHBOURHOOD_SIZE * num_symbols

    def draw_rule(self, key: Key) -> dict:
        w1_key, b1_key, w2_key, b2_key = jax.random.split(key, 4)
        input_scale = self.init_gain / np.sqrt(NEIGHBOURHOOD_SIZE)
        hidden_scale = self.init_gain / np.sqrt(self.hidden_size)
        return {
            "w1": jax.random.normal(w1_key, (self.num_inputs, self.hidden_size)) * input_scale,
            "b1": jax.random.normal(b1_key, (self.hidden_size,)),
            "w2": jax.random.normal(w2_key, (self.hidden_size, self.num_symbols)) * hidden_scale,
            "b2": jax.random.normal(b2_key, (self.num_symbols,)),
        }

    def seed_field(self, key: Key, height: int, width: int) -> Array:
        return jax.random.randint(key, (height, width), 0, self.num_symbols)

    def symbol_logits(self, rule: dict, field: Array) -> Array:
        senses = gather_neighbourhood(field, self.num_symbols)
        hidden = jnp.tanh(senses @ rule["w1"] + rule["b1"])
        return hidden @ rule["w2"] + rule["b2"]

    def evolve_field(self, rule: dict, field: Array, key: Key) -> Array:
        logits = self.symbol_logits(rule, field)
        if self.temperature == 0.0:
            return jnp.argmax(logits, axis=-1).astype(field.dtype)
        return jax.random.categorical(key, logits / self.temperature).astype(field.dtype)

    def run_rollout(self, rule: dict, field: Array, key: Key, num_steps: int) -> Array:
        def advance(field, step_key):
            field = self.evolve_field(rule, field, step_key)
            return field, field

        _, trajectory = jax.lax.scan(advance, field, jax.random.split(key, num_steps))
        return trajectory


def symbol_palette(num_symbols: int) -> np.ndarray:
    hues = [(index * 0.61803398875) % 1.0 for index in range(num_symbols)]
    shades = [0.55 + 0.4 * ((index * 3) % 4) / 3 for index in range(num_symbols)]
    colors = [colorsys.hsv_to_rgb(hue, 0.65, shade) for hue, shade in zip(hues, shades)]
    return (np.asarray(colors) * 255).astype(np.uint8)


def render_field(field: Array, num_symbols: int, scale: int = 1) -> Array:
    frame = jnp.asarray(symbol_palette(num_symbols))[field]
    return jnp.repeat(jnp.repeat(frame, scale, axis=-3), scale, axis=-2)


def measure_liveliness(trajectory: Array, num_symbols: int) -> dict:
    tail = trajectory[trajectory.shape[0] * 3 // 4 :]
    activity = (tail[1:] != tail[:-1]).mean()
    final = tail[-1]
    counts = jax.nn.one_hot(final, num_symbols).mean(axis=(0, 1))
    diversity = -(counts * jnp.log(counts + EPSILON)).sum() / np.log(num_symbols)
    chance = (counts**2).sum()
    matching = 0.5 * ((final == jnp.roll(final, 1, axis=0)).mean() + (final == jnp.roll(final, 1, axis=1)).mean())
    structure = (matching - chance) / (1.0 - chance + EPSILON)
    return {"activity": activity, "diversity": diversity, "structure": structure}
