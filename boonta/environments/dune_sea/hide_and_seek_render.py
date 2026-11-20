import math

import jax
import jax.numpy as jnp
from jaxgl.maths import signed_line_distance
from jaxgl.renderer import clear_screen, make_renderer

from .hide_and_seek import (
    AGENT_RADIUS,
    BOUNDARY,
    GROUND,
    HIDER,
    RAMP_HEIGHT,
    RAMP_WIDTH,
    SEEKER,
    UNLOCKED,
    WALL_WIDTH,
)

BACKGROUND = jnp.array([248.0, 250.0, 252.0])
FIXED = jnp.array([88.0, 101.0, 119.0])
MOVABLE = jnp.array([187.0, 198.0, 211.0])
OUTLINE = jnp.array([34.0, 45.0, 60.0])
JOINT = jnp.array([255.0, 211.0, 97.0])
HIDER_BLUE = (45, 113, 200)
SEEKER_RED = (222, 84, 77)
TEAMS = jnp.array([HIDER_BLUE, SEEKER_RED], dtype=jnp.float32)
SIGHT_ALPHA = 0.60
OUTLINE_WIDTH = 1.5
SIGHT_WIDTH = 1.75
JOINT_RADIUS = 0.06


def polygon_shader(edge):
    def shade(position, current, _unit, uniform):
        starts, ends, valid, colour, outline = uniform
        lengths = jnp.maximum(jnp.linalg.norm(ends - starts, axis=-1), 1e-6)
        sides = (
            jax.vmap(signed_line_distance, (None, 0, 0))(position, starts, ends)
            / lengths
        )
        inside = ((sides <= 0) | ~valid).all() | ((sides >= 0) | ~valid).all()
        near = ((jnp.abs(sides) <= edge) & valid).any()
        fragment = jax.lax.select(near, outline, colour)
        return jax.lax.select(inside & valid.any(), fragment, current)

    return shade


def agent_shader(edge, joint, radius):
    def shade(position, current, _unit, uniform):
        centre, facing, colour, holding, active = uniform
        offset = position - centre
        distance = jnp.linalg.norm(offset)
        rim = distance > radius - edge
        gaze = (jnp.abs(offset[1]) <= edge / 2) & (offset[0] * facing > radius * 0.35)
        fragment = jax.lax.select(rim | gaze, OUTLINE, colour)
        fragment = jax.lax.select(
            holding & (distance < joint + edge / 2), OUTLINE, fragment
        )
        fragment = jax.lax.select(holding & (distance < joint), JOINT, fragment)
        return jax.lax.select((distance <= radius) & active, fragment, current)

    return shade


def line_shader(width):
    def shade(position, current, _unit, uniform):
        start, end, seen = uniform
        along = end - start
        length_squared = jnp.maximum(jnp.sum(along * along, axis=-1), 1e-6)
        fraction = jnp.clip(
            jnp.sum((position - start) * along, axis=-1) / length_squared, 0.0, 1.0
        )
        offset = position - (start + fraction[:, None] * along)
        near = (jnp.sum(offset * offset, axis=-1) <= (width / 2) ** 2) & seen
        blended = SIGHT_ALPHA * TEAMS[SEEKER] + (1.0 - SIGHT_ALPHA) * current
        return jax.lax.select(near.any(), blended, current)

    return shade


def make_render(env, scale: int, height: float, supersample: int, sight: bool):
    columns, rows = int(env.width * scale), int((GROUND + height) * scale)
    if scale < 1 or supersample < 1 or columns < 1 or rows < 1:
        raise ValueError("Render dimensions and supersampling must be positive")

    density = scale * supersample
    edge = OUTLINE_WIDTH * supersample
    # The padding keeps edge patches intact when bodies cross the viewport.
    pad = density
    visible = (columns * supersample, rows * supersample)
    screen = (visible[0] + 2 * pad, visible[1] + 2 * pad)
    cleared = clear_screen(screen, BACKGROUND)
    heights = (jnp.arange(screen[1]) + 0.5 - pad) / density
    floor = (heights < GROUND)[None, :, None]
    rim = ((heights >= GROUND - edge / density) & (heights < GROUND))[None, :, None]
    canvas = jnp.where(rim, OUTLINE, jnp.where(floor, FIXED, cleared))

    def pixel(point):
        return point * density + pad - 0.5

    wall_patch = (math.ceil(WALL_WIDTH * density) + 2, visible[1])
    object_side = math.ceil(math.hypot(RAMP_WIDTH, RAMP_HEIGHT) * density) + 2
    agent_side = math.ceil(2 * AGENT_RADIUS * density) + 2
    walls = make_renderer(screen, polygon_shader(edge), wall_patch, batched=True)
    objects = make_renderer(
        screen, polygon_shader(edge), (object_side, object_side), batched=True
    )
    agents = make_renderer(
        screen,
        agent_shader(edge, JOINT_RADIUS * density, AGENT_RADIUS * density),
        (agent_side, agent_side),
        batched=True,
    )
    draw_sight = sight and env.num_seekers and env.num_hiders
    if draw_sight:
        lines = make_renderer(screen, line_shader(SIGHT_WIDTH * supersample), visible)
        line_origin = jnp.array([pad, pad], dtype=jnp.int32)
        seekers = jnp.flatnonzero(env.team == SEEKER, size=env.num_seekers)
        hiders = jnp.flatnonzero(env.team == HIDER, size=env.num_hiders)

    @jax.jit
    def render(state):
        sim = state.sim
        starts, ends, valid = env.edges(sim)
        starts, ends = pixel(starts), pixel(ends)
        pixels = canvas

        if env.num_walls:
            index = slice(BOUNDARY, env.first_object)
            wall_x = pixel(sim.polygon.position[index, 0]) - wall_patch[0] / 2
            origin = jnp.stack([wall_x, jnp.full(env.num_walls, pad)], axis=-1)
            pixels = walls(
                pixels,
                jnp.floor(origin).astype(jnp.int32),
                (
                    starts[index],
                    ends[index],
                    valid[index],
                    jnp.broadcast_to(FIXED, (env.num_walls, 3)),
                    jnp.broadcast_to(OUTLINE, (env.num_walls, 3)),
                ),
            )

        if env.num_objects:
            index = slice(env.first_object, env.num_polygons)
            locked = state.locked_by != UNLOCKED
            team = jnp.maximum(state.locked_by, 0)
            colour = jnp.where(locked[:, None], FIXED, MOVABLE)
            outline = jnp.where(locked[:, None], TEAMS[team], OUTLINE)
            origin = pixel(sim.polygon.position[index]) - object_side / 2
            pixels = objects(
                pixels,
                jnp.floor(origin).astype(jnp.int32),
                (starts[index], ends[index], valid[index], colour, outline),
            )

        centres = pixel(sim.circle.position)
        if draw_sight:
            seen = env.sight(state)[seekers[:, None], hiders[None, :]]
            if env.preparation_steps:
                seen &= state.time > env.preparation_steps
            uniforms = (
                jnp.repeat(centres[seekers], env.num_hiders, axis=0),
                jnp.tile(centres[hiders], (env.num_seekers, 1)),
                seen.ravel(),
            )
            pixels = jax.lax.cond(
                seen.any(),
                lambda image: lines(image, line_origin, uniforms),
                lambda image: image,
                pixels,
            )

        pixels = agents(
            pixels,
            jnp.floor(centres - agent_side / 2).astype(jnp.int32),
            (
                centres,
                state.facing,
                TEAMS[env.team],
                state.holding >= 0,
                sim.circle.active,
            ),
        )
        pixels = pixels[pad : pad + visible[0], pad : pad + visible[1]]
        pixels = pixels.reshape(columns, supersample, rows, supersample, 3).mean((1, 3))
        return pixels.transpose(1, 0, 2)[::-1].round().astype(jnp.uint8)

    return render
