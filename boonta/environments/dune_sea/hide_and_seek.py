from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax2d.engine import PhysicsEngine, calculate_collision_matrix, create_empty_sim
from jax2d.scene import add_circle_to_scene, add_polygon_to_scene, add_rectangle_to_scene
from jax2d.sim_state import SimParams, SimState, StaticSimParams

from boonta.utils import Array, Key, Timestep

from ..environment import Environment
from ..spaces import Space

(
    NOOP,
    LEFT,
    RIGHT,
    JUMP,
    JUMP_LEFT,
    JUMP_RIGHT,
    GRAB,
    LOCK,
    BROADCAST,
) = range(9)
NUM_ACTIONS = 9
HEADING = jnp.array([0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0], jnp.float32)
LEAPING = jnp.array([False, False, False, True, True, True, False, False, False])

HIDER, SEEKER = 0, 1
EMPTY = -1
UNLOCKED = -1

BOUNDARY = 4
FLOOR_OFFSET = 0.2
GROUND = 2 * FLOOR_OFFSET
AGENT_RADIUS = 0.25
BOX_SIZE = 0.5
RAMP_WIDTH = 1.2
RAMP_HEIGHT = 0.6
WALL_WIDTH = 0.2
MIN_WALL_HEIGHT = 1.2
MAX_WALL_HEIGHT = 2.5
MIN_DOOR_HEIGHT = 0.4
MAX_DOOR_HEIGHT = 0.5
ROOM_SLOTS = 4
INSIDE_BOXES = 2
MARGIN = 0.4
SLOT_WIDTH = 1.3
AGENT_FRICTION = 0.0
WALL_FRICTION = 0.0
OBJECT_FRICTION = 0.5
RISING = jnp.array([[RAMP_WIDTH / 2, RAMP_HEIGHT / 2], [RAMP_WIDTH / 2, -RAMP_HEIGHT / 2], [-RAMP_WIDTH / 2, -RAMP_HEIGHT / 2]])
FALLING = jnp.array([[-RAMP_WIDTH / 2, RAMP_HEIGHT / 2], [RAMP_WIDTH / 2, -RAMP_HEIGHT / 2], [-RAMP_WIDTH / 2, -RAMP_HEIGHT / 2]])

SPEED = 3.5
ACCELERATION = 40.0
JUMP_SPEED = 4.0
VISION_CONE = 135.0
REACH = 0.3
STANDING = 0.5
EPSILON = 1e-8

SELF_FEATURES = 10
AGENT_FEATURES = 8
OBJECT_FEATURES = 11

@struct.dataclass
class Layout:
    wall_x: Array
    wall_height: Array
    object_x: Array
    mirrored: Array
    agent_x: Array
    facing: Array


@struct.dataclass
class HideAndSeekState:
    sim: SimState
    facing: Array
    grounded: Array
    holding: Array
    locked_by: Array
    inverse_mass: Array
    inverse_inertia: Array
    broadcast: Array
    time: Array


Generator = Callable[[Key, float, int, int, int, int, int], Layout]


def num_slots(width: float) -> int:
    return int((width - 2 * MARGIN) // SLOT_WIDTH)


def slot_centres(width: float) -> Array:
    return MARGIN + SLOT_WIDTH * (jnp.arange(num_slots(width)) + 0.5)


def random_facing(key: Key, count: int) -> Array:
    return jnp.where(jax.random.bernoulli(key, 0.5, (count,)), 1.0, -1.0)


def random_layout(
    key: Key,
    width: float,
    num_walls: int,
    num_boxes: int,
    num_ramps: int,
    num_hiders: int,
    num_seekers: int,
    low: float = MIN_WALL_HEIGHT,
    high: float = MAX_WALL_HEIGHT,
) -> Layout:
    slot_key, height_key, mirror_key, facing_key = jax.random.split(key, 4)
    centres = slot_centres(width)
    order = centres[jax.random.permutation(slot_key, len(centres))]
    num_objects = num_boxes + num_ramps
    num_agents = num_hiders + num_seekers
    return Layout(
        wall_x=order[:num_walls],
        wall_height=jax.random.uniform(height_key, (num_walls,), minval=low, maxval=high),
        object_x=order[num_walls : num_walls + num_objects],
        mirrored=jax.random.bernoulli(mirror_key, 0.5, (num_ramps,)),
        agent_x=order[num_walls + num_objects : num_walls + num_objects + num_agents],
        facing=random_facing(facing_key, num_agents),
    )


def shelter_layout(
    key: Key,
    width: float,
    num_walls: int,
    num_boxes: int,
    num_ramps: int,
    num_hiders: int,
    num_seekers: int,
    room_slots: int = ROOM_SLOTS,
) -> Layout:
    slots = num_slots(width)
    inside_boxes = max(min(INSIDE_BOXES, num_boxes, room_slots - num_hiders), 0)
    outside = slots - room_slots - 1
    needed = num_walls - 1 + num_boxes - inside_boxes + num_ramps + num_seekers
    if num_walls < 1:
        raise ValueError("The shelter layout needs a wall for the room's door")
    if num_hiders > room_slots:
        raise ValueError(f"The shelter room fits {room_slots} hiders, not {num_hiders}")
    if needed > outside:
        raise ValueError(f"A width of {width} leaves {outside} slots outside the room, not {needed}")
    side_key, room_key, yard_key, door_key, height_key, mirror_key, facing_key = jax.random.split(key, 7)
    centres = slot_centres(width)
    order = jnp.where(jax.random.bernoulli(side_key), jnp.arange(slots), jnp.arange(slots)[::-1])
    room = centres[order[:room_slots][jax.random.permutation(room_key, room_slots)]]
    door = centres[order[room_slots]]
    yard = centres[order[room_slots + 1 :][jax.random.permutation(yard_key, outside)]]
    walls, boxes = num_walls - 1, num_boxes - inside_boxes
    return Layout(
        wall_x=jnp.concatenate([door[None], yard[:walls]]),
        wall_height=jnp.concatenate(
            [
                jax.random.uniform(door_key, (1,), minval=MIN_DOOR_HEIGHT, maxval=MAX_DOOR_HEIGHT),
                jax.random.uniform(height_key, (walls,), minval=MIN_WALL_HEIGHT, maxval=MAX_WALL_HEIGHT),
            ]
        ),
        object_x=jnp.concatenate(
            [room[num_hiders : num_hiders + inside_boxes], yard[walls : walls + boxes + num_ramps]]
        ),
        mirrored=jax.random.bernoulli(mirror_key, 0.5, (num_ramps,)),
        agent_x=jnp.concatenate(
            [room[:num_hiders], yard[walls + boxes + num_ramps : walls + boxes + num_ramps + num_seekers]]
        ),
        facing=random_facing(facing_key, num_hiders + num_seekers),
    )


LAYOUTS = {"random": random_layout, "shelter": shelter_layout}


def cross(a: Array, b: Array) -> Array:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def intersect(origin: Array, direction: Array, starts: Array, ends: Array) -> tuple[Array, Array, Array]:
    side = ends - starts
    denominator = cross(direction, side)
    parallel = jnp.abs(denominator) < EPSILON
    denominator = jnp.where(parallel, 1.0, denominator)
    offset = starts - origin
    along = cross(offset, side) / denominator
    across = cross(offset, direction) / denominator
    return along, across, ~parallel & (across >= 0.0) & (across <= 1.0)


class HideAndSeek(Environment):
    thread_safe_render = True

    def __init__(
        self,
        num_hiders: int = 2,
        num_seekers: int = 2,
        num_boxes: int = 3,
        num_ramps: int = 1,
        num_walls: int = 2,
        width: float = 16.0,
        time_limit: int = 240,
        preparation: float = 0.4,
        substeps: int = 2,
        num_rays: int = 16,
        grabbing: bool = True,
        locking: bool = True,
        communication: bool = False,
        layout: str = "shelter",
        generator: Generator | None = None,
        num_solver_iterations: int = 10,
        vision_cone: float = VISION_CONE,
        jump_speed: float = JUMP_SPEED,
    ):
        num_agents = num_hiders + num_seekers
        needed = num_walls + num_boxes + num_ramps + num_agents
        if num_agents < 2:
            raise ValueError("Hide and seek needs at least two agents, since Jax2D cannot pair a lone circle")
        if needed > num_slots(width):
            raise ValueError(f"A width of {width} fits {num_slots(width)} walls, objects and agents, not {needed}")
        if num_solver_iterations < 1:
            raise ValueError("Hide and seek needs at least one physics solver iteration")
        if generator is None and layout not in LAYOUTS:
            raise ValueError(f"Unknown layout {layout!r}, expected one of {sorted(LAYOUTS)}")
        if not 0.0 < vision_cone <= 360.0:
            raise ValueError("The vision cone must be between 0 and 360 degrees")
        self.num_hiders = num_hiders
        self.num_seekers = num_seekers
        self.num_agents = num_agents
        self.num_boxes = num_boxes
        self.num_ramps = num_ramps
        self.num_objects = num_boxes + num_ramps
        self.num_walls = num_walls
        self.width = width
        self._renderers = {}
        self._time_limit = time_limit
        self.preparation_steps = int(round(preparation * time_limit))
        self.substeps = substeps
        self.num_rays = num_rays
        self.grabbing = grabbing
        self.locking = locking
        self.communication = communication
        self.generator = generator or LAYOUTS[layout]
        self.jump_speed = jump_speed
        self.vision = float(np.cos(np.radians(vision_cone / 2)))
        self.first_object = BOUNDARY + num_walls
        self.num_polygons = self.first_object + self.num_objects
        self.static_params = StaticSimParams(
            num_polygons=self.num_polygons,
            num_circles=num_agents,
            num_joints=num_agents,
            num_thrusters=1,
            max_polygon_vertices=4,
            num_solver_iterations=num_solver_iterations,
            num_static_fixated_polys=self.first_object if self.num_objects else self.first_object - 1,
        )
        self.params = SimParams(clip_position=3 * width)
        self.engine = PhysicsEngine(self.static_params)
        self.motors = jnp.zeros(self.static_params.num_joints + self.static_params.num_thrusters)
        self.step_time = substeps * self.params.dt
        self.team = jnp.array([HIDER] * num_hiders + [SEEKER] * num_seekers, jnp.int32)
        self.is_ramp = jnp.arange(self.num_objects) >= num_boxes
        self.others = jnp.asarray(
            np.array([[other for other in range(num_agents) if other != agent] for agent in range(num_agents)], np.int32).reshape(
                num_agents, num_agents - 1
            )
        )
        angles = 2 * jnp.pi * jnp.arange(num_rays) / num_rays
        self.rays = jnp.stack([jnp.cos(angles), jnp.sin(angles)], -1)
        self.observation_size = (
            SELF_FEATURES + 2 * num_rays + AGENT_FEATURES * (num_agents - 1) + OBJECT_FEATURES * self.num_objects
        )
        jax.eval_shape(self.generate, jax.random.key(0))

    def generate(self, key: Key) -> Layout:
        return self.generator(
            key, self.width, self.num_walls, self.num_boxes, self.num_ramps, self.num_hiders, self.num_seekers
        )

    def init(self, key: Key) -> tuple[HideAndSeekState, Timestep]:
        state = jax.tree.map(lambda leaf: jnp.asarray(leaf, leaf.dtype), self.build(self.generate(key)))
        sight = self.sight(state)
        hidden, seen = self.hidden(sight)
        timestep = Timestep(
            obs=self.observations(state, sight),
            action=jnp.zeros(self.num_agents, jnp.int32),
            reward=jnp.zeros(self.num_agents, jnp.float32),
            terminated=jnp.ones(self.num_agents, bool),
            truncated=jnp.zeros(self.num_agents, bool),
            info=self.tally(state, hidden, seen, jnp.bool_(self.preparation_steps > 0)),
        )
        return state, timestep

    def build(self, layout: Layout) -> HideAndSeekState:
        sim = create_empty_sim(self.static_params, scene_size=self.width, floor_offset=FLOOR_OFFSET)
        for index in range(self.num_walls):
            height = layout.wall_height[index]
            sim, _ = add_rectangle_to_scene(
                sim,
                self.static_params,
                position=jnp.stack([layout.wall_x[index], GROUND + height / 2]),
                dimensions=jnp.stack([jnp.float32(WALL_WIDTH), height]),
                friction=WALL_FRICTION,
                fixated=True,
            )
        for index in range(self.num_boxes):
            sim, _ = add_rectangle_to_scene(
                sim,
                self.static_params,
                position=jnp.stack([layout.object_x[index], jnp.float32(GROUND + BOX_SIZE / 2)]),
                dimensions=jnp.array([BOX_SIZE, BOX_SIZE]),
                friction=OBJECT_FRICTION,
            )
        for index in range(self.num_ramps):
            sim, _ = add_polygon_to_scene(
                sim,
                self.static_params,
                position=jnp.stack([layout.object_x[self.num_boxes + index], jnp.float32(GROUND + RAMP_HEIGHT / 2)]),
                vertices=jnp.where(layout.mirrored[index], FALLING, RISING),
                n_vertices=3,
                friction=OBJECT_FRICTION,
            )
        for index in range(self.num_agents):
            sim, _ = add_circle_to_scene(
                sim,
                self.static_params,
                position=jnp.stack([layout.agent_x[index], jnp.float32(GROUND + AGENT_RADIUS)]),
                radius=AGENT_RADIUS,
                friction=AGENT_FRICTION,
            )
        sim = sim.replace(
            polygon=sim.polygon.replace(friction=sim.polygon.friction.at[1:BOUNDARY].set(WALL_FRICTION)),
            circle=sim.circle.replace(inverse_inertia=jnp.zeros_like(sim.circle.inverse_inertia)),
        )
        objects = slice(self.first_object, self.num_polygons)
        return HideAndSeekState(
            sim=sim,
            facing=jnp.asarray(layout.facing, jnp.float32),
            grounded=jnp.zeros(self.num_agents, bool),
            holding=jnp.full(self.num_agents, EMPTY, jnp.int32),
            locked_by=jnp.full(self.num_objects, UNLOCKED, jnp.int32),
            inverse_mass=sim.polygon.inverse_mass[objects],
            inverse_inertia=sim.polygon.inverse_inertia[objects],
            broadcast=jnp.zeros(self.num_agents, bool),
            time=jnp.int32(0),
        )

    def step(self, key: Key, state: HideAndSeekState, action: Array) -> tuple[HideAndSeekState, Timestep]:
        chosen = jnp.asarray(action, jnp.int32)
        preparing = state.time < self.preparation_steps
        clipped = jnp.clip(chosen, 0, NUM_ACTIONS - 1)
        allowed = self.legal(preparing)[jnp.arange(self.num_agents), clipped] & (chosen == clipped)
        action = jnp.where(allowed, clipped, NOOP)
        state = self.lock(self.grab(state, action == GRAB), action == LOCK)
        state = self.move(state, action)
        sim, grounded = self.simulate(state.sim)
        state = state.replace(sim=sim, grounded=grounded, broadcast=action == BROADCAST, time=state.time + 1)
        sight = self.sight(state)
        hidden, seen = self.hidden(sight)
        score = jnp.where(hidden, 1.0, -1.0)
        reward = jnp.where(preparing, 0.0, jnp.where(self.team == HIDER, score, -score)).astype(jnp.float32)
        timestep = Timestep(
            obs=self.observations(state, sight),
            action=chosen,
            reward=reward,
            terminated=jnp.zeros(self.num_agents, bool),
            truncated=jnp.full(self.num_agents, state.time >= self._time_limit),
            info=self.tally(state, hidden, seen, preparing),
        )
        return state, timestep

    def legal(self, preparing: Array) -> Array:
        mask = jnp.ones((self.num_agents, NUM_ACTIONS), bool)
        mask = mask.at[:, GRAB].set(self.grabbing and self.num_objects > 0)
        mask = mask.at[:, LOCK].set(self.locking and self.num_objects > 0)
        mask = mask.at[:, BROADCAST].set(self.communication)
        frozen = (self.team == SEEKER) & preparing
        return jnp.where(frozen[:, None], jnp.arange(NUM_ACTIONS) == NOOP, mask)

    def action_mask(self, state: HideAndSeekState) -> Array:
        return self.legal(state.time < self.preparation_steps)

    def edges(self, sim: SimState) -> tuple[Array, Array, Array]:
        polygon = sim.polygon
        cosine, sine = jnp.cos(polygon.rotation), jnp.sin(polygon.rotation)
        rotation = jnp.stack([jnp.stack([cosine, -sine], -1), jnp.stack([sine, cosine], -1)], -2)
        starts = polygon.position[:, None] + jnp.einsum("pij,pvj->pvi", rotation, polygon.vertices)
        corner = jnp.arange(polygon.vertices.shape[1])
        following = (corner[None] + 1) % polygon.n_vertices[:, None]
        ends = jnp.take_along_axis(starts, following[..., None], axis=1)
        valid = polygon.active[:, None] & (corner[None] < polygon.n_vertices[:, None])
        return starts, ends, valid

    def gaps(self, sim: SimState) -> Array:
        starts, ends, valid = (part[self.first_object :] for part in self.edges(sim))
        centre = sim.circle.position[:, None, None]
        side = ends - starts
        fraction = jnp.clip(((centre - starts) * side).sum(-1) / jnp.maximum((side * side).sum(-1), EPSILON), 0.0, 1.0)
        distance = jnp.linalg.norm(centre - (starts + fraction[..., None] * side), axis=-1)
        return jnp.where(valid, distance, jnp.inf).min(-1) - AGENT_RADIUS

    def grab(self, state: HideAndSeekState, pressed: Array) -> HideAndSeekState:
        if self.num_objects == 0:
            return state
        sim = state.sim
        gap = jnp.where(state.locked_by[None] == UNLOCKED, self.gaps(sim), jnp.inf)
        nearest = jnp.argmin(gap, -1)
        holding = state.holding != EMPTY
        taking = pressed & ~holding & (gap.min(-1) < REACH)
        held = jnp.where(taking, nearest, jnp.where(pressed & holding, EMPTY, state.holding))
        index = self.first_object + jnp.maximum(held, 0)
        turn = sim.polygon.rotation[index]
        offset = sim.circle.position - sim.polygon.position[index]
        cosine, sine = jnp.cos(turn), jnp.sin(turn)
        local = jnp.stack(
            [cosine * offset[:, 0] + sine * offset[:, 1], -sine * offset[:, 0] + cosine * offset[:, 1]], -1
        )
        joint = sim.joint
        joint = joint.replace(
            a_index=jnp.where(taking, self.num_polygons + jnp.arange(self.num_agents), joint.a_index),
            b_index=jnp.where(taking, index, joint.b_index),
            a_relative_pos=jnp.where(taking[:, None], 0.0, joint.a_relative_pos),
            b_relative_pos=jnp.where(taking[:, None], local, joint.b_relative_pos),
            rotation=jnp.where(taking, turn - sim.circle.rotation, joint.rotation),
            acc_impulse=jnp.where(taking[:, None], 0.0, joint.acc_impulse),
            acc_r_impulse=jnp.where(taking, 0.0, joint.acc_r_impulse),
            is_fixed_joint=jnp.ones_like(joint.is_fixed_joint),
            active=held != EMPTY,
        )
        sim = sim.replace(joint=joint, collision_matrix=calculate_collision_matrix(self.static_params, joint))
        return state.replace(sim=sim, holding=held)

    def lock(self, state: HideAndSeekState, pressed: Array) -> HideAndSeekState:
        if self.num_objects == 0:
            return state
        sim = state.sim
        objects = jnp.arange(self.num_objects)
        held = (state.holding[:, None] == objects[None]).any(0)
        gap = jnp.where(held[None], jnp.inf, self.gaps(sim))
        acting = pressed & (gap.min(-1) < REACH)
        touched = acting[:, None] & (jnp.argmin(gap, -1)[:, None] == objects[None])
        locked = state.locked_by != UNLOCKED
        claimed = touched.any(0) & ~locked
        released = (touched & (self.team[:, None] == state.locked_by[None])).any(0) & locked
        locked_by = jnp.where(
            claimed, self.team[jnp.argmax(touched, 0)], jnp.where(released, UNLOCKED, state.locked_by)
        )
        frozen = locked_by != UNLOCKED
        index = self.first_object + objects
        polygon = sim.polygon
        polygon = polygon.replace(
            inverse_mass=polygon.inverse_mass.at[index].set(jnp.where(frozen, 0.0, state.inverse_mass)),
            inverse_inertia=polygon.inverse_inertia.at[index].set(jnp.where(frozen, 0.0, state.inverse_inertia)),
            velocity=polygon.velocity.at[index].set(jnp.where(frozen[:, None], 0.0, polygon.velocity[index])),
            angular_velocity=polygon.angular_velocity.at[index].set(
                jnp.where(frozen, 0.0, polygon.angular_velocity[index])
            ),
        )
        return state.replace(sim=sim.replace(polygon=polygon), locked_by=locked_by)

    def move(self, state: HideAndSeekState, action: Array) -> HideAndSeekState:
        circle = state.sim.circle
        heading = HEADING[action]
        change = ACCELERATION * self.step_time
        across = circle.velocity[:, 0] + jnp.clip(heading * SPEED - circle.velocity[:, 0], -change, change)
        up = jnp.where(LEAPING[action] & state.grounded, self.jump_speed, circle.velocity[:, 1])
        circle = circle.replace(velocity=jnp.stack([across, up], -1))
        return state.replace(
            sim=state.sim.replace(circle=circle), facing=jnp.where(heading != 0, heading, state.facing)
        )

    def simulate(self, sim: SimState) -> tuple[SimState, Array]:
        def substep(sim, _):
            sim, (_, polygons, circles) = self.engine.step(sim, self.params, self.motors)
            return sim, self.standing(polygons, circles)

        sim, grounded = jax.lax.scan(substep, sim, None, length=self.substeps)
        return sim, grounded[-1]

    def standing(self, polygons, circles) -> Array:
        on_polygon = (polygons.active & (polygons.normal[:, 1] < -STANDING)).reshape(self.num_agents, -1).any(-1)
        pairs = self.engine.circle_circle_pairs
        below = (circles.active & (circles.normal[:, 1] < -STANDING)).astype(jnp.int32)
        above = (circles.active & (circles.normal[:, 1] > STANDING)).astype(jnp.int32)
        on_agent = jnp.zeros(self.num_agents, jnp.int32).at[pairs[:, 0]].max(below).at[pairs[:, 1]].max(above)
        return on_polygon | (on_agent > 0)

    def sight(self, state: HideAndSeekState) -> Array:
        sim = state.sim
        starts, ends, valid = (part[BOUNDARY:] for part in self.edges(sim))
        occluders = jnp.arange(starts.shape[0])
        targets = jnp.concatenate([sim.circle.position, sim.polygon.position[self.first_object :]])
        owners = jnp.concatenate(
            [jnp.full(self.num_agents, EMPTY), self.first_object - BOUNDARY + jnp.arange(self.num_objects)]
        )

        def visible(eye, facing, target, owner):
            along, _, crossing = intersect(eye, target - eye, starts, ends)
            blocking = crossing & valid & (occluders[:, None] != owner) & (along > 0.0) & (along < 1.0)
            offset = target - eye
            ahead = facing * offset[0] >= self.vision * jnp.linalg.norm(offset)
            return ahead & ~blocking.any()

        looks = jax.vmap(visible, (None, None, 0, 0))
        return jax.vmap(looks, (0, 0, None, None))(sim.circle.position, state.facing, targets, owners)

    def hidden(self, sight: Array) -> tuple[Array, Array]:
        seekers = self.team == SEEKER
        seen = (sight[:, : self.num_agents] & seekers[:, None]).any(0) & (self.team == HIDER)
        return ~seen.any(), seen

    def scan(self, sim: SimState) -> tuple[Array, Array]:
        starts, ends, valid = self.edges(sim)
        movable = jnp.arange(self.num_polygons)[:, None] >= self.first_object

        def ray(origin, direction):
            along, _, crossing = intersect(origin, direction, starts, ends)
            distance = jnp.where(crossing & valid & (along > 0.0), along, jnp.inf)
            closest = distance.min()
            nearest = jnp.argmin(distance.reshape(-1)) // distance.shape[1]
            return jnp.minimum(closest, self.width) / self.width, movable[nearest, 0] & jnp.isfinite(closest)

        cast = jax.vmap(ray, (None, 0))
        return jax.vmap(cast, (0, None))(sim.circle.position, self.rays)

    def observe(self, state: HideAndSeekState) -> dict[str, Array]:
        return self.observations(state, self.sight(state))

    def observations(self, state: HideAndSeekState, sight: Array) -> dict[str, Array]:
        return {"agent": self.view(state, sight), "state": self.view(state, jnp.ones_like(sight))}

    def view(self, state: HideAndSeekState, sight: Array) -> Array:
        sim = state.sim
        position, velocity = sim.circle.position, sim.circle.velocity
        scale = jnp.array([SPEED, self.jump_speed])
        preparing = state.time < self.preparation_steps
        own = jnp.stack(
            [
                position[:, 0] / self.width,
                (position[:, 1] - GROUND) / self.width,
                velocity[:, 0] / SPEED,
                velocity[:, 1] / self.jump_speed,
                state.facing,
                self.team == SEEKER,
                state.holding != EMPTY,
                state.grounded,
                jnp.broadcast_to(preparing, (self.num_agents,)),
                jnp.broadcast_to(state.time / self._time_limit, (self.num_agents,)),
            ],
            -1,
        ).astype(jnp.float32)
        distances, movable = self.scan(sim)

        others = self.others
        spotted = sight[jnp.arange(self.num_agents)[:, None], others][..., None]
        teammate = (self.team[others] == self.team[:, None])[..., None]
        agents = jnp.concatenate(
            [
                spotted * jnp.concatenate(
                    [
                        jnp.ones_like(teammate, jnp.float32),
                        (position[others] - position[:, None]) / self.width,
                        (velocity[others] - velocity[:, None]) / scale,
                        state.facing[others][..., None],
                        teammate,
                    ],
                    -1,
                ),
                teammate & state.broadcast[others][..., None],
            ],
            -1,
        )

        polygon = jax.tree.map(lambda leaf: leaf[self.first_object :], sim.polygon)
        noticed = sight[:, self.num_agents :][..., None]
        shape = (self.num_agents, self.num_objects)
        objects = noticed * jnp.concatenate(
            [
                jnp.ones((*shape, 1), jnp.float32),
                (polygon.position[None] - position[:, None]) / self.width,
                jnp.broadcast_to(polygon.velocity / scale, (*shape, 2)),
                jnp.broadcast_to(jnp.sin(polygon.rotation), shape)[..., None],
                jnp.broadcast_to(jnp.cos(polygon.rotation), shape)[..., None],
                jnp.broadcast_to(self.is_ramp, shape)[..., None],
                jnp.broadcast_to(state.locked_by != UNLOCKED, shape)[..., None],
                (state.locked_by[None] == self.team[:, None])[..., None],
                (state.holding[:, None] == jnp.arange(self.num_objects)[None])[..., None],
            ],
            -1,
        )
        return jnp.concatenate(
            [
                own,
                distances,
                movable,
                agents.reshape(self.num_agents, -1),
                objects.reshape(self.num_agents, -1),
            ],
            -1,
        ).astype(jnp.float32)

    def tally(self, state: HideAndSeekState, hidden: Array, seen: Array, preparing: Array) -> dict[str, Array]:
        sim = state.sim
        height = sim.circle.position[:, 1] - GROUND - AGENT_RADIUS
        info = {
            "hidden": hidden,
            "hiders_seen": seen.sum() / max(self.num_hiders, 1),
            "preparing": preparing,
            "holding": (state.holding != EMPTY).mean(),
            "broadcasts": state.broadcast.mean(),
        }
        for name, team, count in (("hider_height", HIDER, self.num_hiders), ("seeker_height", SEEKER, self.num_seekers)):
            info[name] = jnp.where(self.team == team, height, -jnp.inf).max() if count else 0.0
        if self.num_objects:
            objects = slice(self.first_object, self.num_polygons)
            info["locked"] = (state.locked_by != UNLOCKED).mean()
            info["object_speed"] = jnp.linalg.norm(sim.polygon.velocity[objects], axis=-1).mean()
            info["object_height"] = (sim.polygon.position[objects, 1] - GROUND).max()
        return {name: jnp.asarray(value, jnp.float32) for name, value in info.items()}

    def observation_space(self) -> dict[str, Space]:
        space = lambda: Space(
            shape=(self.num_agents, self.observation_size), dtype=jnp.float32, low=-jnp.inf, high=jnp.inf
        )
        return {"agent": space(), "state": space()}

    def action_space(self) -> Space:
        return Space(shape=(self.num_agents,), dtype=jnp.int32, low=0, high=NUM_ACTIONS - 1)

    def time_limit(self) -> int:
        return self._time_limit

    def render(
        self,
        state: HideAndSeekState,
        scale: int = 32,
        height: float = MAX_WALL_HEIGHT + 3.0,
        *,
        supersample: int = 2,
        sight: bool = False,
    ) -> Array:
        """Return an RGB image of the arena, with height measured above the floor."""
        key = (scale, height, supersample, sight)
        if key not in self._renderers:
            # Cache concrete screen coordinates even if the first call is under JIT.
            with jax.ensure_compile_time_eval():
                from .hide_and_seek_render import make_render

                renderer = make_render(self, scale, height, supersample, sight)
            self._renderers.setdefault(key, renderer)
        return self._renderers[key](state)
