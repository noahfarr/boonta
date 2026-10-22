from collections.abc import Sequence

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct
from mujoco import mjx

from boonta.utils import Array, Key, Timestep

from ..environment import Environment
from ..spaces import Space
from .menagerie import fetch

WEIGHTS = {
    "velocity": 1.0,
    "turning": 0.75,
    "gait": 1.0,
    "upright": -2.0,
    "bounce": -0.5,
    "wobble": -0.05,
    "smoothness": -0.01,
    "pose": -0.2,
    "fall": -50.0,
}
LOOSE_JOINTS = ("hip_pitch", "knee", "ankle")


@struct.dataclass
class WalkState:
    qpos: Array
    qvel: Array
    warmstart: Array
    command: Array
    phase: Array
    action: Array
    time: Array


def conjugate(quat: Array) -> Array:
    return quat * jnp.array([1.0, -1.0, -1.0, -1.0])


def rotate(vector: Array, quat: Array) -> Array:
    scalar, axis = quat[0], quat[1:]
    twist = 2.0 * jnp.cross(axis, vector)
    return vector + scalar * twist + jnp.cross(axis, twist)


class Walk(Environment):
    def __init__(
        self,
        impl: str = "warp",
        worlds: int = 1,
        contacts: int = 24,
        constraints: int = 128,
        episode_length: int = 1000,
        control_dt: float = 0.02,
        action_scale: float = 0.5,
        gait_frequency: float = 1.25,
        foot_height: float = 0.08,
        command_low: Sequence[float] = (-1.0, -0.5, -1.0),
        command_high: Sequence[float] = (1.0, 0.5, 1.0),
        command_period: int = 500,
        standing: float = 0.1,
        noise: float = 0.05,
        weights: dict[str, float] | None = None,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(fetch("unitree_g1") / "scene_mjx.xml"))
        self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        self.impl = impl
        self.mjx_model = mjx.put_model(self.model, impl=impl)
        if impl == "warp":
            self.mjx_model = self.mjx_model.tree_replace({"opt._impl": self.mjx_model.opt._impl.replace(warn_overflow=0)})
        buffers = {"naconmax": contacts * worlds, "njmax": constraints} if impl == "warp" else {}
        self.template = mjx.make_data(self.model, impl=impl, **buffers)
        self.episode_length = episode_length
        self.control_dt = control_dt
        self.substeps = round(control_dt / self.model.opt.timestep)
        self.action_scale = action_scale
        self.gait_frequency = gait_frequency
        self.foot_height = foot_height
        self.command_low = jnp.asarray(command_low, jnp.float32)
        self.command_high = jnp.asarray(command_high, jnp.float32)
        self.command_period = command_period
        self.standing = standing
        self.noise = noise
        self.weights = {**WEIGHTS, **(weights or {})}
        pose = self.model.key("knees_bent")
        self.default_qpos = jnp.asarray(pose.qpos, jnp.float32)
        self.default_ctrl = jnp.asarray(pose.ctrl, jnp.float32)
        self.feet = np.array([self.model.site("left_foot").id, self.model.site("right_foot").id])
        names = [self.model.joint(index).name for index in range(1, self.model.njnt)]
        self.stiffness = jnp.asarray([0.1 if any(part in name for part in LOOSE_JOINTS) else 1.0 for name in names])
        rest = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, rest, pose.id)
        mujoco.mj_forward(self.model, rest)
        self.foot_rest = float(rest.site_xpos[self.feet, 2].mean())
        self.num_joints = self.model.nu

    def draw(self, key: Key) -> Array:
        command_key, standing_key = jax.random.split(key)
        command = jax.random.uniform(command_key, (3,), jnp.float32, self.command_low, self.command_high)
        return jnp.where(jax.random.uniform(standing_key) < self.standing, 0.0, command)

    def init(self, key: Key) -> tuple[WalkState, Timestep]:
        joint_key, velocity_key, yaw_key, command_key = jax.random.split(key, 4)
        yaw = jax.random.uniform(yaw_key, (), jnp.float32, -jnp.pi, jnp.pi)
        qpos = self.default_qpos.at[3:7].set(jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)]))
        qpos = qpos.at[7:].add(jax.random.uniform(joint_key, (self.num_joints,), jnp.float32, -self.noise, self.noise))
        qvel = jax.random.uniform(velocity_key, (self.model.nv,), jnp.float32, -self.noise, self.noise)
        data = self.template.replace(qpos=qpos, qvel=qvel, ctrl=self.default_ctrl)
        data = mjx.forward(self.mjx_model, data)
        state = WalkState(
            qpos=data.qpos,
            qvel=data.qvel,
            warmstart=data.qacc_warmstart,
            command=self.draw(command_key),
            phase=jnp.array([0.0, jnp.pi], jnp.float32),
            action=jnp.zeros(self.num_joints, jnp.float32),
            time=jnp.int32(0),
        )
        timestep = Timestep(
            obs=self.observe(state),
            action=jnp.zeros(self.num_joints, jnp.float32),
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(False),
            truncated=jnp.bool_(False),
            info={},
        )
        return state, timestep

    def step(self, key: Key, state: WalkState, action: Array) -> tuple[WalkState, Timestep]:
        action = jnp.asarray(action, jnp.float32)
        data = self.template.replace(
            qpos=state.qpos,
            qvel=state.qvel,
            qacc_warmstart=state.warmstart,
            ctrl=self.default_ctrl + self.action_scale * action,
        )
        data = jax.lax.fori_loop(0, self.substeps, lambda _, data: mjx.step(self.mjx_model, data), data)
        time = state.time + 1
        phase = jnp.mod(state.phase + 2 * jnp.pi * self.gait_frequency * self.control_dt, 2 * jnp.pi)
        command = jnp.where(time % self.command_period == 0, self.draw(key), state.command)
        next_state = WalkState(
            qpos=data.qpos,
            qvel=data.qvel,
            warmstart=data.qacc_warmstart,
            command=command,
            phase=phase,
            action=action,
            time=time,
        )
        fallen = self.fallen(data.qpos)
        terms = self.score(state, next_state, data, fallen)
        reward = self.control_dt * sum(self.weights[name] * value for name, value in terms.items())
        timestep = Timestep(
            obs=self.observe(next_state),
            action=action,
            reward=jnp.asarray(reward, jnp.float32),
            terminated=fallen,
            truncated=(time >= self.episode_length) & ~fallen,
            info={},
        )
        return next_state, timestep

    def gravity(self, qpos: Array) -> Array:
        return rotate(jnp.array([0.0, 0.0, -1.0]), conjugate(qpos[3:7]))

    def fallen(self, qpos: Array) -> Array:
        return (self.gravity(qpos)[2] > -0.6) | (qpos[2] < 0.45)

    def score(self, state: WalkState, next_state: WalkState, data: mjx.Data, fallen: Array) -> dict[str, Array]:
        quat = data.qpos[3:7]
        velocity = rotate(data.qvel[:3], conjugate(quat))
        spin = data.qvel[3:6]
        feet = data.site_xpos[self.feet, 2] - self.foot_rest
        lift = self.foot_height * jnp.clip(jnp.sin(next_state.phase), 0.0, 1.0)
        command = state.command
        return {
            "velocity": jnp.exp(-jnp.sum(jnp.square(command[:2] - velocity[:2])) / 0.25),
            "turning": jnp.exp(-jnp.square(command[2] - spin[2]) / 0.25),
            "gait": jnp.exp(-jnp.sum(jnp.square(feet - lift)) / 0.005),
            "upright": jnp.sum(jnp.square(self.gravity(data.qpos)[:2])),
            "bounce": jnp.square(data.qvel[2]),
            "wobble": jnp.sum(jnp.square(spin[:2])),
            "smoothness": jnp.sum(jnp.square(next_state.action - state.action)),
            "pose": jnp.sum(self.stiffness * jnp.square(data.qpos[7:] - self.default_qpos[7:])),
            "fall": fallen.astype(jnp.float32),
        }

    def observe(self, state: WalkState) -> Array:
        return jnp.concatenate(
            [
                0.25 * state.qvel[3:6],
                self.gravity(state.qpos),
                state.command,
                state.qpos[7:] - self.default_qpos[7:],
                0.05 * state.qvel[6:],
                state.action,
                jnp.cos(state.phase),
                jnp.sin(state.phase),
            ]
        ).astype(jnp.float32)

    def observation_space(self) -> Space:
        size = 3 + 3 + 3 + 3 * self.num_joints + 4
        return Space(shape=(size,), dtype=jnp.float32, low=-jnp.inf, high=jnp.inf)

    def action_space(self) -> Space:
        return Space(shape=(self.num_joints,), dtype=jnp.float32, low=-1.0, high=1.0)

    def time_limit(self) -> int:
        return self.episode_length

    def render(self, state: WalkState, height: int = 480, width: int = 640, camera: str = "track") -> Array:
        data = mujoco.MjData(self.model)
        data.qpos, data.qvel = np.asarray(state.qpos), np.asarray(state.qvel)
        mujoco.mj_forward(self.model, data)
        renderer = mujoco.Renderer(self.model, height, width)
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()
        renderer.close()
        return frame
