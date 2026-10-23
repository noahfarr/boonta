from dataclasses import dataclass

import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import math

from boonta.utils import Array

from .motions import Motions

NUM_JOINTS = 29
QPOS, QVEL = 7 + NUM_JOINTS, 6 + NUM_JOINTS
STATE = QPOS + QVEL + NUM_JOINTS
OBSERVATION = NUM_JOINTS + NUM_JOINTS + 3 + 3 + NUM_JOINTS
CONTEXT = 1 + NUM_JOINTS + 3
GOAL_FRAME = NUM_JOINTS + NUM_JOINTS + 3 + 6 + 3 + 3
NOISE = jnp.concatenate(
    [
        jnp.full(NUM_JOINTS, 0.01),
        jnp.full(NUM_JOINTS, 0.5),
        jnp.full(3, 0.2),
        jnp.full(3, 0.05),
        jnp.zeros(NUM_JOINTS),
    ]
)
DOWN = jnp.array([0.0, 0.0, -1.0])


@dataclass(frozen=True)
class Tracking:
    kinematics: mjx.Model
    motions: Motions
    tracked: Array
    default_pose: Array
    soft_lower: Array
    soft_upper: Array
    goal_frames: int = 10
    body_position_std: float = 0.3
    body_orientation_std: float = 0.4
    body_linear_velocity_std: float = 1.0
    body_angular_velocity_std: float = 3.14
    root_position_std: float = 0.3
    root_orientation_std: float = 0.4
    root_weight: float = 0.5
    action_rate_weight: float = -0.1
    joint_limit_weight: float = -2.0
    termination_height: float = 0.25
    termination_tilt: float = 0.8
    control_dt: float = 0.02

    def split(self, state: Array) -> tuple[Array, Array, Array]:
        return state[..., :QPOS], state[..., QPOS : QPOS + QVEL], state[..., QPOS + QVEL :]

    def bodies(self, qpos: Array, qvel: Array) -> tuple[Array, Array, Array, Array]:
        def pose(qpos):
            data = mjx.kinematics(self.kinematics, mjx.make_data(self.kinematics).replace(qpos=qpos))
            return data.xpos[self.tracked], data.xquat[self.tracked]

        root = qpos[3:7]
        tangent = jnp.concatenate(
            [qvel[:3], 0.5 * math.quat_mul(root, jnp.concatenate([jnp.zeros(1), qvel[3:6]])), qvel[6:]]
        )
        (position, orientation), (linear, turning) = jax.jvp(pose, (qpos,), (tangent,))
        conjugate = orientation * jnp.array([1.0, -1.0, -1.0, -1.0])
        angular = 2.0 * jax.vmap(math.quat_mul)(turning, conjugate)[:, 1:]
        return position, orientation, linear, angular

    def reward(self, state: Array, following: Array, frame: Array) -> Array:
        anchor, _, previous = self.split(state)
        qpos, qvel, action = self.split(following)
        position, orientation, linear, angular = self.bodies(qpos, qvel)
        motions = self.motions
        reference_position = motions.body_position[frame]
        reference_orientation = motions.body_orientation[frame]
        heading = _yaw(math.quat_mul(anchor[3:7], math.quat_inv(reference_orientation[0])))
        origin = jnp.concatenate([anchor[:2], reference_position[0, 2:]])
        relative_position = origin + jax.vmap(math.rotate, in_axes=(0, None))(
            reference_position - reference_position[0], heading
        )
        relative_orientation = jax.vmap(math.quat_mul, in_axes=(None, 0))(heading, reference_orientation)

        def kernel(error, std):
            return jnp.exp(-error / std**2)

        tracking = (
            kernel(jnp.mean(jnp.sum((relative_position - position) ** 2, axis=-1)), self.body_position_std)
            + kernel(jnp.mean(jax.vmap(_angle)(relative_orientation, orientation) ** 2), self.body_orientation_std)
            + kernel(
                jnp.mean(jnp.sum((motions.body_linear_velocity[frame] - linear) ** 2, axis=-1)),
                self.body_linear_velocity_std,
            )
            + kernel(
                jnp.mean(jnp.sum((motions.body_angular_velocity[frame] - angular) ** 2, axis=-1)),
                self.body_angular_velocity_std,
            )
            + self.root_weight
            * kernel(jnp.sum((reference_position[0] - position[0]) ** 2), self.root_position_std)
            + self.root_weight
            * kernel(_angle(reference_orientation[0], orientation[0]) ** 2, self.root_orientation_std)
        )
        joints = qpos[7:]
        limits = jnp.sum(jnp.maximum(joints - self.soft_upper, 0.0) + jnp.maximum(self.soft_lower - joints, 0.0))
        regularization = self.action_rate_weight * jnp.sum((action - previous) ** 2) + self.joint_limit_weight * limits
        return (tracking + regularization) * self.control_dt

    def errors(self, state: Array, frame: Array) -> tuple[Array, Array]:
        qpos, _, _ = self.split(state)
        reference_position = self.motions.body_position[frame, 0]
        reference_orientation = self.motions.body_orientation[frame, 0]
        height = jnp.abs(reference_position[2] - qpos[2])
        tilt = jnp.abs(
            math.rotate(DOWN, math.quat_inv(reference_orientation))[2] - math.rotate(DOWN, math.quat_inv(qpos[3:7]))[2]
        )
        return height, tilt

    def terminated(self, state: Array, frame: Array) -> Array:
        height, tilt = self.errors(state, frame)
        return (height > self.termination_height) | (tilt > self.termination_tilt)

    def observe(self, state: Array, context: Array) -> Array:
        qpos, qvel, previous = self.split(state)
        bias = context[..., 1 : 1 + NUM_JOINTS]
        gravity = math.rotate(DOWN, math.quat_inv(qpos[3:7]))
        return jnp.concatenate([qpos[7:] + bias - self.default_pose, qvel[6:], qvel[3:6], gravity, previous])

    def perturb(self, key: Array, observation: Array) -> Array:
        return observation + jax.random.uniform(key, observation.shape, minval=-1.0, maxval=1.0) * NOISE

    def advance(self, frame: Array) -> Array:
        motions = self.motions
        return jnp.minimum(frame + 1, motions.clip_end[motions.clip[frame]] - 1)

    def goal(self, frame: Array) -> Array:
        motions = self.motions
        last = motions.clip_end[motions.clip[frame]] - 1
        frames = jnp.minimum(frame + 1 + jnp.arange(self.goal_frames), last)
        root_position, root_orientation = motions.body_position[:, 0], motions.body_orientation[:, 0]
        inverse = math.quat_inv(_yaw(root_orientation[frame]))

        def describe(index):
            orientation = math.quat_mul(inverse, root_orientation[index])
            return jnp.concatenate(
                [
                    motions.joint_position[index],
                    motions.joint_velocity[index],
                    math.rotate(root_position[index] - root_position[frame], inverse),
                    math.quat_to_mat(orientation)[:, :2].reshape(-1),
                    math.rotate(motions.body_linear_velocity[index, 0], inverse),
                    math.rotate(motions.body_angular_velocity[index, 0], inverse),
                ]
            )

        return jax.vmap(describe)(frames).reshape(-1)


def _yaw(quaternion: Array) -> Array:
    w, x, y, z = quaternion
    angle = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return jnp.array([jnp.cos(angle / 2.0), 0.0, 0.0, jnp.sin(angle / 2.0)])


def _angle(left: Array, right: Array) -> Array:
    difference = math.quat_mul(math.quat_inv(left), right)
    return 2.0 * jnp.arctan2(jnp.linalg.norm(difference[1:]), jnp.abs(difference[0]))
