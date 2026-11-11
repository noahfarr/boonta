from dataclasses import dataclass
from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from hydra.utils import instantiate

from boonta import environments
from boonta.algorithms.super_mimic import SuperMimic
from boonta.environments.wrappers import RecordEpisodeStatistics
from boonta.networks import FeatureExtractor, FixedGaussian, Network, WorldModel
from boonta.networks.layers import Normalize
from boonta.utils import Array


@struct.dataclass(frozen=True)
class Motions:
    joint_position: Array
    joint_velocity: Array
    body_position: Array
    body_orientation: Array
    body_linear_velocity: Array
    body_angular_velocity: Array
    clip: Array
    clip_start: Array
    clip_end: Array

    @property
    def root_position(self) -> Array:
        return self.body_position[:, 0]

    @property
    def root_orientation(self) -> Array:
        return self.body_orientation[:, 0]

    @property
    def root_linear_velocity(self) -> Array:
        return self.body_linear_velocity[:, 0]

    @property
    def root_angular_velocity(self) -> Array:
        return self.body_angular_velocity[:, 0]

    def last(self, frame: Array) -> Array:
        return self.clip_end[self.clip[frame]] - 1


@struct.dataclass(frozen=True)
class Robot:
    position: Array
    orientation: Array
    joint_position: Array
    linear_velocity: Array
    angular_velocity: Array
    joint_velocity: Array
    action: Array

    @property
    def qpos(self) -> Array:
        return jnp.concatenate(
            [self.position, self.orientation, self.joint_position], axis=-1
        )

    @property
    def qvel(self) -> Array:
        return jnp.concatenate(
            [self.linear_velocity, self.angular_velocity, self.joint_velocity], axis=-1
        )


def load_motions(library: str, bodies: tuple[int, ...]) -> Motions:
    stored = np.load(library)
    lengths = stored["clip_end"] - stored["clip_start"]
    bodies = np.asarray(bodies)
    return Motions(
        joint_position=jnp.asarray(stored["joint_pos"]),
        joint_velocity=jnp.asarray(stored["joint_vel"]),
        body_position=jnp.asarray(stored["body_pos_w"][:, bodies]),
        body_orientation=jnp.asarray(stored["body_quat_w"][:, bodies]),
        body_linear_velocity=jnp.asarray(stored["body_lin_vel_w"][:, bodies]),
        body_angular_velocity=jnp.asarray(stored["body_ang_vel_w"][:, bodies]),
        clip=jnp.asarray(np.repeat(np.arange(len(lengths)), lengths)),
        clip_start=jnp.asarray(stored["clip_start"]),
        clip_end=jnp.asarray(stored["clip_end"]),
    )


@dataclass(frozen=True)
class Tracking:
    kinematics: "mjx.Model"  # noqa: F821 -- mjx imported lazily to keep mujoco optional
    motions: Motions
    tracked: Array
    default_pose: Array
    soft_lower: Array
    soft_upper: Array
    goal_frames: int = 10
    goal_stride: int = 5
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
    termination_softness: float = 0.05
    root_position_nominal: float = 0.01
    root_orientation_nominal: float = 0.02
    joint_position_nominal: float = 0.1
    joint_velocity_nominal: float = 3.14
    control_dt: float = 0.02

    @property
    def best_reward(self) -> float:
        return (4.0 + 2.0 * self.root_weight) * self.control_dt

    @property
    def num_joints(self) -> int:
        num_joints, *_ = self.default_pose.shape
        return num_joints

    def split(self, state: Array) -> Robot:
        n = self.num_joints
        return Robot(*jnp.split(state, np.cumsum([3, 4, n, 3, 3, n]), axis=-1))

    def bodies(self, robot: Robot) -> tuple[Array, Array, Array, Array]:
        from mujoco import mjx
        from mujoco.mjx._src import math

        def pose(qpos):
            data = mjx.kinematics(
                self.kinematics, mjx.make_data(self.kinematics).replace(qpos=qpos)
            )
            return data.xpos[self.tracked], data.xquat[self.tracked]

        spin = 0.5 * math.quat_mul(
            robot.orientation, jnp.concatenate([jnp.zeros(1), robot.angular_velocity])
        )
        tangent = jnp.concatenate([robot.linear_velocity, spin, robot.joint_velocity])
        (position, orientation), (linear, turning) = jax.jvp(
            pose, (robot.qpos,), (tangent,)
        )
        conjugate = orientation * jnp.array([1.0, -1.0, -1.0, -1.0])
        angular = 2.0 * jax.vmap(math.quat_mul)(turning, conjugate)[:, 1:]
        return position, orientation, linear, angular

    def reward(self, state: Array, next_state: Array, frame: Array) -> Array:
        from mujoco.mjx._src import math

        before, after = self.split(state), self.split(next_state)
        position, orientation, linear, angular = self.bodies(after)
        motions = self.motions
        reference_position = motions.body_position[frame]
        reference_orientation = motions.body_orientation[frame]
        root_position, *_ = position
        root_orientation, *_ = orientation
        reference_root_position, *_ = reference_position
        reference_root_orientation, *_ = reference_orientation
        heading = _yaw(
            math.quat_mul(after.orientation, math.quat_inv(reference_root_orientation))
        )
        origin = reference_root_position.at[:2].set(after.position[:2])
        relative_position = origin + jax.vmap(math.rotate, in_axes=(0, None))(
            reference_position - reference_root_position, heading
        )
        relative_orientation = jax.vmap(math.quat_mul, in_axes=(None, 0))(
            heading, reference_orientation
        )

        def kernel(error, std):
            return jnp.exp(-error / std**2)

        def squared(reference, actual):
            return jnp.mean(jnp.sum((reference - actual) ** 2, axis=-1))

        tracking = (
            kernel(squared(relative_position, position), self.body_position_std)
            + kernel(
                jnp.mean(_angle(relative_orientation, orientation) ** 2),
                self.body_orientation_std,
            )
            + kernel(
                squared(motions.body_linear_velocity[frame], linear),
                self.body_linear_velocity_std,
            )
            + kernel(
                squared(motions.body_angular_velocity[frame], angular),
                self.body_angular_velocity_std,
            )
            + self.root_weight
            * kernel(
                squared(reference_root_position, root_position), self.root_position_std
            )
            + self.root_weight
            * kernel(
                _angle(reference_root_orientation, root_orientation) ** 2,
                self.root_orientation_std,
            )
        )
        joints = after.joint_position
        limits = jnp.sum(
            jnp.maximum(joints - self.soft_upper, 0.0)
            + jnp.maximum(self.soft_lower - joints, 0.0)
        )
        regularization = (
            self.action_rate_weight * jnp.sum((after.action - before.action) ** 2)
            + self.joint_limit_weight * limits
        )
        return (tracking + regularization) * self.control_dt

    def errors(self, state: Array, frame: Array) -> tuple[Array, Array]:
        robot = self.split(state)
        _, _, height = robot.position
        _, _, reference_height = self.motions.root_position[frame]
        _, _, up = _gravity(robot.orientation)
        _, _, reference_up = _gravity(self.motions.root_orientation[frame])
        return jnp.abs(reference_height - height), jnp.abs(reference_up - up)

    def terminated(self, state: Array, frame: Array) -> Array:
        height, tilt = self.errors(state, frame)
        return (height > self.termination_height) | (tilt > self.termination_tilt)

    def termination(self, state: Array, frame: Array) -> Array:
        height, tilt = self.errors(state, frame)
        excess = jnp.maximum(height / self.termination_height, tilt / self.termination_tilt)
        return (excess - 1.0) / self.termination_softness

    def mismatch(self, prediction: Array, target: Array) -> Array:
        predicted, actual = self.split(prediction), self.split(target)
        blocks = [
            (predicted.position, actual.position, self.root_position_nominal),
            (
                jnp.sqrt(0.5) * _matrix(predicted.orientation).reshape(*prediction.shape[:-1], 9),
                jnp.sqrt(0.5) * _matrix(actual.orientation).reshape(*target.shape[:-1], 9),
                self.root_orientation_nominal,
            ),
            (predicted.joint_position, actual.joint_position, self.joint_position_nominal),
            (predicted.joint_velocity, actual.joint_velocity, self.joint_velocity_nominal),
        ]
        weighted = sum(jnp.sum((left - right) ** 2, axis=-1) / nominal**2 for left, right, nominal in blocks)
        width = sum(jnp.shape(left)[-1] for left, _, _ in blocks)
        return weighted / width

    def observe(self, state: Array, context: Array) -> Array:
        robot = self.split(state)
        _, bias, _ = jnp.split(context, np.cumsum([1, self.num_joints]), axis=-1)
        return jnp.concatenate(
            [
                robot.joint_position + bias - self.default_pose,
                robot.joint_velocity,
                robot.angular_velocity,
                _gravity(robot.orientation),
                robot.action,
            ],
            axis=-1,
        )

    def perturb(self, key: Array, observation: Array) -> Array:
        n = self.num_joints
        noise = jnp.concatenate(
            [
                jnp.full(n, 0.01),
                jnp.full(n, 0.5),
                jnp.full(3, 0.2),
                jnp.full(3, 0.05),
                jnp.zeros(n),
            ]
        )
        return (
            observation
            + jax.random.uniform(key, jnp.shape(observation), minval=-1.0, maxval=1.0)
            * noise
        )

    def advance(self, frame: Array) -> Array:
        return jnp.minimum(frame + 1, self.motions.last(frame))

    def goal(self, state: Array, frame: Array) -> tuple[Array, Array, Array]:
        from mujoco.mjx._src import math

        motions = self.motions
        robot = self.split(state)
        frames = jnp.minimum(
            frame + self.goal_stride * jnp.arange(self.goal_frames), motions.last(frame)
        )
        content = jnp.concatenate(
            [motions.joint_position[frames], motions.joint_velocity[frames]], axis=-1
        )
        inverse = math.quat_inv(robot.orientation)
        position = jax.vmap(math.rotate, in_axes=(0, None))(
            motions.root_position[frames] - robot.position, inverse
        )
        orientation = jax.vmap(
            lambda target: _columns(math.quat_mul(inverse, target))
        )(motions.root_orientation[frames])
        return content.reshape(-1), position.reshape(-1), orientation.reshape(-1)


def _angle(left: Array, right: Array) -> Array:
    from mujoco.mjx._src import math

    return jnp.linalg.norm(
        jnp.vectorize(math.quat_sub, signature="(4),(4)->(3)")(left, right), axis=-1
    )


def _gravity(orientation: Array) -> Array:
    from mujoco.mjx._src import math

    down = jnp.array([0.0, 0.0, -1.0])
    return math.rotate(down, math.quat_inv(orientation))


def _columns(orientation: Array) -> Array:
    from mujoco.mjx._src import math

    return math.quat_to_mat(orientation)[:, :2].reshape(-1)


def describe(env) -> Tracking:
    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
    from mujoco import mjx

    skeleton = mujoco.MjModel.from_xml_path(str(G1_XML))
    skeleton.geom_contype[:] = 0
    skeleton.geom_conaffinity[:] = 0
    command = env.unwrapped.env.cfg.commands["motion"]
    tracked = [
        mujoco.mj_name2id(skeleton, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in command.body_names
    ]
    robot = env.unwrapped.env.scene["robot"].data
    limits = robot.soft_joint_pos_limits[0].cpu().numpy()
    return Tracking(
        control_dt=env.unwrapped.env.step_dt,
        kinematics=mjx.put_model(skeleton),
        motions=load_motions(command.motion_file, tuple(body - 1 for body in tracked)),
        tracked=jnp.asarray(tracked),
        default_pose=jnp.asarray(robot.default_joint_pos[0].cpu().numpy()),
        soft_lower=jnp.asarray(limits[:, 0]),
        soft_upper=jnp.asarray(limits[:, 1]),
    )


class GoalConditioned(nn.Module):
    encoder: nn.Module
    decoder: nn.Module
    goal_size: int
    observation_size: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        goal = Normalize()(x[..., -self.goal_size :])
        history = x[..., : -self.goal_size]
        *batch, width = jnp.shape(history)
        frames = jnp.reshape(
            history, (*batch, width // self.observation_size, self.observation_size)
        )
        history = jnp.reshape(Normalize()(frames), (*batch, width))
        latent = jnp.tanh(self.encoder(goal))
        return self.decoder(jnp.concatenate([latent, history], axis=-1))


def _quaternion_product(left: Array, right: Array) -> Array:
    lw, lx, ly, lz = jnp.moveaxis(left, -1, 0)
    rw, rx, ry, rz = jnp.moveaxis(right, -1, 0)
    return jnp.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def _matrix(quaternion: Array) -> Array:
    w, x, y, z = jnp.moveaxis(quaternion, -1, 0)
    rows = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    return jnp.stack([jnp.stack(row, axis=-1) for row in rows], axis=-2)


def _yaw(quaternion: Array) -> Array:
    w, x, y, z = jnp.moveaxis(quaternion, -1, 0)
    angle = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    zero = jnp.zeros_like(angle)
    return jnp.stack([jnp.cos(angle / 2.0), zero, zero, jnp.sin(angle / 2.0)], axis=-1)


def _exponential(rotation: Array) -> Array:
    angle = jnp.sqrt(jnp.sum(rotation**2, axis=-1, keepdims=True) + 1e-12)
    return jnp.concatenate(
        [jnp.cos(angle / 2.0), jnp.sin(angle / 2.0) * rotation / angle], axis=-1
    )


def _apply(matrix: Array, vector: Array) -> Array:
    return jnp.einsum("...ij,...j->...i", matrix, vector)


class FloatingBase(nn.Module):
    network: nn.Module
    num_joints: int
    dt: float
    substep: float

    def __call__(self, state: Array, action: Array, context: Array, **_) -> Array:
        n = self.num_joints
        position, orientation, joints, velocity, turning, joint_velocity, _ = jnp.split(
            state, np.cumsum([3, 4, n, 3, 3, n]), axis=-1
        )
        rotation = _matrix(orientation)
        local_velocity = jnp.einsum("...ji,...j->...i", rotation, velocity)
        features = jnp.concatenate(
            [
                position[..., 2:],
                -rotation[..., 2, :],
                local_velocity,
                turning,
                joints,
                joint_velocity,
                action,
                context,
            ],
            axis=-1,
        )
        acceleration, spin, joint_acceleration = jnp.split(
            self.network(features), [3, 6], axis=-1
        )
        step = _exponential((turning + self.substep * spin) * self.dt)
        following = _quaternion_product(orientation, step)
        following = following / jnp.linalg.norm(following, axis=-1, keepdims=True)
        return jnp.concatenate(
            [
                position
                + _apply(rotation, local_velocity + self.substep * acceleration) * self.dt,
                following,
                joints + (joint_velocity + self.substep * joint_acceleration) * self.dt,
                _apply(rotation, local_velocity + acceleration),
                jnp.einsum("...ji,...j->...i", _matrix(step), turning + spin),
                joint_velocity + joint_acceleration,
                action,
            ],
            axis=-1,
        )


class Linear(nn.Module):
    features: int
    dtype: Any = None
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        *_, width = jnp.shape(x)
        bound = 1.0 / np.sqrt(width)

        def uniform(key, shape, dtype=self.param_dtype):
            return jax.random.uniform(key, shape, dtype, -bound, bound)

        return nn.Dense(
            self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.variance_scaling(1.0 / 3.0, "fan_in", "uniform"),
            bias_init=uniform,
        )(x)


def mlp(sizes, outputs=None, dtype=None):
    def block(size):
        return (
            Linear(size, dtype=dtype),
            nn.LayerNorm(epsilon=1e-5, dtype=dtype, param_dtype=jnp.float32),
            partial(nn.gelu, approximate=False),
        )

    layers = [layer for size in sizes for layer in block(size)]
    if outputs:
        layers.append(Linear(outputs, dtype=dtype))
    layers.append(lambda x: x.astype(jnp.float32))
    return nn.Sequential(layers)


def normalized(module):
    return nn.Sequential([Normalize(), module])


def make(cfg):
    import boonta.environments.mjlab.tracking.config.g1  # noqa: F401

    env = environments.make(**cfg.environment)
    task = describe(env)
    num_joints = task.num_joints
    goal_size = task.goal_frames * (2 * num_joints + 6)
    observation_size = 3 * num_joints + 6
    decimation = env.unwrapped.env.cfg.decimation
    compute_dtype = jnp.dtype(cfg.network.dtype) if cfg.network.dtype else None

    def frame(command):
        return command[..., 0].astype(jnp.int32)

    def reward(state, next_state, command, **_):
        return jax.vmap(task.reward)(state, next_state, frame(command))

    def goal(state, command):
        content, _, orientation = jax.vmap(task.goal)(state, frame(command))
        return jnp.concatenate([content, orientation], axis=-1)

    def critic_goal(state, command):
        return jnp.concatenate(jax.vmap(task.goal)(state, frame(command)), axis=-1)

    def termination(state, command, **_):
        return jax.vmap(task.termination)(state, frame(command))

    def advance(command):
        return jax.vmap(task.advance)(frame(command))[..., None].astype(command.dtype)

    def observe(state, context, **_):
        return jax.vmap(task.observe)(state, context)

    def clipped(*transforms):
        return optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm), *transforms
        )

    def decayed(steps_per_iteration):
        return optax.cosine_decay_schedule(
            cfg.optimizer.lr,
            decay_steps=cfg.optimizer.decay_iterations * steps_per_iteration,
            alpha=cfg.optimizer.final_scale,
        )

    algorithm_cfg = cfg.algorithm
    num_critic_steps = (
        algorithm_cfg.policy_updates
        * algorithm_cfg.critic_epochs
        * algorithm_cfg.critic_minibatches
    )

    algorithm = SuperMimic(
        cfg=instantiate(cfg.algorithm),
        actor=Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=GoalConditioned(
                    encoder=mlp(cfg.network.encoder, cfg.network.latent, compute_dtype),
                    decoder=mlp(cfg.network.decoder, dtype=compute_dtype),
                    goal_size=goal_size,
                    observation_size=observation_size,
                )
            ),
            head=FixedGaussian(Linear(num_joints, dtype=compute_dtype), scale=0.0),
        ),
        critic=Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=mlp(cfg.network.critic, dtype=compute_dtype)
            ),
            head=Linear(1, dtype=compute_dtype),
        ),
        world_model=WorldModel(
            dynamics_fn=FloatingBase(
                normalized(
                    mlp(
                        cfg.network.world_model,
                        6 + num_joints,
                        compute_dtype,
                    )
                ),
                num_joints=num_joints,
                dt=task.control_dt,
                substep=(decimation + 1) / (2 * decimation),
            ),
            reward_fn=reward,
            termination_fn=termination,
            observation_fn=observe,
        ),
        buffer=instantiate(cfg.buffer),
        actor_optimizer=clipped(
            optax.clip(cfg.optimizer.max_actor_grad),
            optax.adam(decayed(algorithm_cfg.policy_updates), b1=0.9, b2=0.95),
        ),
        critic_optimizer=clipped(optax.adam(decayed(num_critic_steps))),
        world_model_optimizer=clipped(
            optax.adam(decayed(algorithm_cfg.world_model_updates))
        ),
        command_key="frame",
        goal=goal,
        critic_goal=critic_goal,
        state_error=task.mismatch,
        advance=advance,
        sensor_noise=task.perturb,
        maximum_value=task.best_reward / (1.0 - cfg.algorithm.gamma),
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
