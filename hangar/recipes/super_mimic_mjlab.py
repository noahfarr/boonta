from dataclasses import dataclass

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
            math.quat_mul(before.orientation, math.quat_inv(reference_root_orientation))
        )
        origin = reference_root_position.at[:2].set(before.position[:2])
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

    def goal(self, frame: Array) -> Array:
        from mujoco.mjx._src import math

        motions = self.motions
        frames = jnp.minimum(
            frame + 1 + jnp.arange(self.goal_frames), motions.last(frame)
        )
        inverse = math.quat_inv(_yaw(motions.root_orientation[frame]))

        def describe_frame(index):
            return jnp.concatenate(
                [
                    motions.joint_position[index],
                    motions.joint_velocity[index],
                    math.rotate(
                        motions.root_position[index] - motions.root_position[frame],
                        inverse,
                    ),
                    _columns(math.quat_mul(inverse, motions.root_orientation[index])),
                    math.rotate(motions.root_linear_velocity[index], inverse),
                    math.rotate(motions.root_angular_velocity[index], inverse),
                ]
            )

        return jax.vmap(describe_frame)(frames).reshape(-1)


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

    def __call__(self, x: Array) -> Array:
        latent = jnp.tanh(self.encoder(x[..., -self.goal_size :]))
        return self.decoder(
            jnp.concatenate([latent, x[..., : -self.goal_size]], axis=-1)
        )


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

    def __call__(self, state: Array, action: Array, context: Array, **_) -> Array:
        n = self.num_joints
        *batch, _ = jnp.shape(state)
        position, orientation, joints = (
            state[..., :3],
            state[..., 3:7],
            state[..., 7 : 7 + n],
        )
        velocity, turning = state[..., 7 + n : 10 + n], state[..., 10 + n : 13 + n]
        joint_velocity = state[..., 13 + n : 13 + 2 * n]
        previous = state[..., 13 + 2 * n :]
        heading = _matrix(_yaw(orientation))
        tilt = jnp.einsum("...ji,...jk->...ik", heading, _matrix(orientation))[
            ..., :, :2
        ]
        features = jnp.concatenate(
            [
                position[..., 2:],
                jnp.reshape(tilt, (*batch, 6)),
                jnp.einsum("...ji,...j->...i", heading, velocity),
                turning,
                joints,
                joint_velocity,
                previous,
                action,
                context,
            ],
            axis=-1,
        )
        change = self.network(features)
        shift, rotation, acceleration, spin = (
            change[..., 3 * i : 3 * (i + 1)] for i in range(4)
        )
        joint_change, joint_acceleration = (
            change[..., 12 : 12 + n],
            change[..., 12 + n : 12 + 2 * n],
        )
        following = _quaternion_product(orientation, _exponential(rotation))
        following = following / jnp.linalg.norm(following, axis=-1, keepdims=True)
        return jnp.concatenate(
            [
                position + _apply(heading, shift),
                following,
                joints + joint_change,
                velocity + _apply(heading, acceleration),
                turning + spin,
                joint_velocity + joint_acceleration,
                action,
            ],
            axis=-1,
        )


def mlp(sizes, outputs=None, dtype=None):
    def dense(size):
        return nn.Dense(size, dtype=dtype, param_dtype=jnp.float32)

    layers = [layer for size in sizes for layer in (dense(size), nn.elu)]
    if outputs:
        layers.append(dense(outputs))
    layers.append(lambda x: x.astype(jnp.float32))
    return nn.Sequential(layers)


def normalized(module):
    return nn.Sequential([Normalize(), module])


def make(cfg):
    import boonta.environments.mjlab.tracking.config.g1  # noqa: F401

    env = environments.make(**cfg.environment)
    task = describe(env)
    goal_size, *_ = jnp.shape(task.goal(jnp.int32(0)))
    num_joints = task.num_joints
    compute_dtype = jnp.dtype(cfg.network.dtype) if cfg.network.dtype else None

    def frame(command):
        return command[..., 0].astype(jnp.int32)

    def reward(state, next_state, command, **_):
        return jax.vmap(task.reward)(state, next_state, frame(command))

    def goal(command):
        return jax.vmap(task.goal)(frame(command))

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
                observation_extractor=normalized(
                    GoalConditioned(
                        encoder=mlp(
                            cfg.network.encoder, cfg.network.latent, compute_dtype
                        ),
                        decoder=mlp(cfg.network.decoder, dtype=compute_dtype),
                        goal_size=goal_size,
                    )
                )
            ),
            head=FixedGaussian(
                nn.Dense(num_joints, dtype=compute_dtype, param_dtype=jnp.float32),
                scale=0.0,
            ),
        ),
        critic=Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=normalized(
                    mlp(cfg.network.critic, dtype=compute_dtype)
                )
            ),
            head=nn.Dense(1, dtype=compute_dtype, param_dtype=jnp.float32),
        ),
        world_model=WorldModel(
            dynamics_fn=FloatingBase(
                normalized(
                    mlp(
                        cfg.network.world_model,
                        12 + 2 * num_joints,
                        compute_dtype,
                    )
                ),
                num_joints=num_joints,
            ),
            reward_fn=reward,
            termination_fn=nn.Sequential(
                [
                    lambda state, goal, **_: jnp.concatenate([state, goal], axis=-1),
                    normalized(mlp(cfg.network.heads, 1, compute_dtype)),
                ]
            ),
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
        advance=advance,
        sensor_noise=task.perturb,
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
