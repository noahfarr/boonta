import math as pymath
from dataclasses import dataclass, fields
from pathlib import Path

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
from boonta.networks import FeatureExtractor, FixedGaussian, FloatingBase, GoalConditioned, Network, WorldModel
from boonta.networks.layers import Normalize
from boonta.utils import Array

TASK = "SuperMimic-Motion-Tracking-Flat-Unitree-G1"
GAMMA = 0.99
BIN_SECONDS = 1.0
UNIFORM, HARDEST, REMEMBERED = 0.5, 0.4, 0.1
SHARPNESS = 25.0
SMOOTHING = 0.1
ACTION_NOISE = 0.1

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

TRACKED = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


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


def _slerp(left: np.ndarray, right: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    dot = np.sum(left * right, axis=-1, keepdims=True)
    right = np.where(dot < 0.0, -right, right)
    angle = np.arccos(np.clip(np.abs(dot), -1.0, 1.0))
    sine = np.sin(angle)
    near = sine < 1e-6
    fraction = fraction[:, None]
    left_weight = np.where(near, 1.0 - fraction, np.sin((1.0 - fraction) * angle) / np.where(near, 1.0, sine))
    right_weight = np.where(near, fraction, np.sin(fraction * angle) / np.where(near, 1.0, sine))
    blended = left_weight * left + right_weight * right
    return blended / np.linalg.norm(blended, axis=-1, keepdims=True)


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def _turning(orientation: np.ndarray, step: float) -> np.ndarray:
    change = _quaternion_product(orientation[2:], orientation[:-2] * np.array([1.0, -1.0, -1.0, -1.0]))
    change = np.where(change[:, :1] < 0.0, -change, change)
    norm = np.linalg.norm(change[:, 1:], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, change[:, :1])
    turning = change[:, 1:] / np.where(norm < 1e-9, 1.0, norm) * angle / (2.0 * step)
    return np.concatenate([turning[:1], turning, turning[-1:]], axis=0)


def _resample(frames: np.ndarray, source_rate: int, rate: int) -> tuple[np.ndarray, np.ndarray]:
    count, _ = frames.shape
    duration = (count - 1) / source_rate
    times = np.arange(0.0, duration, 1.0 / rate, dtype=np.float32)
    phase = times / duration * (count - 1)
    lower = np.floor(phase).astype(int)
    upper = np.minimum(lower + 1, count - 1)
    fraction = phase - lower
    linear = frames[lower] * (1.0 - fraction[:, None]) + frames[upper] * fraction[:, None]
    wxyz = frames[:, [6, 3, 4, 5]]
    orientation = _slerp(wxyz[lower], wxyz[upper], fraction)
    step = 1.0 / rate
    qpos = np.concatenate([linear[:, :3], orientation, linear[:, 7:]], axis=-1)
    qvel = np.concatenate(
        [
            np.gradient(linear[:, :3], step, axis=0),
            _turning(orientation, step),
            np.gradient(linear[:, 7:], step, axis=0),
        ],
        axis=-1,
    )
    return qpos, qvel


def _motion_clip(model, path: Path, source_rate: int, rate: int) -> dict:
    import mujoco

    qpos, qvel = _resample(np.loadtxt(path, delimiter=","), source_rate, rate)
    data = mujoco.MjData(model)
    bodies = range(1, model.nbody)
    frames = {"body_pos_w": [], "body_quat_w": [], "body_lin_vel_w": [], "body_ang_vel_w": []}
    velocity = np.zeros(6)
    for position, rate_of_change in zip(qpos, qvel):
        data.qpos[:] = position
        data.qvel[:] = rate_of_change
        data.qvel[3:6] = data.qvel[3:6] @ _rotation(position[3:7])
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        twists = []
        for body in bodies:
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
            twists.append(velocity.copy())
        twists = np.stack(twists)
        frames["body_pos_w"].append(data.xpos[1:].copy())
        frames["body_quat_w"].append(data.xquat[1:].copy())
        frames["body_lin_vel_w"].append(twists[:, 3:])
        frames["body_ang_vel_w"].append(twists[:, :3])
    return {
        "joint_pos": qpos[:, 7:],
        "joint_vel": qvel[:, 6:],
        **{name: np.stack(values) for name, values in frames.items()},
    }


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    import mujoco

    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def convert_motions(directory: str, model_path: str, source_rate: int = 30, rate: int = 50) -> Path:
    import mujoco

    directory = Path(directory)
    library = directory / f"library_{rate}hz.npz"
    if not library.exists():
        model = mujoco.MjModel.from_xml_path(model_path)
        paths = sorted(directory.glob("*.csv"))
        clips = [_motion_clip(model, path, source_rate, rate) for path in paths]
        lengths = np.array([len(clip["joint_pos"]) for clip in clips])
        joined = {name: np.concatenate([clip[name] for clip in clips]).astype(np.float32) for name in clips[0]}
        np.savez(
            library,
            fps=np.array([rate]),
            names=np.array([path.stem for path in paths]),
            clip_start=np.concatenate([[0], np.cumsum(lengths)[:-1]]),
            clip_end=np.cumsum(lengths),
            **joined,
        )
    return library


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
    """The differentiable tracking task SuperMimic's world model computes on imagined states.

    mujoco/G1-specific, so it lives here in the recipe rather than in ``boonta.networks``
    (pure JAX) or ``boonta.environments`` (adapter only).
    """

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

    def split(self, state: Array) -> tuple[Array, Array, Array]:
        return state[..., :QPOS], state[..., QPOS : QPOS + QVEL], state[..., QPOS + QVEL :]

    def bodies(self, qpos: Array, qvel: Array) -> tuple[Array, Array, Array, Array]:
        from mujoco import mjx
        from mujoco.mjx._src import math

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
        from mujoco.mjx._src import math

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
        from mujoco.mjx._src import math

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
        from mujoco.mjx._src import math

        qpos, qvel, previous = self.split(state)
        bias = context[..., 1 : 1 + NUM_JOINTS]
        gravity = math.rotate(DOWN, math.quat_inv(qpos[3:7]))
        return jnp.concatenate([qpos[7:] + bias - self.default_pose, qvel[6:], qvel[3:6], gravity, previous])

    def perturb(self, key: Array, observation: Array) -> Array:
        return observation + jax.random.uniform(key, jnp.shape(observation), minval=-1.0, maxval=1.0) * NOISE

    def advance(self, frame: Array) -> Array:
        motions = self.motions
        return jnp.minimum(frame + 1, motions.clip_end[motions.clip[frame]] - 1)

    def goal(self, frame: Array) -> Array:
        from mujoco.mjx._src import math

        motions = self.motions
        last = motions.clip_end[motions.clip[frame]] - 1
        frames = jnp.minimum(frame + 1 + jnp.arange(self.goal_frames), last)
        root_position, root_orientation = motions.body_position[:, 0], motions.body_orientation[:, 0]
        inverse = math.quat_inv(_yaw(root_orientation[frame]))

        def describe_frame(index):
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

        return jax.vmap(describe_frame)(frames).reshape(-1)


def _yaw(quaternion: Array) -> Array:
    w, x, y, z = quaternion
    angle = jnp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return jnp.array([jnp.cos(angle / 2.0), 0.0, 0.0, jnp.sin(angle / 2.0)])


def _angle(left: Array, right: Array) -> Array:
    from mujoco.mjx._src import math

    difference = math.quat_mul(math.quat_inv(left), right)
    return 2.0 * jnp.arctan2(jnp.linalg.norm(difference[1:]), jnp.abs(difference[0]))


def describe(env) -> Tracking:
    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
    from mujoco import mjx

    skeleton = mujoco.MjModel.from_xml_path(str(G1_XML))
    skeleton.geom_contype[:] = 0
    skeleton.geom_conaffinity[:] = 0
    tracked = [mujoco.mj_name2id(skeleton, mujoco.mjtObj.mjOBJ_BODY, name) for name in TRACKED]
    robot = env.unwrapped.env.scene["robot"].data
    limits = robot.soft_joint_pos_limits[0].cpu().numpy()
    return Tracking(
        kinematics=mjx.put_model(skeleton),
        motions=load_motions(
            env.unwrapped.env.cfg.commands["motion"].motion_file, tuple(body - 1 for body in tracked)
        ),
        tracked=jnp.asarray(tracked),
        default_pose=jnp.asarray(robot.default_joint_pos[0].cpu().numpy()),
        soft_lower=jnp.asarray(limits[:, 0]),
        soft_upper=jnp.asarray(limits[:, 1]),
    )


def _state(env):
    import torch

    data = env.scene["robot"].data
    return torch.cat(
        [
            data.root_link_pos_w - env.scene.env_origins,
            data.root_link_quat_w,
            data.joint_pos,
            data.root_link_lin_vel_w,
            data.root_link_ang_vel_b,
            data.joint_vel,
            env.action_manager.action,
        ],
        dim=-1,
    )


def _context(env, feet: str, torso: str):
    import torch

    robot = env.scene["robot"]
    foot_ids, _ = robot.find_geoms(feet)
    torso_ids, _ = robot.find_bodies(torso)
    foot = robot.indexing.geom_ids[foot_ids[0]]
    body = robot.indexing.body_ids[torso_ids[0]]
    nominal = torch.as_tensor(env.sim.mj_model.body_ipos[int(body)], device=env.device, dtype=torch.float32)
    return torch.cat(
        [
            env.sim.model.geom_friction[:, foot, :1],
            robot.data.encoder_bias,
            env.sim.model.body_ipos[:, body] - nominal,
        ],
        dim=-1,
    )


def _frame(env, command_name: str):
    command = env.command_manager.get_term(command_name)
    return command.time_steps[:, None].float()


def _executed(env, action_name: str):
    return env.action_manager.get_term(action_name).executed


def _motion_end(env, command_name: str):
    command = env.command_manager.get_term(command_name)
    return command.time_steps + 1 >= command.clip_end[command.clip[command.time_steps]]


def _command(base, attainable_return: float, play: bool):
    from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg

    class ReturnSampledMotion(MotionCommand):
        def __init__(self, cfg, env):
            import torch

            super().__init__(cfg, env)
            library = np.load(cfg.motion_file)
            lengths = library["clip_end"] - library["clip_start"]
            self.clip_start = torch.as_tensor(library["clip_start"], device=self.device)
            self.clip_end = torch.as_tensor(library["clip_end"], device=self.device)
            self.clip = torch.as_tensor(np.repeat(np.arange(len(lengths)), lengths), device=self.device)
            self.frames_per_bin = int(round(BIN_SECONDS / env.step_dt))
            self.num_bins = pymath.ceil(self.motion.time_step_total / self.frames_per_bin)
            self.bin_return = torch.zeros(self.num_bins, device=self.device)
            self.remembered = torch.zeros(self.num_bins, device=self.device)
            self.start_bin = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
            self.returned = torch.zeros(self.num_envs, device=self.device)
            self.discount = torch.ones(self.num_envs, device=self.device)
            self.ceiling = torch.ones(self.num_envs, device=self.device)
            self.fresh = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        def _tally(self, env_ids):
            import torch

            ended = env_ids[self.start_bin[env_ids] >= 0]
            if len(ended) == 0:
                return
            final = self.returned[ended] + self.discount[ended] * self._env.reward_buf[ended]
            bins = self.start_bin[ended]
            totals = torch.zeros_like(self.bin_return).index_add_(0, bins, final / self.ceiling[ended])
            counts = torch.zeros_like(self.bin_return).index_add_(0, bins, torch.ones_like(final))
            visited = counts > 0
            estimate = totals[visited] / counts[visited]
            self.bin_return[visited] += SMOOTHING * (estimate - self.bin_return[visited])

        def _adaptive_sampling(self, env_ids):
            import torch

            if play:
                self.time_steps[env_ids] = self.clip_start[env_ids % len(self.clip_start)]
                return
            self._tally(env_ids)
            hardest = torch.softmax(-SHARPNESS * self.bin_return, dim=0)
            self.remembered += hardest
            mixture = (
                UNIFORM / self.num_bins
                + HARDEST * hardest
                + REMEMBERED * self.remembered / self.remembered.sum()
            )
            bins = torch.multinomial(mixture, len(env_ids), replacement=True)
            offsets = torch.rand(len(env_ids), device=self.device) * self.frames_per_bin
            frames = torch.clamp((bins * self.frames_per_bin + offsets).long(), max=self.motion.time_step_total - 1)
            ends = self.clip_end[self.clip[frames]]
            frames = torch.minimum(frames, ends - 3)
            remaining = torch.clamp(torch.minimum(ends - 2 - frames, torch.tensor(self._env.max_episode_length)), min=1)
            self.time_steps[env_ids] = frames
            self.start_bin[env_ids] = bins
            self.returned[env_ids] = 0.0
            self.discount[env_ids] = 1.0
            self.ceiling[env_ids] = attainable_return * (1.0 - GAMMA ** remaining.float()) / (1.0 - GAMMA)
            self.fresh[env_ids] = True

        def _update_command(self, env_ids=None):
            import torch

            if env_ids is None and hasattr(self._env, "reward_buf"):
                stale = self.fresh
                self.returned += torch.where(stale, 0.0, self.discount * self._env.reward_buf)
                self.discount *= torch.where(stale, 1.0, GAMMA)
                self.fresh[:] = False
            super()._update_command(env_ids)

    @dataclass(kw_only=True)
    class ReturnSampledMotionCfg(MotionCommandCfg):
        def build(self, env):
            return ReturnSampledMotion(self, env)

    return ReturnSampledMotionCfg(**{field.name: getattr(base, field.name) for field in fields(base)})


def _action(base, noise: float):
    from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

    class NoisyJointPositionAction(JointPositionAction):
        def __init__(self, cfg, env):
            import torch

            super().__init__(cfg, env)
            self.executed = torch.zeros_like(self._raw_actions)

        def process_actions(self, actions):
            import torch

            self.executed = actions + noise * torch.randn_like(actions)
            super().process_actions(self.executed)

    @dataclass(kw_only=True)
    class NoisyJointPositionActionCfg(JointPositionActionCfg):
        def build(self, env):
            return NoisyJointPositionAction(self, env)

    return NoisyJointPositionActionCfg(
        **{field.name: getattr(base, field.name) for field in fields(base) if field.init}
    )


def attainable(cfg) -> float:
    weights = sum(term.weight for term in cfg.rewards.values() if term.weight > 0)
    return weights * cfg.sim.mujoco.timestep * cfg.decimation


def _configure(play: bool):
    from mjlab.envs.mdp import observations as observed
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

    cfg = unitree_g1_flat_tracking_env_cfg()
    cfg.scene.sensors = ()
    feet = cfg.events["foot_friction"].params["asset_cfg"].geom_names
    torso, *_ = cfg.events["base_com"].params["asset_cfg"].body_names
    del cfg.rewards["self_collisions"]
    cfg.rewards["joint_limit"].weight = -2.0
    motion = cfg.commands["motion"]
    motion.anchor_body_name = "pelvis"
    motion.sampling_mode = "adaptive"
    motion.debug_vis = False
    cfg.actions["joint_pos"] = _action(cfg.actions["joint_pos"], 0.0 if play else ACTION_NOISE)
    cfg.events["foot_friction"].mode = "reset"
    cfg.events["encoder_bias"].mode = "reset"
    if play:
        motion.pose_range, motion.velocity_range, motion.joint_position_range = {}, {}, (0.0, 0.0)
        del cfg.events["base_com"], cfg.events["encoder_bias"]
        cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
        cfg.episode_length_s = 1e9
    cfg.commands["motion"] = _command(motion, attainable(cfg), play)
    del cfg.terminations["ee_body_pos"]
    cfg.terminations["motion_end"] = TerminationTermCfg(
        func=_motion_end, params={"command_name": "motion"}, time_out=True
    )
    del cfg.events["push_robot"]

    def group(terms):
        return ObservationGroupCfg(terms=terms, concatenate_terms=True, enable_corruption=False)

    cfg.observations = {
        "observation": group(
            {
                "joint_pos": ObservationTermCfg(func=observed.joint_pos_rel, params={"biased": True}),
                "joint_vel": ObservationTermCfg(func=observed.joint_vel_rel),
                "base_ang_vel": ObservationTermCfg(
                    func=observed.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
                ),
                "projected_gravity": ObservationTermCfg(func=observed.projected_gravity),
                "actions": ObservationTermCfg(func=observed.last_action),
            }
        ),
        "state": group({"state": ObservationTermCfg(func=_state)}),
        "context": group(
            {
                "context": ObservationTermCfg(
                    func=_context,
                    params={"feet": feet, "torso": torso},
                )
            }
        ),
        "frame": group({"frame": ObservationTermCfg(func=_frame, params={"command_name": "motion"})}),
        "executed": group({"executed": ObservationTermCfg(func=_executed, params={"action_name": "joint_pos"})}),
    }
    return cfg


def register():
    from mjlab.tasks.registry import list_tasks, register_mjlab_task
    from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

    if TASK not in list_tasks():
        register_mjlab_task(
            task_id=TASK,
            env_cfg=_configure(play=False),
            play_env_cfg=_configure(play=True),
            rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
        )


def mlp(sizes, outputs=None):
    layers = [layer for size in sizes for layer in (nn.Dense(size), nn.elu)]
    return nn.Sequential(layers + ([nn.Dense(outputs)] if outputs else []))


def normalized(module):
    return nn.Sequential([Normalize(), module])


def make(cfg):
    register()
    env = environments.make(**cfg.environment)
    task = describe(env)
    goal_size, *_ = jnp.shape(task.goal(jnp.int32(0)))

    def frame(command):
        return command[..., 0].astype(jnp.int32)

    def reward(state, following, command, **_):
        return jax.vmap(task.reward)(state, following, frame(command))

    def goal(command):
        return jax.vmap(task.goal)(frame(command))

    def advance(command):
        return jax.vmap(task.advance)(frame(command))[..., None].astype(command.dtype)

    def observe(state, context, **_):
        return jax.vmap(task.observe)(state, context)

    def clipped(*transforms):
        return optax.chain(optax.clip_by_global_norm(cfg.optimizer.max_grad_norm), *transforms)

    def decayed(steps_per_iteration):
        return optax.cosine_decay_schedule(
            cfg.optimizer.lr,
            decay_steps=cfg.optimizer.decay_iterations * steps_per_iteration,
            alpha=cfg.optimizer.final_scale,
        )

    algorithm_cfg = cfg.algorithm
    critic_steps = algorithm_cfg.policy_updates * algorithm_cfg.critic_epochs * algorithm_cfg.critic_minibatches

    algorithm = SuperMimic(
        cfg=instantiate(cfg.algorithm),
        actor=Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=normalized(
                    GoalConditioned(
                        encoder=mlp(cfg.network.encoder, cfg.network.latent),
                        decoder=mlp(cfg.network.decoder),
                        goal_size=goal_size,
                    )
                )
            ),
            head=FixedGaussian(nn.Dense(NUM_JOINTS), scale=0.0),
        ),
        critic=Network(
            feature_extractor=FeatureExtractor(observation_extractor=normalized(mlp(cfg.network.critic))),
            head=nn.Dense(1),
        ),
        world_model=WorldModel(
            dynamics_fn=FloatingBase(
                normalized(mlp(cfg.network.world_model, 12 + 2 * NUM_JOINTS)), num_joints=NUM_JOINTS
            ),
            reward_fn=reward,
            termination_fn=nn.Sequential(
                [
                    lambda state, goal, **_: jnp.concatenate([state, goal], axis=-1),
                    normalized(mlp(cfg.network.heads, 1)),
                ]
            ),
            observation_fn=observe,
            goal=goal,
            advance=advance,
        ),
        buffer=instantiate(cfg.buffer),
        actor_optimizer=clipped(
            optax.clip(cfg.optimizer.max_actor_grad),
            optax.adam(decayed(algorithm_cfg.policy_updates), b1=0.9, b2=0.95),
        ),
        critic_optimizer=clipped(optax.adam(decayed(critic_steps))),
        world_model_optimizer=clipped(optax.adam(decayed(algorithm_cfg.world_model_updates))),
        command_key="frame",
        sensor_noise=task.perturb,
    )

    algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
    env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
    return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
