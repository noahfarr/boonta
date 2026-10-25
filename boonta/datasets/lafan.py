from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import struct

from boonta.utils import Array

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


def _product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
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
    change = _product(orientation[2:], orientation[:-2] * np.array([1.0, -1.0, -1.0, -1.0]))
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


def _clip(model, path: Path, source_rate: int, rate: int) -> dict:
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


def convert(directory: str, model_path: str, source_rate: int = 30, rate: int = 50) -> Path:
    import mujoco

    directory = Path(directory)
    library = directory / f"library_{rate}hz.npz"
    if not library.exists():
        model = mujoco.MjModel.from_xml_path(model_path)
        paths = sorted(directory.glob("*.csv"))
        clips = [_clip(model, path, source_rate, rate) for path in paths]
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


def load(library: str, bodies: tuple[int, ...]) -> Motions:
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
