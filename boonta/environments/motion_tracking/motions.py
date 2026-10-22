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
ANCHOR = "torso_link"


@struct.dataclass(frozen=True)
class Motions:
    root_position: Array
    root_orientation: Array
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
    dot = np.abs(dot)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    sine = np.sin(angle)
    near = sine < 1e-6
    fraction = fraction[:, None]
    left_weight = np.where(near, 1.0 - fraction, np.sin((1.0 - fraction) * angle) / np.where(near, 1.0, sine))
    right_weight = np.where(near, fraction, np.sin(fraction * angle) / np.where(near, 1.0, sine))
    blended = left_weight * left + right_weight * right
    return blended / np.linalg.norm(blended, axis=-1, keepdims=True)


def _resample(frames: np.ndarray, source_rate: float, rate: float) -> np.ndarray:
    count, _ = frames.shape
    times = np.arange(0.0, (count - 1) / source_rate + 1e-9, 1.0 / rate) * source_rate
    lower = np.minimum(np.floor(times).astype(int), count - 2)
    fraction = times - lower
    linear = frames[lower] * (1.0 - fraction[:, None]) + frames[lower + 1] * fraction[:, None]
    xyzw = _slerp(frames[lower, 3:7], frames[lower + 1, 3:7], fraction)
    wxyz = np.concatenate([xyzw[:, 3:], xyzw[:, :3]], axis=-1)
    return np.concatenate([linear[:, :3], wxyz, linear[:, 7:]], axis=-1)


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


def _angular_velocity(orientation: np.ndarray, rate: float) -> np.ndarray:
    following = np.concatenate([orientation[1:], orientation[-1:]], axis=0)
    preceding = np.concatenate([orientation[:1], orientation[:-1]], axis=0)
    steps = np.full(len(orientation), 2.0)
    steps[0] = steps[-1] = 1.0
    conjugate = preceding * np.array([1.0, -1.0, -1.0, -1.0])
    change = _quaternion_product(following, conjugate)
    change = np.where(change[..., :1] < 0.0, -change, change)
    vector = change[..., 1:]
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, change[..., :1])
    axis = vector / np.where(norm < 1e-9, 1.0, norm)
    shape = (len(orientation),) + (1,) * (orientation.ndim - 1)
    return axis * angle * rate / steps.reshape(shape)


def _kinematics(model, qpos: np.ndarray, bodies: list[int]) -> tuple[np.ndarray, np.ndarray]:
    import mujoco

    data = mujoco.MjData(model)
    positions, orientations = [], []
    for frame in qpos:
        data.qpos[:] = frame
        mujoco.mj_kinematics(model, data)
        positions.append(data.xpos[bodies].copy())
        orientations.append(data.xquat[bodies].copy())
    return np.stack(positions), np.stack(orientations)


def _process(model, frames: np.ndarray, source_rate: float, rate: float) -> dict:
    import mujoco

    qpos = _resample(frames, source_rate, rate)
    bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in TRACKED]
    body_position, body_orientation = _kinematics(model, qpos, bodies)
    return {
        "root_position": qpos[:, :3],
        "root_orientation": qpos[:, 3:7],
        "joint_position": qpos[:, 7:],
        "joint_velocity": np.gradient(qpos[:, 7:], 1.0 / rate, axis=0),
        "body_position": body_position,
        "body_orientation": body_orientation,
        "body_linear_velocity": np.gradient(body_position, 1.0 / rate, axis=0),
        "body_angular_velocity": _angular_velocity(body_orientation, rate),
    }


def load(directory: str, model_path: str, source_rate: float = 30.0, rate: float = 50.0) -> Motions:
    import mujoco

    directory = Path(directory)
    cache = directory / f"motions_{rate:g}hz.npz"
    if not cache.exists():
        model = mujoco.MjModel.from_xml_path(model_path)
        clips = [
            _process(model, np.loadtxt(path, delimiter=","), source_rate, rate)
            for path in sorted(directory.glob("*.csv"))
        ]
        lengths = np.array([len(clip["root_position"]) for clip in clips])
        joined = {name: np.concatenate([clip[name] for clip in clips]) for name in clips[0]}
        joined["clip"] = np.repeat(np.arange(len(clips)), lengths)
        joined["clip_start"] = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        joined["clip_end"] = np.cumsum(lengths)
        np.savez(cache, **joined)
    stored = np.load(cache)
    return Motions(**{name: jnp.asarray(stored[name]) for name in stored.files})
