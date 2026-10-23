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


def _clip(model, path: Path, source_rate: int, rate: int) -> dict:
    import mujoco
    from mjlab.scripts.csv_to_npz import MotionLoader

    motion = MotionLoader(str(path), source_rate, rate, "cpu")
    qpos = np.concatenate(
        [motion.motion_base_poss.numpy(), motion.motion_base_rots.numpy(), motion.motion_dof_poss.numpy()], axis=-1
    )
    qvel = np.concatenate(
        [motion.motion_base_lin_vels.numpy(), motion.motion_base_ang_vels.numpy(), motion.motion_dof_vels.numpy()],
        axis=-1,
    )
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
