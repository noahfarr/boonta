"""Preprocess G1 CSV motion clips into an NPZ library for SuperMimic."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco


def _resample(path: Path, source_rate: int, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a CSV with mjlab's loader and return MuJoCo-ordered qpos and qvel."""
    from mjlab.scripts.csv_to_npz import MotionLoader

    motion = MotionLoader(
        motion_file=str(path), input_fps=source_rate, output_fps=rate, device="cpu"
    )
    qpos = np.concatenate(
        [
            motion.motion_base_poss.cpu().numpy(),
            motion.motion_base_rots.cpu().numpy(),
            motion.motion_dof_poss.cpu().numpy(),
        ],
        axis=-1,
    )
    qvel = np.concatenate(
        [
            motion.motion_base_lin_vels.cpu().numpy(),
            motion.motion_base_ang_vels.cpu().numpy(),
            motion.motion_dof_vels.cpu().numpy(),
        ],
        axis=-1,
    )
    return qpos, qvel


def _rotation(quaternion: np.ndarray) -> np.ndarray:
    import mujoco

    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def _motion_clip(
    model: mujoco.MjModel, path: Path, source_rate: int, rate: int
) -> dict[str, np.ndarray]:
    """Generate joint and world-space body data for one motion clip."""
    import mujoco

    qpos, qvel = _resample(path, source_rate, rate)
    _validate_motion(path, model, qpos, qvel)

    data = mujoco.MjData(model)
    body_ids = range(1, model.nbody)
    num_frames = len(qpos)
    num_bodies = model.nbody - 1
    frames = {
        "body_pos_w": np.empty((num_frames, num_bodies, 3), dtype=np.float32),
        "body_quat_w": np.empty((num_frames, num_bodies, 4), dtype=np.float32),
        "body_lin_vel_w": np.empty((num_frames, num_bodies, 3), dtype=np.float32),
        "body_ang_vel_w": np.empty((num_frames, num_bodies, 3), dtype=np.float32),
    }
    velocity = np.zeros(6)
    twists = np.empty((num_bodies, 6))
    for frame, (position, rate_of_change) in enumerate(zip(qpos, qvel)):
        data.qpos[:] = position
        data.qvel[:] = rate_of_change
        data.qvel[3:6] = data.qvel[3:6] @ _rotation(position[3:7])
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        for body_index, body in enumerate(body_ids):
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0
            )
            twists[body_index] = velocity
        frames["body_pos_w"][frame] = data.xpos[1:]
        frames["body_quat_w"][frame] = data.xquat[1:]
        frames["body_lin_vel_w"][frame] = twists[:, 3:]
        frames["body_ang_vel_w"][frame] = twists[:, :3]
    return {
        "joint_pos": qpos[:, 7:],
        "joint_vel": qvel[:, 6:],
        **frames,
    }


def _validate_motion(
    path: Path, model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray
) -> None:
    """Raise a useful error if a resampled clip cannot fit the MuJoCo model."""
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(
            f"{path} produced position data with shape {qpos.shape}; expected (*, {model.nq})"
        )
    if qvel.ndim != 2 or qvel.shape[1] != model.nv:
        raise ValueError(
            f"{path} produced velocity data with shape {qvel.shape}; expected (*, {model.nv})"
        )
    if qpos.shape[0] != qvel.shape[0]:
        raise ValueError(
            f"{path} produced mismatched position and velocity frame counts"
        )
    if qpos.shape[0] == 0 or not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError(f"{path} produced empty or non-finite motion data")


def _validate_inputs(
    directory: Path,
    output: Path,
    model_path: Path,
    source_rate: int,
    rate: int,
    overwrite: bool,
) -> list[Path]:
    """Validate conversion arguments and return the sorted source CSV paths."""
    if source_rate <= 0 or rate <= 0:
        raise ValueError("Source and output frame rates must be positive")
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if output.suffix != ".npz":
        raise ValueError(f"Output path must end in .npz: {output}")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output} (pass --overwrite to replace it)"
        )

    paths = sorted(directory.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV motion files found in {directory}")
    return paths


def convert(
    directory: Path,
    output: Path,
    model_path: Path,
    source_rate: int,
    rate: int,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert all CSV clips in a directory into one motion-library archive."""
    import mujoco

    directory = Path(directory)
    output = Path(output)
    model_path = Path(model_path)
    paths = _validate_inputs(
        directory, output, model_path, source_rate, rate, overwrite
    )

    model = mujoco.MjModel.from_xml_path(str(model_path))
    clips = [_motion_clip(model, path, source_rate, rate) for path in paths]
    lengths = np.asarray([len(clip["joint_pos"]) for clip in clips])
    joined = {
        name: np.concatenate([clip[name] for clip in clips]).astype(np.float32)
        for name in clips[0]
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", suffix=".npz", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez(
            temporary_path,
            fps=np.asarray([rate]),
            names=np.asarray([path.stem for path in paths]),
            clip_start=np.concatenate([[0], np.cumsum(lengths)[:-1]]),
            clip_end=np.cumsum(lengths),
            **joined,
        )
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output


def main() -> None:
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Directory containing source CSV motion clips"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output NPZ (defaults to data/motions/mjlab/lafan1_g1/library_<rate>hz.npz)",
    )
    parser.add_argument("--source-rate", type=int, default=30)
    parser.add_argument("--rate", type=int, default=50)
    parser.add_argument("--model", type=Path, default=Path(G1_XML))
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output archive"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    output = args.output or (
        project_root
        / "data"
        / "motions"
        / "mjlab"
        / "lafan1_g1"
        / f"library_{args.rate}hz.npz"
    )
    print(
        convert(
            args.directory,
            output,
            args.model,
            args.source_rate,
            args.rate,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
