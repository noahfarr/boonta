from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .commands import ReturnSampledMotion

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def state(env: ManagerBasedRlEnv) -> torch.Tensor:
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


def context(env: ManagerBasedRlEnv, feet: str, torso: str) -> torch.Tensor:
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


def frame(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(ReturnSampledMotion, env.command_manager.get_term(command_name))
    return command.time_steps[:, None].float()


def executed(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
    return env.action_manager.get_term(action_name).executed
