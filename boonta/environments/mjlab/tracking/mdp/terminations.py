from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .commands import ReturnSampledMotion

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def motion_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    command = cast(ReturnSampledMotion, env.command_manager.get_term(command_name))
    return command.time_steps + 1 >= command.clip_end[command.clip[command.time_steps]]
