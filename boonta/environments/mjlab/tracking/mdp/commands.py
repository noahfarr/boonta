import math
from dataclasses import dataclass, fields

import numpy as np
import torch
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg


class ReturnSampledMotion(MotionCommand):
    cfg: "ReturnSampledMotionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        library = np.load(cfg.motion_file)
        lengths = library["clip_end"] - library["clip_start"]
        self.clip_start = torch.as_tensor(library["clip_start"], device=self.device)
        self.clip_end = torch.as_tensor(library["clip_end"], device=self.device)
        self.clip = torch.as_tensor(np.repeat(np.arange(len(lengths)), lengths), device=self.device)
        self.frames_per_bin = int(round(cfg.bin_seconds / env.step_dt))
        self.num_bins = math.ceil(self.motion.time_step_total / self.frames_per_bin)
        self.bin_return = torch.zeros(self.num_bins, device=self.device)
        self.remembered = torch.zeros(self.num_bins, device=self.device)
        self.start_bin = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.returned = torch.zeros(self.num_envs, device=self.device)
        self.discount = torch.ones(self.num_envs, device=self.device)
        self.ceiling = torch.ones(self.num_envs, device=self.device)
        self.fresh = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _tally(self, env_ids):
        ended = env_ids[self.start_bin[env_ids] >= 0]
        if len(ended) == 0 or not hasattr(self._env, "reward_buf"):
            return
        final = self.returned[ended] + self.discount[ended] * self._env.reward_buf[ended]
        bins = self.start_bin[ended]
        totals = torch.zeros_like(self.bin_return).index_add_(0, bins, final / self.ceiling[ended])
        counts = torch.zeros_like(self.bin_return).index_add_(0, bins, torch.ones_like(final))
        visited = counts > 0
        estimate = totals[visited] / counts[visited]
        self.bin_return[visited] += self.cfg.smoothing * (estimate - self.bin_return[visited])

    def _adaptive_sampling(self, env_ids):
        if self.cfg.play:
            self.time_steps[env_ids] = self.clip_start[env_ids % len(self.clip_start)]
            return
        self._tally(env_ids)
        hardest = torch.softmax(-self.cfg.sharpness * self.bin_return, dim=0)
        self.remembered += hardest
        mixture = (
            self.cfg.uniform_ratio / self.num_bins
            + self.cfg.hardest_ratio * hardest
            + self.cfg.remembered_ratio * self.remembered / self.remembered.sum()
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
        gamma = self.cfg.gamma
        self.ceiling[env_ids] = self.cfg.attainable_return * (1.0 - gamma ** remaining.float()) / (1.0 - gamma)
        self.fresh[env_ids] = True

    def _update_command(self, env_ids=None):
        if env_ids is None and hasattr(self._env, "reward_buf"):
            stale = self.fresh
            self.returned += torch.where(stale, 0.0, self.discount * self._env.reward_buf)
            self.discount *= torch.where(stale, 1.0, self.cfg.gamma)
            self.fresh[:] = False
        super()._update_command(env_ids)


@dataclass(kw_only=True)
class ReturnSampledMotionCfg(MotionCommandCfg):
    attainable_return: float = 1.0
    play: bool = False
    gamma: float = 0.99
    bin_seconds: float = 1.0
    uniform_ratio: float = 0.5
    hardest_ratio: float = 0.4
    remembered_ratio: float = 0.1
    sharpness: float = 25.0
    smoothing: float = 0.1

    @classmethod
    def derive(cls, base: MotionCommandCfg, **fields_):
        return cls(**{field.name: getattr(base, field.name) for field in fields(base) if field.init}, **fields_)

    def build(self, env):
        return ReturnSampledMotion(self, env)
