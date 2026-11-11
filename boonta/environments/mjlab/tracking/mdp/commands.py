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
        self.value_ceiling = cfg.attainable_return / (1.0 - cfg.gamma)
        self.bin_value = torch.full((self.num_bins,), self.value_ceiling, device=self.device)
        self.bin_samples = torch.zeros(self.num_bins, device=self.device)
        self.effort = torch.zeros(self.num_bins, dtype=torch.float64, device=self.device)
        self.distribution = torch.full((self.num_bins,), 1.0 / self.num_bins, device=self.device)
        shape = (self.num_envs, cfg.horizon)
        self.recorded_frames = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.recorded_rewards = torch.zeros(shape, device=self.device)
        self.recorded_terminated = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.recorded_clip_end = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.recorded_timed_out = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.column = 0
        self.departed_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.departed_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.recorded_step = -1

    def _bins(self, frames):
        return torch.clamp(frames // self.frames_per_bin, max=self.num_bins - 1)

    def _adaptive_sampling(self, env_ids):
        if self.cfg.play:
            self.time_steps[env_ids] = self.clip_start[env_ids % len(self.clip_start)]
            return
        self.departed_frame[env_ids] = self.time_steps[env_ids]
        self.departed_step[env_ids] = self._env.common_step_counter
        bins = torch.multinomial(self.distribution, len(env_ids), replacement=True)
        offsets = torch.rand(len(env_ids), device=self.device) * self.frames_per_bin
        frames = torch.clamp((bins * self.frames_per_bin + offsets).long(), max=self.motion.time_step_total - 1)
        self.time_steps[env_ids] = torch.minimum(frames, self.clip_end[self.clip[frames]] - 3)

    def _record(self):
        manager = self._env.termination_manager
        departed = self.departed_step == self._env.common_step_counter
        truncated = manager.time_outs.bool()
        clip_end = manager.get_term("motion_end").bool()
        column = self.column
        self.recorded_frames[:, column] = torch.where(departed, self.departed_frame, self.time_steps)
        self.recorded_rewards[:, column] = self._env.reward_buf
        self.recorded_terminated[:, column] = manager.terminated.bool() & ~truncated
        self.recorded_clip_end[:, column] = clip_end
        self.recorded_timed_out[:, column] = truncated & ~clip_end
        self.column += 1
        if self.column < self.cfg.horizon:
            return
        self.column = 0
        self._estimate()
        self.distribution = self._distribute()

    def _dones(self):
        return self.recorded_terminated | self.recorded_clip_end | self.recorded_timed_out

    def _returns(self):
        gamma = self.cfg.gamma
        following = torch.clamp(self.recorded_frames + 1, max=self.motion.time_step_total - 1)
        next_value = torch.where(
            self.recorded_terminated,
            0.0,
            torch.where(self.recorded_clip_end, self.bin_value.mean(), self.bin_value[self._bins(following)]),
        )
        dones = self._dones()
        returns = torch.empty_like(self.recorded_rewards)
        next_return = next_value[:, -1]
        for step in reversed(range(self.cfg.horizon)):
            next_return = torch.where(dones[:, step], next_value[:, step], next_return)
            returns[:, step] = self.recorded_rewards[:, step] + gamma * next_return
            next_return = returns[:, step]
        return returns

    def _estimate(self):
        returns = self._returns()
        dones = self._dones().long()
        num_envs, horizon = dones.shape
        episode = torch.cumsum(dones, dim=1) - dones
        envs = torch.arange(num_envs, device=self.device).unsqueeze(1)
        key = (envs * horizon + episode) * self.num_bins + self._bins(self.recorded_frames)
        keys, inverse, visits = torch.unique(key.reshape(-1), return_inverse=True, return_counts=True)
        samples = torch.zeros(keys.numel(), device=self.device).scatter_add_(0, inverse, returns.reshape(-1))
        samples /= visits
        bins = keys % self.num_bins
        fresh = torch.zeros_like(self.bin_samples).scatter_add_(0, bins, torch.ones_like(samples))
        total = torch.zeros_like(self.bin_value).scatter_add_(0, bins, samples)
        self.bin_samples = self.bin_samples.clamp(max=self.cfg.memory) + fresh
        self.bin_value = torch.where(
            fresh > 0, self.bin_value + (total - fresh * self.bin_value) / self.bin_samples, self.bin_value
        )

    def _distribute(self):
        cfg = self.cfg
        focus = torch.softmax(-cfg.steepness * self.bin_value / self.value_ceiling, dim=0)
        self.effort += focus.double().square()
        keep = (self.effort / self.effort.sum()).float()
        return cfg.explore / self.num_bins + (1.0 - cfg.explore - cfg.keep) * focus + cfg.keep * keep

    def _update_command(self, env_ids=None):
        counter = self._env.common_step_counter
        if not self.cfg.play and hasattr(self._env, "reward_buf") and counter != self.recorded_step:
            self.recorded_step = counter
            self._record()
        super()._update_command(env_ids)


@dataclass(kw_only=True)
class ReturnSampledMotionCfg(MotionCommandCfg):
    attainable_return: float = 1.0
    play: bool = False
    gamma: float = 0.99
    bin_seconds: float = 1.0
    horizon: int = 20
    memory: int = 256
    steepness: float = 25.0
    explore: float = 0.5
    keep: float = 0.1

    @classmethod
    def derive(cls, base: MotionCommandCfg, **fields_):
        return cls(**{field.name: getattr(base, field.name) for field in fields(base) if field.init}, **fields_)

    def build(self, env):
        return ReturnSampledMotion(self, env)
