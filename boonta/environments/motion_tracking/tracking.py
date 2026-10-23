import math
from dataclasses import dataclass, fields

import numpy as np

from ..mjlab import Mjlab
from ..wrappers import Batched
from . import motions

BIN_SECONDS = 1.0
UNIFORM, HARDEST, REMEMBERED = 0.5, 0.4, 0.1
SHARPNESS = 25.0
SMOOTHING = 0.1


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


def _motion_end(env, command_name: str):
    command = env.command_manager.get_term(command_name)
    return command.time_steps + 1 >= command.clip_end[command.clip[command.time_steps]]


def _command(base, gamma: float, attainable: float, benchmark: bool):
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
            self.num_bins = math.ceil(self.motion.time_step_total / self.frames_per_bin)
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

            if benchmark:
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
            self.ceiling[env_ids] = attainable * (1.0 - gamma ** remaining.float()) / (1.0 - gamma)
            self.fresh[env_ids] = True

        def _update_command(self, env_ids=None):
            import torch

            if env_ids is None and hasattr(self._env, "reward_buf"):
                stale = self.fresh
                self.returned += torch.where(stale, 0.0, self.discount * self._env.reward_buf)
                self.discount *= torch.where(stale, 1.0, gamma)
                self.fresh[:] = False
            super()._update_command(env_ids)

    @dataclass(kw_only=True)
    class ReturnSampledMotionCfg(MotionCommandCfg):
        def build(self, env):
            return ReturnSampledMotion(self, env)

    return ReturnSampledMotionCfg(**{field.name: getattr(base, field.name) for field in fields(base)})


def _attainable(cfg) -> float:
    return sum(term.weight for term in cfg.rewards.values() if term.weight > 0) * cfg.sim.mujoco.timestep * cfg.decimation


def configure(library: str, num_envs: int, gamma: float, benchmark: bool = False):
    from mjlab.envs.mdp import observations as observed
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

    cfg = unitree_g1_flat_tracking_env_cfg()
    cfg.scene.num_envs = num_envs
    cfg.scene.sensors = ()
    feet = cfg.events["foot_friction"].params["asset_cfg"].geom_names
    torso, *_ = cfg.events["base_com"].params["asset_cfg"].body_names
    del cfg.rewards["self_collisions"]
    cfg.rewards["joint_limit"].weight = -2.0
    attainable = _attainable(cfg)
    motion = cfg.commands["motion"]
    motion.motion_file = str(library)
    motion.anchor_body_name = "pelvis"
    motion.sampling_mode = "adaptive"
    motion.debug_vis = False
    if benchmark:
        motion.pose_range, motion.velocity_range, motion.joint_position_range = {}, {}, (0.0, 0.0)
        del cfg.events["base_com"], cfg.events["encoder_bias"]
        cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
        stored = np.load(library)
        cfg.episode_length_s = float(np.max(stored["clip_end"] - stored["clip_start"])) * cfg.sim.mujoco.timestep * cfg.decimation
    cfg.commands["motion"] = _command(motion, gamma, attainable, benchmark)
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
    }
    return cfg


def make(
    env_id: str,
    directory: str,
    num_envs: int = 1,
    device: str = "cuda:0",
    gamma: float = 0.99,
    benchmark: bool = False,
):
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
    from mjlab.envs import ManagerBasedRlEnv

    library = motions.convert(directory, str(G1_XML))
    cfg = configure(library, num_envs, gamma, benchmark)
    env = Mjlab(ManagerBasedRlEnv(cfg, device=device))
    env.library, env.attainable = library, _attainable(cfg)
    return Batched(env, num_envs=num_envs)


def describe(env):
    import jax.numpy as jnp
    import mujoco
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
    from mujoco import mjx

    from .task import Tracking

    skeleton = mujoco.MjModel.from_xml_path(str(G1_XML))
    skeleton.geom_contype[:] = 0
    skeleton.geom_conaffinity[:] = 0
    tracked = [mujoco.mj_name2id(skeleton, mujoco.mjtObj.mjOBJ_BODY, name) for name in motions.TRACKED]
    robot = env.unwrapped.env.scene["robot"].data
    limits = robot.soft_joint_pos_limits[0].cpu().numpy()
    return Tracking(
        kinematics=mjx.put_model(skeleton),
        motions=motions.load(env.unwrapped.library, tuple(body - 1 for body in tracked)),
        tracked=jnp.asarray(tracked),
        default_pose=jnp.asarray(robot.default_joint_pos[0].cpu().numpy()),
        soft_lower=jnp.asarray(limits[:, 0]),
        soft_upper=jnp.asarray(limits[:, 1]),
    )
