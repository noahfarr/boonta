from ... import mdp

ACTION_NOISE = 0.1


def unitree_g1_flat_tracking_env_cfg(play: bool = False):
    from mjlab.envs.mdp import observations as observed
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.tasks.tracking.config.g1 import env_cfgs as base

    cfg = base.unitree_g1_flat_tracking_env_cfg()
    cfg.scene.sensors = ()
    feet = cfg.events["foot_friction"].params["asset_cfg"].geom_names
    torso, *_ = cfg.events["base_com"].params["asset_cfg"].body_names
    del cfg.rewards["self_collisions"]
    cfg.rewards["joint_limit"].weight = -2.0
    motion = cfg.commands["motion"]
    motion.anchor_body_name = "pelvis"
    motion.sampling_mode = "adaptive"
    motion.debug_vis = False
    cfg.actions["joint_pos"] = mdp.NoisyJointPositionActionCfg.derive(
        cfg.actions["joint_pos"], noise=0.0 if play else ACTION_NOISE
    )
    cfg.events["foot_friction"].mode = "reset"
    cfg.events["encoder_bias"].mode = "reset"
    if play:
        motion.pose_range, motion.velocity_range, motion.joint_position_range = (
            {},
            {},
            (0.0, 0.0),
        )
        del cfg.events["base_com"], cfg.events["encoder_bias"]
        cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
        cfg.episode_length_s = 1e9
    cfg.commands["motion"] = mdp.ReturnSampledMotionCfg.derive(
        motion, attainable_return=mdp.attainable(cfg), play=play
    )
    del cfg.terminations["ee_body_pos"]
    cfg.terminations["motion_end"] = TerminationTermCfg(
        func=mdp.motion_end, params={"command_name": "motion"}, time_out=True
    )
    del cfg.events["push_robot"]

    def group(terms):
        return ObservationGroupCfg(
            terms=terms, concatenate_terms=True, enable_corruption=False
        )

    cfg.observations = {
        "observation": group(
            {
                "joint_pos": ObservationTermCfg(
                    func=observed.joint_pos_rel, params={"biased": True}
                ),
                "joint_vel": ObservationTermCfg(func=observed.joint_vel_rel),
                "base_ang_vel": ObservationTermCfg(
                    func=observed.builtin_sensor,
                    params={"sensor_name": "robot/imu_ang_vel"},
                ),
                "projected_gravity": ObservationTermCfg(
                    func=observed.projected_gravity
                ),
                "actions": ObservationTermCfg(func=observed.last_action),
            }
        ),
        "state": group({"state": ObservationTermCfg(func=mdp.state)}),
        "context": group(
            {
                "context": ObservationTermCfg(
                    func=mdp.context,
                    params={"feet": feet, "torso": torso},
                )
            }
        ),
        "frame": group(
            {
                "frame": ObservationTermCfg(
                    func=mdp.frame, params={"command_name": "motion"}
                )
            }
        ),
        "executed": group(
            {
                "executed": ObservationTermCfg(
                    func=mdp.executed, params={"action_name": "joint_pos"}
                )
            }
        ),
    }
    return cfg
