from omegaconf import MISSING

from ..sections import Environment
from . import environment


def isaaclab(suite, env_id, num_envs=4096):
    return Environment(
        namespace="isaaclab",
        suite=suite,
        env_id=env_id,
        num_envs=num_envs,
        kwargs=dict(num_envs="${environment.num_envs}", device="cuda:0", headless=True),
    )


environment(isaaclab("isaaclab", MISSING), name="isaaclab/isaaclab")
environment(isaaclab("classic", "Isaac-Ant-v0"), name="isaaclab/classic/ant")
environment(isaaclab("classic", "Isaac-Cartpole-v0"), name="isaaclab/classic/cartpole")
environment(isaaclab("classic", "Isaac-Humanoid-v0"), name="isaaclab/classic/humanoid")
environment(
    isaaclab("locomotion", "Isaac-Velocity-Flat-Anymal-C-v0"), name="isaaclab/locomotion/anymal_c_flat"
)
environment(
    isaaclab("locomotion", "Isaac-Velocity-Rough-Anymal-C-v0"), name="isaaclab/locomotion/anymal_c_rough"
)
environment(
    isaaclab("manipulation", "Isaac-Open-Drawer-Franka-v0"), name="isaaclab/manipulation/franka_cabinet"
)
environment(isaaclab("manipulation", "Isaac-Lift-Cube-Franka-v0"), name="isaaclab/manipulation/franka_lift")
environment(isaaclab("manipulation", "Isaac-Reach-Franka-v0"), name="isaaclab/manipulation/franka_reach")
environment(
    isaaclab("manipulation", "Isaac-Repose-Cube-Shadow-Direct-v0", num_envs=8192),
    name="isaaclab/manipulation/shadow_hand",
)
environment(isaaclab("multirotor", "Isaac-Quadcopter-Direct-v0"), name="isaaclab/multirotor/quadcopter")
