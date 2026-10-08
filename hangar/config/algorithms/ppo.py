from boonta.algorithms.ppo import PPOConfig

from ..sections import Optimizer, Rollout
from ..store import fbuilds
from . import algorithm, special

algorithm(
    dict(
        algorithm=fbuilds(
            PPOConfig,
            num_minibatches=16,
            update_epochs=2,
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            gamma=0.99,
            gae_lambda=0.95,
            normalize_advantage=False,
        ),
        rollout=Rollout(num_steps=16),
        environment=dict(num_envs=4096),
        optimizer=Optimizer(lr=2.5e-4, max_grad_norm=0.5),
    ),
    name="ppo",
)

special(
    "ppo",
    "brax",
    total_timesteps=50_000_000,
    algorithm=dict(
        gae_lambda=0.95,
        num_minibatches=32,
        update_epochs=4,
        normalize_advantage=True,
        clip_coefficient=0.2,
        clip_value_loss=True,
        entropy_coefficient=0.0,
    ),
    rollout=dict(num_steps=16),
    environment=dict(num_envs=4096),
)
special("ppo", "craftax", total_timesteps=1_000_000_000)
special(
    "ppo",
    "gymnasium",
    total_timesteps=500_000,
    algorithm=dict(num_minibatches=4, update_epochs=4, normalize_advantage=True),
    rollout=dict(num_steps=128),
    environment=dict(num_envs=16),
    optimizer=dict(lr=3e-3),
)
special(
    "ppo",
    "gymnax/bsuite",
    total_timesteps=50_000_000,
    algorithm=dict(
        gamma=0.995,
        gae_lambda=0.95,
        num_minibatches=8,
        update_epochs=4,
        entropy_coefficient=0.01,
        normalize_advantage=True,
    ),
    rollout=dict(num_steps=32),
    environment=dict(num_envs=1024),
    optimizer=dict(lr=3e-4, max_grad_norm=0.5),
)
special(
    "ppo",
    "gymnax/classic_control",
    total_timesteps=20_000_000,
    algorithm=dict(
        gae_lambda=0.95,
        num_minibatches=8,
        update_epochs=4,
        normalize_advantage=True,
        clip_coefficient=0.2,
        clip_value_loss=True,
        entropy_coefficient=0.0,
    ),
    rollout=dict(num_steps=32),
    environment=dict(num_envs=4096),
)
special("ppo", "gymnax/minatar", total_timesteps=20_000_000)

special(
    "ppo",
    "isaaclab",
    algorithm=dict(
        num_minibatches=4,
        update_epochs=5,
        clip_coefficient=0.2,
        clip_value_loss=True,
        entropy_coefficient=0.005,
        value_coefficient=1.0,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantage=True,
    ),
    rollout=dict(num_steps=24),
    optimizer=dict(lr=1.0e-3, max_grad_norm=1.0),
    total_timesteps=150_000_000,
)

isaaclab = ["ppo/isaaclab", "_self_"]

special(
    "ppo",
    "isaaclab/classic/ant",
    defaults=isaaclab,
    algorithm=dict(entropy_coefficient=0.0),
    rollout=dict(num_steps=32),
    optimizer=dict(lr=5.0e-4),
    total_timesteps=131_000_000,
)
special(
    "ppo",
    "isaaclab/classic/cartpole",
    defaults=isaaclab,
    rollout=dict(num_steps=16),
    total_timesteps=10_000_000,
)
special(
    "ppo",
    "isaaclab/classic/humanoid",
    defaults=isaaclab,
    algorithm=dict(entropy_coefficient=0.0),
    rollout=dict(num_steps=32),
    optimizer=dict(lr=1.0e-4),
    total_timesteps=131_000_000,
)
special("ppo", "isaaclab/locomotion/anymal_c_flat", defaults=isaaclab, total_timesteps=30_000_000)
special("ppo", "isaaclab/locomotion/anymal_c_rough", defaults=isaaclab, total_timesteps=147_000_000)
special(
    "ppo",
    "isaaclab/manipulation/franka_cabinet",
    defaults=isaaclab,
    algorithm=dict(entropy_coefficient=1.0e-3),
    rollout=dict(num_steps=96),
    optimizer=dict(lr=5.0e-4),
    total_timesteps=157_000_000,
)
special(
    "ppo",
    "isaaclab/manipulation/franka_lift",
    defaults=isaaclab,
    algorithm=dict(entropy_coefficient=0.006, gamma=0.98),
    optimizer=dict(lr=1.0e-4),
    total_timesteps=147_000_000,
)
special(
    "ppo",
    "isaaclab/manipulation/franka_reach",
    defaults=isaaclab,
    algorithm=dict(update_epochs=8, entropy_coefficient=0.001),
    total_timesteps=98_000_000,
)
special(
    "ppo",
    "isaaclab/manipulation/shadow_hand",
    defaults=isaaclab,
    rollout=dict(num_steps=16),
    optimizer=dict(lr=5.0e-4),
    total_timesteps=655_000_000,
)
special(
    "ppo",
    "isaaclab/multirotor/quadcopter",
    defaults=isaaclab,
    algorithm=dict(entropy_coefficient=0.0),
    optimizer=dict(lr=5.0e-4),
    total_timesteps=20_000_000,
)

special(
    "ppo",
    "jumanji/sokoban",
    total_timesteps=100_000_000,
    training=dict(num_epochs=50),
    algorithm=dict(
        num_minibatches=8,
        update_epochs=2,
        entropy_coefficient=0.01,
        normalize_advantage=True,
        gamma=0.99,
        gae_lambda=0.95,
    ),
    rollout=dict(num_steps=64),
    environment=dict(num_envs=1024),
    optimizer=dict(lr=3e-4, max_grad_norm=0.5),
)
special(
    "ppo",
    "mujoco_playground",
    total_timesteps=60_000_000,
    algorithm=dict(
        gamma=0.995,
        gae_lambda=0.95,
        num_minibatches=32,
        update_epochs=4,
        normalize_advantage=True,
        clip_coefficient=0.2,
        clip_value_loss=True,
        entropy_coefficient=0.0,
    ),
    rollout=dict(num_steps=16),
    environment=dict(num_envs=4096),
)
special("ppo", "xland_minigrid/minigrid", total_timesteps=20_000_000)
special("ppo", "xland_minigrid/xland", total_timesteps=1_000_000_000)
