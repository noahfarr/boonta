from boonta.artisans import Gauntlet
from boonta.environments.connectx import Negamax, uniform

from .store import builds, fbuilds, place


def lookup(table, algorithm, environment):
    parts = environment.split("/")
    while parts:
        path = "/".join(parts)
        if (algorithm, path) in table:
            return f"{algorithm}/{path}"
        parts.pop()
    return algorithm


board = dict(
    rows="${environment.kwargs.rows}",
    columns="${environment.kwargs.columns}",
    inarow="${environment.kwargs.inarow}",
)

connectx = dict(
    total_timesteps=20_000_000,
    environment=dict(num_envs=1024),
    rollout=dict(num_steps=32),
    algorithm=dict(
        num_minibatches=8, update_epochs=2, gamma=1.0, gae_lambda=0.95, entropy_coefficient=0.01
    ),
    optimizer=dict(lr=3e-4, max_grad_norm=0.5),
    network=dict(features=256, layers=2),
    score="gauntlet/mean",
    artisans=dict(
        gauntlet=fbuilds(
            Gauntlet,
            opponents=dict(
                random=builds(uniform, zen_partial=True),
                **{f"negamax_{depth}": builds(Negamax, depth=depth, **board) for depth in (1, 2, 4, 6)},
            ),
        )
    ),
)

isaaclab = ["ppo/isaaclab", "_self_"]

table = {
    ("bc", "brax/mujoco"): dict(environment=dict(kwargs=dict(backend="mjx"))),
    ("dqn", "gymnax/minatar"): dict(total_timesteps=10_000_000),
    ("grpo", "gymnax/minatar"): dict(total_timesteps=50_000_000),
    ("ippo", "connectx"): connectx,
    ("ippo", "jaxmarl"): dict(total_timesteps=20_000_000),
    ("ippo", "mapox"): dict(
        total_timesteps=20_000_000,
        environment=dict(num_envs=2048),
        rollout=dict(num_steps=16),
        algorithm=dict(
            num_minibatches=8, update_epochs=2, gamma=0.99, gae_lambda=0.95, entropy_coefficient=0.01
        ),
        optimizer=dict(lr=3e-4, max_grad_norm=0.5),
    ),
    ("iql", "brax/mujoco"): dict(environment=dict(kwargs=dict(backend="mjx"))),
    ("mappo", "jaxmarl"): dict(total_timesteps=20_000_000),
    ("mmd", "connectx"): connectx,
    ("ppo", "brax"): dict(
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
    ),
    ("ppo", "craftax"): dict(total_timesteps=1_000_000_000),
    ("ppo", "gymnasium"): dict(
        total_timesteps=500_000,
        algorithm=dict(num_minibatches=4, update_epochs=4, normalize_advantage=True),
        rollout=dict(num_steps=128),
        environment=dict(num_envs=16),
        optimizer=dict(lr=3e-3),
    ),
    ("ppo", "gymnax/bsuite"): dict(
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
    ),
    ("ppo", "gymnax/classic_control"): dict(
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
    ),
    ("ppo", "gymnax/minatar"): dict(total_timesteps=20_000_000),
    ("ppo", "isaaclab"): dict(
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
    ),
    ("ppo", "isaaclab/classic/ant"): dict(
        defaults=isaaclab,
        algorithm=dict(entropy_coefficient=0.0),
        rollout=dict(num_steps=32),
        optimizer=dict(lr=5.0e-4),
        total_timesteps=131_000_000,
    ),
    ("ppo", "isaaclab/classic/cartpole"): dict(
        defaults=isaaclab,
        rollout=dict(num_steps=16),
        total_timesteps=10_000_000,
    ),
    ("ppo", "isaaclab/classic/humanoid"): dict(
        defaults=isaaclab,
        algorithm=dict(entropy_coefficient=0.0),
        rollout=dict(num_steps=32),
        optimizer=dict(lr=1.0e-4),
        total_timesteps=131_000_000,
    ),
    ("ppo", "isaaclab/locomotion/anymal_c_flat"): dict(defaults=isaaclab, total_timesteps=30_000_000),
    ("ppo", "isaaclab/locomotion/anymal_c_rough"): dict(defaults=isaaclab, total_timesteps=147_000_000),
    ("ppo", "isaaclab/manipulation/franka_cabinet"): dict(
        defaults=isaaclab,
        algorithm=dict(entropy_coefficient=1.0e-3),
        rollout=dict(num_steps=96),
        optimizer=dict(lr=5.0e-4),
        total_timesteps=157_000_000,
    ),
    ("ppo", "isaaclab/manipulation/franka_lift"): dict(
        defaults=isaaclab,
        algorithm=dict(entropy_coefficient=0.006, gamma=0.98),
        optimizer=dict(lr=1.0e-4),
        total_timesteps=147_000_000,
    ),
    ("ppo", "isaaclab/manipulation/franka_reach"): dict(
        defaults=isaaclab,
        algorithm=dict(update_epochs=8, entropy_coefficient=0.001),
        total_timesteps=98_000_000,
    ),
    ("ppo", "isaaclab/manipulation/shadow_hand"): dict(
        defaults=isaaclab,
        rollout=dict(num_steps=16),
        optimizer=dict(lr=5.0e-4),
        total_timesteps=655_000_000,
    ),
    ("ppo", "isaaclab/multirotor/quadcopter"): dict(
        defaults=isaaclab,
        algorithm=dict(entropy_coefficient=0.0),
        optimizer=dict(lr=5.0e-4),
        total_timesteps=20_000_000,
    ),
    ("ppo", "jumanji/sokoban"): dict(
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
    ),
    ("ppo", "mujoco_playground"): dict(
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
    ),
    ("ppo", "xland_minigrid/minigrid"): dict(total_timesteps=20_000_000),
    ("ppo", "xland_minigrid/xland"): dict(total_timesteps=1_000_000_000),
    ("pqn", "gymnax/minatar"): dict(
        total_timesteps=80_000_000,
        algorithm=dict(num_minibatches=16),
        environment=dict(num_envs=4096),
    ),
    ("recurrent_bc", "kinetix"): dict(
        defaults=[{"/dataset": "kinetix/offline_m"}],
        algorithm=dict(batch_size="${dataset.kwargs.batch_size}"),
        network=dict(features=128, num_layers=2, num_heads=8, actor_depth=5, actor_width=128),
        podracer=dict(config=dict(batch_shape=["${dataset.kwargs.batch_size}", 256])),
    ),
    ("recurrent_dqn", "gymnax/minatar"): dict(total_timesteps=10_000_000),
    ("recurrent_grpo", "wordle"): dict(
        environment=dict(num_envs=32),
        network=dict(repo_id="Qwen/Qwen3-0.6B-Base", dtype="bfloat16", param_dtype="bfloat16"),
    ),
    ("recurrent_ppo", "gymnax/minatar"): dict(
        total_timesteps=10_000_000,
        environment=dict(num_envs=512),
        rollout=dict(num_steps=16),
    ),
    ("recurrent_ppo", "peanut_gb/pokemon_red"): dict(
        total_timesteps=1_000_000_000,
        training=dict(num_epochs=635),
        environment=dict(num_envs=1024),
        rollout=dict(num_steps=128),
        cell=dict(features=256, dtype="bfloat16"),
        network=dict(dtype="bfloat16"),
        algorithm=dict(
            num_minibatches=16,
            update_epochs=1,
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=2.5e-4,
            value_coefficient=0.5,
            gamma=0.999,
            gae_lambda=0.95,
            normalize_advantage=False,
        ),
        optimizer=dict(
            lr=5e-3, anneal=True, min_lr_ratio=0.0, beta=0.95, weight_decay=0.01, max_grad_norm=0.5
        ),
    ),
    ("recurrent_ppo", "wordle"): dict(
        environment=dict(num_envs=32),
        algorithm=dict(num_minibatches=16),
        network=dict(repo_id="Qwen/Qwen3-0.6B-Base", dtype="bfloat16", param_dtype="bfloat16"),
    ),
    ("recurrent_pqn", "gymnax/minatar"): dict(total_timesteps=10_000_000),
    ("recurrent_pupo", "ale/montezuma"): dict(
        total_timesteps=50_000_000,
        training=dict(num_epochs=50),
        environment=dict(num_envs=128),
        rollout=dict(num_steps=64),
        num_layers=2,
        cell=dict(features=256, dtype="bfloat16"),
        network=dict(dtype="bfloat16"),
        algorithm=dict(
            num_minibatches=8,
            update_epochs=2,
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=0.01,
            value_coefficient=2.0,
            gamma=0.999,
            gae_lambda=0.95,
        ),
        optimizer=dict(name="muon", lr=0.005, anneal=True, min_lr_ratio=0.0, max_grad_norm=1.5),
    ),
    ("recurrent_pupo", "craftax"): dict(
        total_timesteps=100_000_000,
        training=dict(num_epochs=20),
        environment=dict(num_envs=1024, reset_ratio=64),
        rollout=dict(num_steps=64),
        num_layers=2,
        cell=dict(features=256, dtype="bfloat16"),
        network=dict(dtype="bfloat16"),
        algorithm=dict(
            num_minibatches=16,
            update_epochs=2,
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=0.01,
            value_coefficient=2.0,
            gamma=0.99,
            gae_lambda=0.90,
        ),
        optimizer=dict(name="muon", lr=0.005, anneal=True, min_lr_ratio=0.0, max_grad_norm=1.5),
    ),
    ("recurrent_pupo", "gymnax/minatar"): dict(
        total_timesteps=104_857_600,
        training=dict(num_epochs=20),
        environment=dict(
            num_envs=8192,
            sticky_action_prob=0.1,
            kwargs=dict(use_minimal_action_set=False, params=dict(max_steps_in_episode=100000)),
        ),
        rollout=dict(num_steps=64),
        num_layers=2,
        cell=dict(features=256, dtype="bfloat16"),
        network=dict(encoder="linear", dtype="bfloat16"),
        algorithm=dict(
            num_minibatches=16,
            update_epochs=2,
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=0.01,
            value_coefficient=2.0,
            gamma=0.99,
            gae_lambda=0.90,
        ),
        optimizer=dict(name="muon", lr=0.015, anneal=True, max_grad_norm=1.5),
    ),
    ("recurrent_pupo", "jumanji/sokoban"): dict(
        total_timesteps=100_000_000,
        training=dict(num_epochs=50),
        environment=dict(num_envs=4096),
        rollout=dict(num_steps=64),
        num_layers=2,
        cell=dict(features=256, dtype="bfloat16"),
        network=dict(dtype="bfloat16"),
        algorithm=dict(
            num_minibatches=16,
            update_epochs=2,
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=0.01,
            value_coefficient=2.0,
            gamma=0.99,
            gae_lambda=0.90,
        ),
        optimizer=dict(name="muon", lr=0.005, anneal=True, min_lr_ratio=0.0, max_grad_norm=1.5),
    ),
    ("recurrent_pupo", "mapox/find_return"): dict(
        total_timesteps=1_500_000_000,
        training=dict(num_epochs=100),
        environment=dict(num_envs=2048, kwargs=dict(num_agents=8)),
        rollout=dict(num_steps=64),
        num_layers=2,
        cell=dict(features=256, dtype="bfloat16"),
        network=dict(dtype="bfloat16"),
        algorithm=dict(
            num_minibatches=16,
            update_epochs=1,
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=0.001,
            value_coefficient=0.5,
            gamma=0.966,
            gae_lambda=0.95,
        ),
        optimizer=dict(name="muon", lr=0.003, max_grad_norm=0.4),
    ),
    ("recurrent_pupo", "mapox"): dict(
        total_timesteps=104_857_600,
        training=dict(num_epochs=20),
        environment=dict(num_envs=2048),
        rollout=dict(num_steps=64),
        num_layers=2,
        cell=dict(features=256, dtype="bfloat16"),
        network=dict(dtype="bfloat16"),
        algorithm=dict(
            num_minibatches=16,
            update_epochs=1,
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=0.001,
            value_coefficient=0.5,
            gamma=0.966,
            gae_lambda=0.95,
        ),
        optimizer=dict(name="muon", lr=0.003, max_grad_norm=0.4),
    ),
    ("reppo", "brax"): dict(
        total_timesteps=50_000_000,
        rollout=dict(num_steps=32),
        environment=dict(num_envs=4096),
    ),
}

spaces = {
    ("ippo", "connectx"): dict(
        n_trials=1024,
        n_jobs=1,
        num_random_samples=16,
        resample_frequency=16,
        max_failure_rate=0.5,
        seed=0,
        max_suggestion_cost=900,
        params={
            "total_timesteps": dict(distribution="log_normal", min=5e7, max=2e9, center=7.14e8, scale=1.92),
            "environment.num_envs": dict(
                distribution="uniform_pow2", min=512, max=65536, center=16384, scale=8.33
            ),
            "rollout.num_steps": dict(distribution="uniform_pow2", min=2, max=32, center=8, scale=3.33),
            "algorithm.num_minibatches": dict(
                distribution="uniform_pow2", min=1, max=32, center=4, scale=5.0
            ),
            "algorithm.update_epochs": dict(distribution="int_uniform", min=1, max=8, center=6, scale=8.33),
            "network.features": dict(distribution="uniform_pow2", min=64, max=1024, center=512, scale=5.0),
            "network.layers": dict(distribution="int_uniform", min=1, max=4, center=2, scale=3.33),
            "optimizer.lr": dict(distribution="log_normal", min=1e-5, max=3e-3, center=1.46e-3, scale=3.61),
            "optimizer.max_grad_norm": dict(
                distribution="log_normal", min=0.05, max=5.0, center=2.44, scale=2.81
            ),
            "algorithm.entropy_coefficient": dict(
                distribution="log_normal", min=1e-5, max=0.5, center=0.0994, scale=6.66
            ),
            "algorithm.gae_lambda": dict(
                distribution="logit_normal", min=0.5, max=0.995, center=0.989, scale=3.26
            ),
            "algorithm.clip_coefficient": dict(
                distribution="uniform", min=0.1, max=0.5, center=0.155, scale=0.57
            ),
            "algorithm.value_coefficient": dict(
                distribution="log_normal", min=0.1, max=10.0, center=0.723, scale=1.9
            ),
        },
    ),
}

for (algorithm, environment), node in table.items():
    place(node, f"hyperparameters/{algorithm}/{environment}", package="_global_")
