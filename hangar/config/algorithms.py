import optax

from boonta.algorithms.bc import BCConfig
from boonta.algorithms.dqn import DQNConfig
from boonta.algorithms.grpo import GRPOConfig
from boonta.algorithms.iql import IQLConfig
from boonta.algorithms.mmd import MMDConfig
from boonta.algorithms.ppo import PPOConfig
from boonta.algorithms.pqn import PQNConfig
from boonta.algorithms.recurrent_bc import RecurrentBCConfig
from boonta.algorithms.recurrent_dqn import RecurrentDQNConfig
from boonta.algorithms.recurrent_grpo import RecurrentGRPOConfig
from boonta.algorithms.recurrent_ppo import RecurrentPPOConfig
from boonta.algorithms.recurrent_pqn import RecurrentPQNConfig
from boonta.algorithms.recurrent_pupo import RecurrentPuPOConfig
from boonta.algorithms.recurrent_sac import RecurrentSACConfig
from boonta.algorithms.reppo import REPPOConfig
from boonta.algorithms.sac import SACConfig

from .sections import Exploration, Optimizer, Optimizers, Replay, Rollout
from .store import builds, fbuilds, store

spaces = {}

algorithm = store(group="algorithm", package="_global_")

recurrent = [{"/torso": "default"}, {"/cell": "gru"}, {"/stack": "default"}]
offline = [{"/dataset": "minari/mujoco/expert"}, {"override /podracer": "quadinaros"}]
epsilon = Exploration(start=1.0, end=0.01, fraction=0.2)

algorithm(
    dict(
        defaults=offline,
        algorithm=fbuilds(BCConfig, batch_size=256, entropy_coefficient=0.0),
        environment=dict(num_envs=32),
        optimizer=Optimizer(lr=3e-4),
    ),
    name="bc",
)

algorithm(
    dict(
        defaults=[{"/buffer": "transition"}],
        algorithm=fbuilds(DQNConfig, updates_per_step=32, gamma=0.99, tau=0.005),
        rollout=Rollout(num_steps=1),
        environment=dict(num_envs=128),
        replay=Replay(capacity=262_144),
        optimizer=Optimizer(lr=5e-4, max_grad_norm=10.0),
        exploration=epsilon,
    ),
    name="dqn",
)

algorithm(
    dict(
        algorithm=fbuilds(
            GRPOConfig,
            group_size=8,
            num_minibatches=16,
            update_epochs=2,
            clip_coefficient=0.2,
            kl_coefficient=0.04,
            gamma=0.99,
        ),
        rollout=Rollout(num_steps=256),
        environment=dict(num_envs=512),
        optimizer=Optimizer(lr=2.5e-4, max_grad_norm=0.5),
    ),
    name="grpo",
)

for name in ("ippo", "mappo"):
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
            ),
            rollout=Rollout(num_steps=16),
            environment=dict(num_envs=4096),
            optimizer=Optimizer(lr=2.5e-4, max_grad_norm=0.5),
        ),
        name=name,
    )

algorithm(
    dict(
        defaults=offline,
        algorithm=fbuilds(
            IQLConfig,
            gamma=0.99,
            tau=0.005,
            batch_size=256,
            expectile=0.7,
            beta=3.0,
            max_advantage_weight=100.0,
        ),
        environment=dict(num_envs=32),
        optimizer=Optimizers(actor_lr=3e-4, critic_lr=3e-4, value_lr=3e-4),
    ),
    name="iql",
)

algorithm(
    dict(
        algorithm=fbuilds(
            MMDConfig,
            num_minibatches=16,
            update_epochs=2,
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=0.0,
            value_coefficient=0.5,
            gamma=0.99,
            gae_lambda=0.95,
            magnet_coefficient=0.01,
            magnet_decay=0.99,
            magnet_anneal=1.0,
        ),
        rollout=Rollout(num_steps=16),
        environment=dict(num_envs=4096),
        optimizer=Optimizer(lr=2.5e-4, max_grad_norm=0.5),
    ),
    name="mmd",
)

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

algorithm(
    dict(
        algorithm=fbuilds(PQNConfig, num_minibatches=4, update_epochs=1, gamma=0.99, q_lambda=0.65),
        rollout=Rollout(num_steps=16),
        environment=dict(num_envs=128),
        optimizer=Optimizer(lr=5e-4, max_grad_norm=10.0),
        exploration=epsilon,
    ),
    name="pqn",
)

algorithm(
    dict(
        defaults=[*recurrent, {"override /podracer": "quadinaros"}],
        algorithm=fbuilds(RecurrentBCConfig, batch_size=32, entropy_coefficient=0.0),
        optimizer=Optimizer(lr=3e-4, max_grad_norm=0.5),
    ),
    name="recurrent_bc",
)

algorithm(
    dict(
        defaults=[*recurrent, {"/buffer": "trajectory"}],
        algorithm=fbuilds(RecurrentDQNConfig, updates_per_step=32, gamma=0.99, tau=0.005),
        rollout=Rollout(num_steps=1),
        environment=dict(num_envs=128),
        replay=Replay(capacity=262_144),
        optimizer=Optimizer(lr=5e-4, max_grad_norm=10.0),
        exploration=epsilon,
    ),
    name="recurrent_dqn",
)

algorithm(
    dict(
        algorithm=fbuilds(
            RecurrentGRPOConfig,
            group_size=8,
            num_minibatches=16,
            update_epochs=2,
            clip_coefficient=0.2,
            kl_coefficient=0.04,
            gamma=1.0,
        ),
        rollout=Rollout(num_steps=30),
        environment=dict(num_envs=128),
        optimizer=Optimizer(lr=1e-5, max_grad_norm=0.5),
    ),
    name="recurrent_grpo",
)

algorithm(
    dict(
        defaults=recurrent,
        algorithm=fbuilds(
            RecurrentPPOConfig,
            num_minibatches=8,
            update_epochs=4,
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            gamma=0.99,
            gae_lambda=0.95,
            normalize_advantage=False,
        ),
        rollout=Rollout(num_steps=128),
        environment=dict(num_envs=64),
        optimizer=Optimizer(lr=2.5e-4, max_grad_norm=0.5),
    ),
    name="recurrent_ppo",
)

algorithm(
    dict(
        defaults=recurrent,
        algorithm=fbuilds(RecurrentPQNConfig, num_minibatches=4, update_epochs=1, gamma=0.99, q_lambda=0.65),
        rollout=Rollout(num_steps=16),
        environment=dict(num_envs=128),
        optimizer=Optimizer(lr=5e-4, max_grad_norm=10.0),
        exploration=epsilon,
    ),
    name="recurrent_pqn",
)

algorithm(
    dict(
        defaults=[{"/torso": "default"}, {"/cell": "min_gru"}, {"/stack": "highway"}],
        algorithm=fbuilds(
            RecurrentPuPOConfig,
            num_minibatches=8,
            update_epochs=4,
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            gamma=0.99,
            gae_lambda=0.95,
            advantage_clip=1.0,
            trace_clip=1.0,
            priority_exponent=0.8,
            normalize_advantage=True,
        ),
        importance_exponent=builds(optax.constant_schedule, value=0.2),
        rollout=Rollout(num_steps=128),
        environment=dict(num_envs=64),
        optimizer=Optimizer(lr=2.5e-4, max_grad_norm=0.5),
    ),
    name="recurrent_pupo",
)

algorithm(
    dict(
        defaults=[{"/torso": "default"}, {"/cell": "min_gru"}, {"/stack": "default"}, {"/buffer": "trajectory"}],
        algorithm=fbuilds(
            RecurrentSACConfig,
            updates_per_step=64,
            gamma=0.997,
            tau=0.005,
            zen_exclude=("target_entropy",),
        ),
        rollout=Rollout(num_steps=1),
        environment=dict(num_envs=128),
        replay=Replay(capacity=1_048_576),
        optimizer=Optimizers(actor_lr=6e-4, critic_lr=6e-4, alpha_lr=6e-4),
    ),
    name="recurrent_sac",
)

algorithm(
    dict(
        algorithm=fbuilds(
            REPPOConfig,
            num_minibatches=128,
            update_epochs=4,
            gamma=0.99,
            td_lambda=0.95,
            target_kl=0.1,
            target_entropy_scale=0.5,
            num_kl_samples=16,
            vmin=0.0,
            vmax=150.0,
            num_bins=151,
            zen_exclude=("action_dim",),
        ),
        rollout=Rollout(num_steps=128),
        environment=dict(num_envs=1024),
        optimizer=Optimizers(actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, lagrangian_lr=3e-4),
    ),
    name="reppo",
)

algorithm(
    dict(
        defaults=[{"/buffer": "transition"}],
        algorithm=fbuilds(
            SACConfig,
            updates_per_step=64,
            gamma=0.997,
            tau=0.005,
            zen_exclude=("target_entropy",),
        ),
        rollout=Rollout(num_steps=1),
        environment=dict(num_envs=128),
        replay=Replay(capacity=1_048_576),
        optimizer=Optimizers(actor_lr=6e-4, critic_lr=6e-4, alpha_lr=6e-4),
    ),
    name="sac",
)
