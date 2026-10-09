from functools import partial

import flashbax as fbx
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from boonta.algorithms.bc import BC, BCConfig
from boonta.algorithms.dqn import DQN, DQNConfig
from boonta.algorithms.grpo import GRPO, GRPOConfig
from boonta.algorithms.iql import IQL, IQLConfig
from boonta.algorithms.mmd import MMD, MMDConfig
from boonta.algorithms.ppo import PPO, PPOConfig
from boonta.algorithms.pqn import PQN, PQNConfig
from boonta.algorithms.recurrent_bc import RecurrentBC, RecurrentBCConfig
from boonta.algorithms.recurrent_dqn import RecurrentDQN, RecurrentDQNConfig
from boonta.algorithms.recurrent_grpo import RecurrentGRPO, RecurrentGRPOConfig
from boonta.algorithms.recurrent_ppo import RecurrentPPO, RecurrentPPOConfig
from boonta.algorithms.recurrent_pqn import RecurrentPQN, RecurrentPQNConfig
from boonta.algorithms.recurrent_pupo import RecurrentPuPO, RecurrentPuPOConfig
from boonta.algorithms.recurrent_sac import RecurrentSAC, RecurrentSACConfig
from boonta.algorithms.reppo import REPPO, REPPOConfig
from boonta.algorithms.sac import SAC, SACConfig
from boonta.environments.wrappers import (GroupedAutoReset,
                                          RecordEpisodeStatistics,
                                          SameStepAutoReset, Vectorize)
from boonta.networks import (RNN, SSM, ActorCritic, Categorical,
                             EpsilonGreedy, FeatureExtractor, GatedDeltaNet,
                             Gaussian, Highway, LinearAttention, MinGRUCell,
                             Network, Qwen3_5,
                             RTUCell,
                             SelfAttention, SquashedGaussian, Tower,
                             causal_attention_mask, llama, repeat)
from boonta.networks.layers import Identity, Parameter
from boonta.podracers import anakin, quadinaros, sebulba
from boonta.utils import mesh

from dummies import Episodes, demonstrations, flatten

WIDTH = 32


def discrete(environment) -> bool:
    return jnp.issubdtype(environment.action_space().dtype, jnp.integer)


def num_actions(environment) -> int:
    return environment.action_space().num_actions


def action_dim(environment) -> int:
    dim, *_ = environment.action_space().shape
    return dim


def encoder():
    return FeatureExtractor(
        observation_extractor=nn.Sequential(
            [nn.Dense(WIDTH), nn.relu, nn.Dense(WIDTH), nn.relu]
        )
    )


def gru(dtype=None):
    return RNN(cell=nn.GRUCell(features=WIDTH, dtype=dtype))


def min_gru(dtype=None):
    return SSM(cell=MinGRUCell(features=WIDTH, dtype=dtype))


def rtu(dtype=None):
    return RNN(cell=RTUCell(features=WIDTH, dtype=dtype))


def attention(context_length=4, dtype=None):
    return SelfAttention(
        features=WIDTH,
        num_heads=2,
        context_length=context_length,
        attention_mask=causal_attention_mask,
        dtype=dtype,
    )


def gated_delta_net(dtype=None):
    return LinearAttention(
        cell=GatedDeltaNet(features=WIDTH, num_heads=2, head_dim=8, dtype=dtype),
        chunk_size=2,
    )


def qwen3_5(dtype=None):
    return Qwen3_5(
        features=WIDTH,
        layer_types=("linear_attention", "full_attention"),
        num_heads=2,
        num_groups=1,
        head_dim=8,
        rotary_dim=4,
        max_wavelength=10_000.0,
        context_length=6,
        linear_num_heads=2,
        linear_num_value_heads=4,
        linear_head_dim=8,
        linear_value_head_dim=8,
        chunk_size=2,
        hidden_dim=2 * WIDTH,
        dtype=dtype,
    )


def highway(dtype=None):
    return Tower(block=Highway(blocks=min_gru(dtype), dtype=dtype), num_layers=2)


def llama_stack(dtype=None):
    return llama(attention(dtype=dtype), num_layers=2, features=WIDTH, dtype=dtype)


def repeated(dtype=None):
    return repeat(min_gru(dtype), num_layers=2)


TORSOS = {
    "gru": gru,
    "min_gru": min_gru,
    "rtu": rtu,
    "attention-one-rollout": partial(attention, 4),
    "attention-longer-than-a-rollout": partial(attention, 6),
    "gated_delta_net": gated_delta_net,
    "qwen3_5": qwen3_5,
    "highway": highway,
    "llama": llama_stack,
    "repeat": repeated,
}


def policy(environment):
    if discrete(environment):
        return Categorical(nn.Dense(num_actions(environment)))
    return Gaussian(nn.Dense(2 * action_dim(environment)))


def critics(torso=None):
    return Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=Identity(), action_extractor=Identity()
        ),
        torso=torso,
        head=nn.vmap(
            nn.Sequential,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=2,
        )([nn.Dense(WIDTH), nn.relu, nn.Dense(WIDTH), nn.relu, nn.Dense(1)]),
    )


def adam(learning_rate=3e-3):
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))


def explore(num_steps):
    return optax.linear_schedule(1.0, 0.05, num_steps)


def transitions(num_envs, capacity=4096, batch_size=64):
    return fbx.make_trajectory_buffer(
        max_length_time_axis=capacity // num_envs,
        min_length_time_axis=max(1, batch_size // num_envs),
        sample_batch_size=batch_size,
        add_batch_size=num_envs,
        sample_sequence_length=1,
        period=1,
    )


def trajectories(num_envs, length=8, capacity=4096, batch_size=32):
    return fbx.make_trajectory_buffer(
        max_length_time_axis=capacity // num_envs,
        min_length_time_axis=length,
        sample_batch_size=batch_size,
        add_batch_size=num_envs,
        sample_sequence_length=length,
        period=1,
    )


def wrap(environment, num_envs, gamma=0.99):
    environment = SameStepAutoReset(environment)
    environment = RecordEpisodeStatistics(environment, gamma=gamma)
    return Vectorize(environment, num_envs=num_envs)


def group(environment, num_envs, num_steps, group_size, gamma=0.99):
    environment = Vectorize(SameStepAutoReset(environment), num_envs=num_envs)
    environment = GroupedAutoReset(environment, num_steps=num_steps, group_size=group_size)
    return RecordEpisodeStatistics(environment, gamma=gamma)


def online(algorithm, environment, num_envs, num_steps, devices=1):
    config = anakin.AnakinConfig(
        num_envs=num_envs, num_steps=num_steps, mesh=mesh(devices)
    )
    return anakin.make(config, algorithm, environment)


def asynchronous(algorithm, environment, num_envs, num_steps, actors=2):
    config = sebulba.SebulbaConfig(
        num_envs=num_envs,
        num_steps=num_steps,
        learner=mesh(actors),
        actor=mesh(actors, start=actors),
    )
    return sebulba.make(config, algorithm, environment)


def offline(algorithm, environment, dataset, batch_shape, num_envs=16):
    config = quadinaros.QuadinarosConfig(
        num_envs=num_envs, batch_shape=batch_shape, mesh=mesh(1)
    )
    return quadinaros.make(
        config, algorithm, wrap(environment, num_envs), dataset
    )


def ppo(
    environment,
    num_envs=32,
    num_steps=16,
    podracer=online,
    optimizer=None,
    auxiliary_losses=(),
    wrapper=wrap,
):
    algorithm = PPO(
        cfg=PPOConfig(
            num_minibatches=4,
            update_epochs=4,
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            gamma=0.99,
            gae_lambda=0.95,
        ),
        network=Network(
            feature_extractor=encoder(),
            head=ActorCritic(actor=policy(environment), critic=nn.Dense(1)),
        ),
        optimizer=optimizer or adam(),
        auxiliary_losses=auxiliary_losses,
    )
    return podracer(algorithm, wrapper(environment, num_envs), num_envs, num_steps)


def mmd(
    environment,
    num_envs=32,
    num_steps=16,
    podracer=online,
    optimizer=None,
    auxiliary_losses=(),
):
    algorithm = MMD(
        cfg=MMDConfig(
            num_minibatches=4,
            update_epochs=4,
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
        network=Network(
            feature_extractor=encoder(),
            head=ActorCritic(actor=policy(environment), critic=nn.Dense(1)),
        ),
        optimizer=optimizer or adam(),
        auxiliary_losses=auxiliary_losses,
    )
    return podracer(algorithm, wrap(environment, num_envs), num_envs, num_steps)


def grpo(
    environment,
    num_envs=32,
    num_steps=12,
    podracer=online,
    optimizer=None,
    auxiliary_losses=(),
    group_size=4,
):
    algorithm = GRPO(
        cfg=GRPOConfig(
            group_size=group_size,
            num_minibatches=4,
            update_epochs=2,
            clip_coefficient=0.2,
            kl_coefficient=0.0,
            gamma=0.99,
        ),
        network=Network(feature_extractor=encoder(), head=policy(environment)),
        optimizer=optimizer or adam(),
        auxiliary_losses=auxiliary_losses,
    )
    return podracer(
        algorithm,
        group(environment, num_envs, num_steps, group_size),
        num_envs,
        num_steps,
    )


def pqn(
    environment,
    num_envs=32,
    num_steps=16,
    podracer=online,
    optimizer=None,
    auxiliary_losses=(),
):
    algorithm = PQN(
        cfg=PQNConfig(num_minibatches=4, update_epochs=2, gamma=0.99, q_lambda=0.65),
        network=Network(
            feature_extractor=encoder(),
            head=EpsilonGreedy(nn.Dense(num_actions(environment))),
        ),
        exploration_schedule=explore(num_envs * num_steps * 100),
        optimizer=optimizer or adam(),
        auxiliary_losses=auxiliary_losses,
    )
    return podracer(algorithm, wrap(environment, num_envs), num_envs, num_steps)


def dqn(environment, num_envs=32, podracer=online, auxiliary_losses=()):
    algorithm = DQN(
        cfg=DQNConfig(updates_per_step=4, gamma=0.99, tau=0.05),
        network=Network(
            feature_extractor=encoder(),
            head=EpsilonGreedy(nn.Dense(num_actions(environment))),
        ),
        exploration_schedule=explore(num_envs * 300),
        buffer=transitions(num_envs),
        optimizer=adam(1e-3),
        auxiliary_losses=auxiliary_losses,
    )
    return podracer(algorithm, wrap(environment, num_envs), num_envs, 1)


def sac(environment, num_envs=32, podracer=online, auxiliary_losses=()):
    algorithm = SAC(
        cfg=SACConfig(
            updates_per_step=4,
            gamma=0.99,
            tau=0.05,
            target_entropy=-float(action_dim(environment)),
        ),
        actor=Network(
            feature_extractor=encoder(),
            head=SquashedGaussian(nn.Dense(2 * action_dim(environment))),
        ),
        critic=critics(),
        alpha=Parameter(),
        buffer=transitions(num_envs),
        actor_optimizer=optax.adam(1e-3),
        critic_optimizer=optax.adam(1e-3),
        alpha_optimizer=optax.adam(1e-3),
        auxiliary_losses=auxiliary_losses,
    )
    return podracer(algorithm, wrap(environment, num_envs), num_envs, 1)


def reppo(
    environment,
    num_envs=32,
    num_steps=16,
    podracer=online,
    optimizer=None,
    auxiliary_losses=(),
):
    algorithm = REPPO(
        cfg=REPPOConfig(
            num_minibatches=4,
            update_epochs=4,
            gamma=0.99,
            td_lambda=0.95,
            target_kl=0.1,
            target_entropy_scale=0.5,
            num_kl_samples=4,
            vmin=-1.0,
            vmax=2.0,
            num_bins=31,
            action_dim=action_dim(environment),
        ),
        actor=Network(
            feature_extractor=encoder(),
            head=SquashedGaussian(nn.Dense(2 * action_dim(environment))),
        ),
        critic=Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=Identity(), action_extractor=Identity()
            ),
            head=nn.Sequential(
                [nn.Dense(WIDTH), nn.relu, nn.Dense(WIDTH), nn.relu, nn.Dense(31)]
            ),
        ),
        alpha=Parameter(),
        lagrangian=Parameter(),
        actor_optimizer=optimizer or optax.adam(1e-3),
        critic_optimizer=optimizer or optax.adam(1e-3),
        alpha_optimizer=optimizer or optax.adam(1e-3),
        lagrangian_optimizer=optimizer or optax.adam(1e-3),
        auxiliary_losses=auxiliary_losses,
    )
    return podracer(algorithm, wrap(environment, num_envs), num_envs, num_steps)


def recurrent_ppo(
    environment,
    num_envs=32,
    num_steps=16,
    podracer=online,
    optimizer=None,
    auxiliary_losses=(),
    torso=None,
):
    algorithm = RecurrentPPO(
        cfg=RecurrentPPOConfig(
            num_minibatches=4,
            update_epochs=4,
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            gamma=0.99,
            gae_lambda=0.95,
        ),
        network=Network(
            feature_extractor=encoder(),
            torso=torso or gru(),
            head=ActorCritic(actor=policy(environment), critic=nn.Dense(1)),
        ),
        optimizer=optimizer or adam(),
        auxiliary_losses=auxiliary_losses,
    )
    return podracer(algorithm, wrap(environment, num_envs), num_envs, num_steps)


def recurrent_pupo(
    environment,
    num_envs=32,
    num_steps=16,
    podracer=online,
    optimizer=None,
    auxiliary_losses=(),
    torso=None,
):
    algorithm = RecurrentPuPO(
        cfg=RecurrentPuPOConfig(
            num_minibatches=4,
            update_epochs=4,
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=0.01,
            value_coefficient=0.5,
            gamma=0.99,
            gae_lambda=0.95,
        ),
        network=Network(
            feature_extractor=encoder(),
            torso=torso or gru(),
            head=ActorCritic(actor=policy(environment), critic=nn.Dense(1)),
        ),
        optimizer=optimizer or adam(),
        auxiliary_losses=auxiliary_losses,
    )
    return podracer(algorithm, wrap(environment, num_envs), num_envs, num_steps)


def recurrent_grpo(
    environment,
    num_envs=32,
    num_steps=12,
    podracer=online,
    optimizer=None,
    auxiliary_losses=(),
    torso=None,
    group_size=4,
):
    algorithm = RecurrentGRPO(
        cfg=RecurrentGRPOConfig(
            group_size=group_size,
            num_minibatches=4,
            update_epochs=2,
            clip_coefficient=0.2,
            kl_coefficient=0.0,
            gamma=0.99,
        ),
        network=Network(
            feature_extractor=encoder(),
            torso=torso or gru(),
            head=policy(environment),
        ),
        optimizer=optimizer or adam(),
        auxiliary_losses=auxiliary_losses,
    )
    return podracer(
        algorithm,
        group(environment, num_envs, num_steps, group_size),
        num_envs,
        num_steps,
    )


def recurrent_pqn(
    environment,
    num_envs=32,
    num_steps=16,
    podracer=online,
    optimizer=None,
    auxiliary_losses=(),
    torso=None,
):
    algorithm = RecurrentPQN(
        cfg=RecurrentPQNConfig(
            num_minibatches=4, update_epochs=2, gamma=0.99, q_lambda=0.65
        ),
        network=Network(
            feature_extractor=encoder(),
            torso=torso or gru(),
            head=EpsilonGreedy(nn.Dense(num_actions(environment))),
        ),
        exploration_schedule=explore(num_envs * num_steps * 100),
        optimizer=optimizer or adam(),
        auxiliary_losses=auxiliary_losses,
    )
    return podracer(algorithm, wrap(environment, num_envs), num_envs, num_steps)


def recurrent_dqn(environment, num_envs=32, podracer=online, auxiliary_losses=()):
    algorithm = RecurrentDQN(
        cfg=RecurrentDQNConfig(updates_per_step=4, gamma=0.99, tau=0.05),
        network=Network(
            feature_extractor=encoder(),
            torso=gru(),
            head=EpsilonGreedy(nn.Dense(num_actions(environment))),
        ),
        exploration_schedule=explore(num_envs * 300),
        buffer=trajectories(num_envs),
        optimizer=adam(1e-3),
        auxiliary_losses=auxiliary_losses,
    )
    return podracer(algorithm, wrap(environment, num_envs), num_envs, 1)


def recurrent_sac(environment, num_envs=32, podracer=online, auxiliary_losses=()):
    algorithm = RecurrentSAC(
        cfg=RecurrentSACConfig(
            updates_per_step=4,
            gamma=0.99,
            tau=0.05,
            target_entropy=-float(action_dim(environment)),
        ),
        actor=Network(
            feature_extractor=encoder(),
            torso=gru(),
            head=SquashedGaussian(nn.Dense(2 * action_dim(environment))),
        ),
        critic=critics(torso=gru()),
        alpha=Parameter(),
        buffer=trajectories(num_envs),
        actor_optimizer=optax.adam(1e-3),
        critic_optimizer=optax.adam(1e-3),
        alpha_optimizer=optax.adam(1e-3),
        auxiliary_losses=auxiliary_losses,
    )
    return podracer(algorithm, wrap(environment, num_envs), num_envs, 1)


def bc(environment, batch_size=64, auxiliary_losses=()):
    algorithm = BC(
        cfg=BCConfig(batch_size=batch_size),
        network=Network(feature_extractor=encoder(), head=policy(environment)),
        optimizer=adam(),
        auxiliary_losses=auxiliary_losses,
    )
    dataset = Episodes(flatten(demonstrations(environment, jax.random.key(7), 256)))
    return offline(algorithm, environment, dataset, (batch_size,))


def iql(environment, batch_size=64, auxiliary_losses=()):
    algorithm = IQL(
        cfg=IQLConfig(gamma=0.99, tau=0.05, batch_size=batch_size),
        actor=Network(feature_extractor=encoder(), head=policy(environment)),
        critic=critics(),
        value=Network(feature_extractor=encoder(), head=nn.Dense(1)),
        actor_optimizer=optax.adam(1e-3),
        critic_optimizer=optax.adam(1e-3),
        value_optimizer=optax.adam(1e-3),
        auxiliary_losses=auxiliary_losses,
    )
    dataset = Episodes(flatten(demonstrations(environment, jax.random.key(7), 256)))
    return offline(algorithm, environment, dataset, (batch_size,))


def recurrent_bc(environment, batch_size=32, auxiliary_losses=()):
    algorithm = RecurrentBC(
        cfg=RecurrentBCConfig(batch_size=batch_size),
        network=Network(
            feature_extractor=encoder(), torso=gru(), head=policy(environment)
        ),
        optimizer=adam(),
        auxiliary_losses=auxiliary_losses,
    )
    dataset = Episodes(demonstrations(environment, jax.random.key(7), 256))
    return offline(
        algorithm, environment, dataset, (batch_size, environment.shortest)
    )
