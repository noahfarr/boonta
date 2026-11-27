"""Unit tests for Scalable PSR (SPSR), TransitionForecastAdvantage, and SnapshotAutoReset."""

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from boonta import environments
from boonta.algorithms.ppo import PPO, PPOConfig
from boonta.curricula import spsr
from boonta.curricula.psr import (
    CountedRelevance,
    Diagonal,
    Selector,
    SuccessorFootprint,
    SuccessorRepresentation,
    TransitionForecastAdvantage,
    WaterFilling,
    make as make_psr,
)
from boonta.environments.wrappers.archive_auto_reset import (
    ArchiveAutoReset,
    SnapshotAutoReset,
    optimistic,
)
from boonta.environments import gymnax as gymnax_env
from boonta.networks import ActorCritic, Categorical, FeatureExtractor, Gaussian, Network
from boonta.podracers import anakin
from boonta.podracers.anakin import AnakinConfig
from boonta.utils import mesh as make_mesh


def _make_ppo(network, lr=1e-3):
    cfg = PPOConfig(
        num_minibatches=4,
        update_epochs=2,
        clip_coefficient=0.2,
        clip_value_loss=True,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantage=True,
    )
    return PPO(cfg=cfg, network=network, optimizer=optax.adam(lr))


def test_spsr_discrete_minatar():
    env = environments.make(namespace="gymnax", suite="minatar", env_id="Asterix-MinAtar")
    num_actions = env.action_space().num_actions
    env = SnapshotAutoReset(env, num_envs=64, archive_size=256, stride=8, max_writes_per_step=16)
    net = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential([
                lambda x: x.reshape((*x.shape[:-3], -1)).astype(jnp.float32),
                nn.Dense(64),
                nn.relu,
            ])
        ),
        head=ActorCritic(actor=Categorical(nn.Dense(num_actions)), critic=nn.Dense(1)),
    )
    algo = _make_ppo(net)
    algo, env, pit, lap = spsr.make(algo, env, k=4.0, proj_dim=32, refresh_chunk=64)
    racer = anakin.make(config=AnakinConfig(num_steps=16, mesh=make_mesh(1)), algorithm=algo, environment=env, pit=pit, lap=lap)
    state = racer.init(jax.random.key(0))
    state, logs = racer.train(state, jax.random.key(1), 4)
    assert jnp.all(jnp.isfinite(state.pit_state.w_psr))
    assert "spsr/admitted_share" in logs


def test_spsr_continuous_pendulum():
    env = environments.make(namespace="gymnax", suite="continuous_control", env_id="Pendulum-v1")
    act_dim = env.action_space().shape[0]
    env = SnapshotAutoReset(env, num_envs=64, archive_size=256, stride=8, max_writes_per_step=16, fresh_fn=optimistic)
    net = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential([nn.Dense(64), nn.tanh, nn.Dense(64), nn.tanh])
        ),
        head=ActorCritic(actor=Gaussian(nn.Dense(2 * act_dim)), critic=nn.Dense(1)),
    )
    algo = _make_ppo(net)
    algo, env, pit, lap = spsr.make(algo, env, k=4.0, proj_dim=32, refresh_chunk=64)
    racer = anakin.make(config=AnakinConfig(num_steps=16, mesh=make_mesh(1)), algorithm=algo, environment=env, pit=pit, lap=lap)
    state = racer.init(jax.random.key(42))
    state, logs = racer.train(state, jax.random.key(43), 4)
    assert jnp.all(jnp.isfinite(state.pit_state.w_psr))


def test_transition_forecast_in_psr():
    env = environments.make(namespace="gymnax", suite="minatar", env_id="Asterix-MinAtar")
    num_actions = env.action_space().num_actions
    env = ArchiveAutoReset(env, num_envs=64, cell_fn=gymnax_env.keyed, archive_size=256)
    net = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential([
                lambda x: x.reshape((*x.shape[:-3], -1)).astype(jnp.float32),
                nn.Dense(64),
                nn.relu,
            ])
        ),
        head=ActorCritic(actor=Categorical(nn.Dense(num_actions)), critic=nn.Dense(1)),
    )
    algo = _make_ppo(net)
    sel = Selector(
        successor=SuccessorRepresentation(gamma=0.9375, rate=0.1, columns=32),
        footprint=SuccessorFootprint(),
        relevance=CountedRelevance(),
        gain=TransitionForecastAdvantage(gamma=0.99, gae_lambda=0.95, horizon=4),
        score=Diagonal(),
        allocation=WaterFilling(iterations=30, rho_gate=True),
        k=4.0,
    )
    algo, env, pit, lap = make_psr(algo, env, selector=sel)
    racer = anakin.make(config=AnakinConfig(num_steps=16, mesh=make_mesh(1)), algorithm=algo, environment=env, pit=pit, lap=lap)
    state = racer.init(jax.random.key(7))
    state, logs = racer.train(state, jax.random.key(8), 4)
    assert jnp.all(jnp.isfinite(state.pit_state.gain.gain_sums))
