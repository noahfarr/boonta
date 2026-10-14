import dataclasses
from functools import cache

import flashbax as fbx
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct

from boonta import environments
from boonta.algorithms.bc import BC, BCConfig
from boonta.algorithms.dqn import DQN, DQNConfig
from boonta.algorithms.iql import IQL, IQLConfig
from boonta.algorithms.ppo import PPO, PPOConfig
from boonta.algorithms.pqn import PQN, PQNConfig
from boonta.environments.jumanji import Jumanji
from boonta.environments.wrappers import RecordEpisodeStatistics, SameStepAutoReset
from boonta.networks import (ActorCritic, Categorical, EpsilonGreedy,
                             FeatureExtractor, Network)
from boonta.podracers.anakin import Anakin, AnakinConfig
from boonta.podracers.quadinaros import Quadinaros, QuadinarosConfig
from boonta.podracers.sebulba import Sebulba, SebulbaConfig
from boonta.utils import Transition, mesh

@struct.dataclass(frozen=True)
class BundledState:
    transitions: Transition = struct.field(metadata={"axis": "data"})


class Bundled:
    def __init__(self, transitions):
        self.transitions = transitions

    def init(self):
        return BundledState(self.transitions)

    def update(self, state, key, sharding):
        return state

    def sample(self, state, key, batch_shape):
        leaf, *_ = jax.tree.leaves(state.transitions)
        index = jax.random.randint(key, batch_shape, 0, leaf.shape[0])
        return jax.tree.map(lambda x: x[index], state.transitions)

    def close(self):
        pass


class Snake(Jumanji):
    def observe(self, timestep):
        return timestep.replace(obs=timestep.obs[..., :4])

    def init(self, key):
        state, timestep = super().init(jax.random.key_data(key))
        return state, self.observe(timestep)

    def step(self, key, state, action):
        state, timestep = super().step(key, state, action)
        return state, self.observe(timestep)

    def observation_space(self):
        space = super().observation_space()
        return dataclasses.replace(space, shape=space.shape[:-1] + (4,))


TRACKS = {
    "cartpole": "CartPole-v1",
    "breakout": "Breakout-MinAtar",
    "catch": "Catch-bsuite",
}


KINETIX_LEVELS = ["s/h0_weak_thrust", "s/h2_one_wheel_car", "s/h3_point_the_thruster"]


PICTURES = {"kinetix": 5}


@cache
def painter(track):
    from kinetix.render.renderer_pixels import make_render_pixels

    environment = game(track)
    static = environment._env.static_env_params.replace(downscale=PICTURES[track])
    with jax.ensure_compile_time_eval():
        return make_render_pixels(environment._params, static)


def picture(track, state):
    pixels = painter(track)(state)
    return pixels.transpose(1, 0, 2)[::-1].astype(jnp.uint8)


@cache
def game(track):
    if track == "kinetix":
        from kinetix.util.saving import load_evaluation_levels

        _, static_env_params = load_evaluation_levels(KINETIX_LEVELS)
        return environments.make(
            "kinetix", KINETIX_LEVELS, kwargs={"static_env_params": static_env_params}
        )
    if track == "snake":
        return Snake(environments.make("jumanji", "Snake-v1")._env)
    return environments.make("gymnax", TRACKS[track])


def wrap(environment, num_envs):
    environment = SameStepAutoReset(environment, num_envs=num_envs)
    return RecordEpisodeStatistics(environment, gamma=0.99)


def flattener(track):
    rank = len(game(track).observation_space().shape)
    return lambda x: x.reshape(*x.shape[: x.ndim - rank], -1).astype(jnp.float32)


def conv(track):
    rank = len(game(track).observation_space().shape)
    return nn.Sequential(
        [
            lambda x: x.astype(jnp.float32),
            nn.Conv(16, (3, 3), padding="VALID"),
            nn.relu,
            lambda x: x.reshape(*x.shape[: x.ndim - rank], -1),
        ]
    )


def trunk(track, h, activation, norm=False):
    layers = [flattener(track)] if h.get("encoder", "mlp") == "mlp" else [conv(track)]
    for _ in range(h.get("depth", 2)):
        layers.append(nn.Dense(h["width"]))
        if norm:
            layers.append(nn.LayerNorm())
        layers.append(activation)
    return nn.Sequential(layers)


ACTIVATIONS = {"relu": nn.relu, "tanh": nn.tanh}


def num_actions(track):
    with jax.ensure_compile_time_eval():
        return game(track).action_space().num_actions


def rate(h):
    if "anneal" not in h:
        return h["learning_rate"]
    batch = h["num_envs"] * h["num_steps"]
    count = h["anneal"] // batch * h["num_minibatches"] * h["update_epochs"]
    return optax.linear_schedule(h["learning_rate"], h["learning_rate"] * 0.1, count)


def ppo(track, h):
    return PPO(
        cfg=PPOConfig(
            num_minibatches=h["num_minibatches"],
            update_epochs=h["update_epochs"],
            clip_coefficient=0.2,
            clip_value_loss=True,
            entropy_coefficient=h.get("entropy", 0.01),
            value_coefficient=0.5,
            gamma=h.get("gamma", 0.99),
            gae_lambda=0.95,
            normalize_advantage=True,
        ),
        network=Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=trunk(track, h, ACTIVATIONS[h.get("activation", "relu")])
            ),
            head=ActorCritic(
                actor=Categorical(nn.Dense(num_actions(track))), critic=nn.Dense(1)
            ),
        ),
        optimizer=optax.chain(
            optax.clip_by_global_norm(0.5), optax.adam(rate(h))
        ),
    )


def pqn(track, h):
    return PQN(
        cfg=PQNConfig(
            num_minibatches=h["num_minibatches"],
            update_epochs=h["update_epochs"],
            gamma=h.get("gamma", 0.99),
            q_lambda=h.get("q_lambda", 0.65),
        ),
        network=Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=trunk(track, h, nn.relu, norm=True)
            ),
            head=EpsilonGreedy(nn.Dense(num_actions(track))),
        ),
        exploration_schedule=optax.linear_schedule(
            1.0, h.get("epsilon", 0.05), h["explore"]
        ),
        optimizer=optax.chain(
            optax.clip_by_global_norm(10.0), optax.adam(h["learning_rate"])
        ),
    )


def dqn(track, h):
    num_envs, batch_size = h["num_envs"], h["batch_size"]
    buffer = fbx.make_trajectory_buffer(
        max_length_time_axis=h["capacity"] // num_envs,
        min_length_time_axis=max(1, h.get("warmup", batch_size) // num_envs),
        sample_batch_size=batch_size,
        add_batch_size=num_envs,
        sample_sequence_length=1,
        period=1,
    )
    return DQN(
        cfg=DQNConfig(
            updates_per_step=h["updates_per_step"],
            gamma=h.get("gamma", 0.99),
            tau=h["tau"],
        ),
        network=Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=trunk(track, h, nn.relu)
            ),
            head=EpsilonGreedy(nn.Dense(num_actions(track))),
        ),
        exploration_schedule=optax.linear_schedule(
            1.0, h.get("epsilon", 0.05), h["explore"]
        ),
        buffer=buffer,
        optimizer=optax.adam(h["learning_rate"]),
    )


def bc(track, h):
    return BC(
        cfg=BCConfig(batch_size=h["batch_size"]),
        network=Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=trunk(track, h, nn.relu)
            ),
            head=Categorical(nn.Dense(num_actions(track))),
        ),
        optimizer=optax.adam(h["learning_rate"]),
    )


def iql(track, h):
    count = num_actions(track)
    critic = Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=flattener(track),
            action_extractor=lambda action: jax.nn.one_hot(action, count),
        ),
        head=nn.vmap(
            nn.Sequential,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=2,
        )(
            [
                nn.Dense(h["width"]),
                nn.relu,
                nn.Dense(h["width"]),
                nn.relu,
                nn.Dense(1),
            ]
        ),
    )
    return IQL(
        cfg=IQLConfig(
            gamma=h.get("gamma", 0.99),
            tau=h.get("tau", 0.005),
            batch_size=h["batch_size"],
            expectile=h.get("expectile", 0.7),
            beta=h.get("beta", 3.0),
        ),
        actor=Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=trunk(track, h, nn.relu)
            ),
            head=Categorical(nn.Dense(count)),
        ),
        critic=critic,
        value=Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=trunk(track, h, nn.relu)
            ),
            head=nn.Dense(1),
        ),
        actor_optimizer=optax.adam(h["learning_rate"]),
        critic_optimizer=optax.adam(h["learning_rate"]),
        value_optimizer=optax.adam(h["learning_rate"]),
    )


PILOTS = {"ppo": ppo, "pqn": pqn, "dqn": dqn, "bc": bc, "iql": iql}


def policy(pilot, algorithm):
    if pilot == "iql":
        return algorithm.actor
    return algorithm.network


def anakin(pilot, track, h):
    algorithm = PILOTS[pilot](track, h)
    environment = wrap(game(track), h["num_envs"])
    config = AnakinConfig(num_envs=h["num_envs"], num_steps=h["num_steps"], mesh=mesh(1))
    return Anakin(config, algorithm, environment)


def sebulba(pilot, track, h):
    algorithm = PILOTS[pilot](track, h)
    environment = wrap(game(track), h["num_envs"])
    config = SebulbaConfig(
        num_envs=h["num_envs"],
        num_steps=h["num_steps"],
        learner=mesh(1),
        actor=mesh(1),
    )
    return Sebulba(config, algorithm, environment)


def quadinaros(pilot, track, h, transitions):
    algorithm = PILOTS[pilot](track, h)
    environment = wrap(game(track), h["num_envs"])
    config = QuadinarosConfig(
        num_envs=h["num_envs"], batch_shape=(h["batch_size"],), mesh=mesh(1)
    )
    return Quadinaros(config, algorithm, environment, Bundled(transitions))
