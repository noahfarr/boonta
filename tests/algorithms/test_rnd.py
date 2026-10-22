from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pytest
from flax import struct

import zoo
from boonta.algorithms.advantage_estimators import generalized_advantage_estimator
from boonta.algorithms.wrappers import RND
from boonta.utils import Timestep, Transition
from dummies import corridor

GAMMA = 0.99
LAMBDA = 0.95


@struct.dataclass
class ToyState:
    step: jnp.ndarray
    params: jnp.ndarray


@dataclass
class Toy:
    record: list = None
    seen: list = None
    advantage_estimator: callable = generalized_advantage_estimator

    def init(self, key, timestep):
        return ToyState(step=jnp.array(0), params=jnp.zeros(2)), None

    def step(self, state, carry, key, timestep, temperature=1.0):
        envs = timestep.reward.shape[0]
        return state, carry, jnp.zeros(envs, jnp.int32), {}

    def update(self, state, key, transitions):
        steps, envs = transitions.second.reward.shape
        if self.seen is not None:
            self.seen.append(state.step)
        self.record.append(
            self.advantage_estimator(
                transitions, jnp.zeros((steps, envs)), jnp.zeros(envs), GAMMA, LAMBDA
            )
        )
        return state.replace(step=state.step + 1)


class Embed(nn.Module):
    width: int = 4

    @nn.compact
    def __call__(self, obs):
        return nn.Dense(self.width)(obs)


class Scalar(nn.Module):
    @nn.compact
    def __call__(self, obs):
        return nn.Dense(1)(obs)


def frame(envs, obs):
    zeros = jnp.zeros(envs)
    return Timestep(
        obs=obs,
        action=jnp.zeros(envs, jnp.int32),
        reward=zeros,
        terminated=jnp.zeros(envs, bool),
        truncated=jnp.zeros(envs, bool),
        info={},
    )


def rollout(steps, envs, obs, reward, terminated=None):
    zeros = jnp.zeros((steps, envs), bool)
    stamp = Timestep(
        obs=obs,
        action=jnp.zeros((steps, envs), jnp.int32),
        reward=jnp.asarray(reward, jnp.float32),
        terminated=zeros if terminated is None else terminated,
        truncated=zeros,
        info={},
    )
    return Transition(
        first=stamp, second=stamp, aux={"value": jnp.zeros((steps, envs))}
    )


def build(record, coefficient=1.0, proportion=1.0, epochs=1, num_minibatches=1):
    return RND(
        algorithm=Toy(record=record),
        random_network=Embed(),
        predictor_network=Embed(),
        critic_network=Scalar(),
        optimizer=optax.sgd(1e-2),
        intrinsic_coefficient=coefficient,
        update_proportion=proportion,
        update_epochs=epochs,
        num_minibatches=num_minibatches,
    )


def sighted(state, transitions):
    return state.obs_normalizer.absorb(transitions.second.obs, axis=(0, 1))


def errors_of(env, state, transitions):
    return env.distillation_error(
        state.random_network_params,
        state.predictor_params,
        transitions.second.obs,
        sighted(state, transitions),
    )


def moved(before, after):
    return any(
        bool(jnp.any(a != b))
        for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after))
    )


def test_target_and_predictor_start_apart_so_the_error_is_positive():
    env = build([])
    key = jax.random.key(0)
    obs = jax.random.normal(key, (2, 3, 4))
    transitions = rollout(2, 3, obs, jnp.zeros((2, 3)))
    state, _ = env.init(key, frame(3, obs[0]))
    assert float(jnp.mean(errors_of(env, state, transitions))) > 0.0


def test_the_bonus_reaches_the_actor_through_the_advantage():
    record = []
    env = build(record)
    key = jax.random.key(0)
    obs = jax.random.normal(key, (2, 3, 4))
    state, _ = env.init(key, frame(3, obs[0]))
    env.update(state, key, rollout(2, 3, obs, jnp.zeros((2, 3))))
    advantages, _ = record[0]
    assert bool(jnp.any(advantages != 0.0))


def test_the_extrinsic_critic_target_never_sees_the_bonus():
    record = []
    env = build(record)
    key = jax.random.key(0)
    obs = jax.random.normal(key, (2, 3, 4))
    state, _ = env.init(key, frame(3, obs[0]))
    env.update(state, key, rollout(2, 3, obs, jnp.zeros((2, 3))))
    _, returns = record[0]
    assert returns == pytest.approx(jnp.zeros((2, 3)), abs=1e-6)


def test_zero_coefficient_leaves_the_advantage_untouched():
    record = []
    env = build(record, coefficient=0.0)
    key = jax.random.key(0)
    obs = jax.random.normal(key, (2, 3, 4))
    transitions = rollout(2, 3, obs, jnp.full((2, 3), 0.5))
    state, _ = env.init(key, frame(3, obs[0]))
    env.update(state, key, transitions)
    advantages, _ = record[0]
    plain, _ = generalized_advantage_estimator(
        transitions, jnp.zeros((2, 3)), jnp.zeros(3), GAMMA, LAMBDA
    )
    assert advantages == pytest.approx(plain, abs=1e-6)


def test_updating_does_not_rewire_the_wrapped_algorithm():
    env = build([])
    key = jax.random.key(0)
    obs = jax.random.normal(key, (2, 3, 4))
    state, _ = env.init(key, frame(3, obs[0]))
    env.update(state, key, rollout(2, 3, obs, jnp.zeros((2, 3))))
    assert env.algorithm.advantage_estimator is generalized_advantage_estimator


def test_no_bonus_is_paid_for_a_state_the_reset_handed_over():
    env = build([])
    key = jax.random.key(0)
    obs = jax.random.normal(key, (4, 3, 4))
    terminated = jnp.zeros((4, 3), bool).at[1].set(True)
    transitions = rollout(4, 3, obs, jnp.zeros((4, 3)), terminated)
    state, _ = env.init(key, frame(3, obs[0]))
    bonus = env.intrinsic_reward(state, transitions, errors_of(env, state, transitions))
    assert bonus[1] == pytest.approx(jnp.zeros(3), abs=1e-8)
    assert bool(jnp.all(bonus[0] > 0.0))
    assert bool(jnp.all(bonus[2] > 0.0))


def test_the_predictor_still_trains_on_the_states_it_is_not_paid_for():
    env = build([])
    key = jax.random.key(0)
    obs = jax.random.normal(key, (4, 3, 4))
    transitions = rollout(4, 3, obs, jnp.zeros((4, 3)), jnp.ones((4, 3), bool))
    state, _ = env.init(key, frame(3, obs[0]))
    bonus = env.intrinsic_reward(state, transitions, errors_of(env, state, transitions))
    after = env.update(state, key, transitions)
    assert bool(jnp.all(bonus == 0.0))
    assert moved(state.predictor_params, after.predictor_params)


def test_the_intrinsic_advantage_ignores_termination():
    env = build([])
    key = jax.random.key(0)
    obs = jax.random.normal(key, (4, 3, 4))
    terminated = jnp.zeros((4, 3), bool).at[1].set(True)
    transitions = rollout(4, 3, obs, jnp.zeros((4, 3)), terminated)
    state, _ = env.init(key, frame(3, obs[0]))
    sight = sighted(state, transitions)
    bonus = jnp.ones((4, 3))
    values = env.intrinsic_value(state.critic_params, transitions.first.obs, sight)
    bootstrap = env.intrinsic_value(state.critic_params, transitions.second.obs, sight)[
        -1
    ]
    intrinsic, _ = env.intrinsic_advantage_estimator(bonus, values, bootstrap)

    def gae(terminated):
        second = transitions.second.replace(reward=bonus, terminated=terminated)
        return generalized_advantage_estimator(
            transitions.replace(second=second),
            values,
            bootstrap,
            env.intrinsic_gamma,
            env.intrinsic_lambda,
        )[0]

    flat = gae(jnp.zeros((4, 3), bool))
    episodic = gae(terminated)
    assert intrinsic == pytest.approx(flat, abs=1e-6)
    assert bool(jnp.any(jnp.abs(intrinsic - episodic) > 1e-3))


def test_the_target_never_moves_and_the_predictor_does():
    env = build([])
    key = jax.random.key(0)
    obs = jax.random.normal(key, (2, 3, 4))
    state, _ = env.init(key, frame(3, obs[0]))
    after = env.update(state, key, rollout(2, 3, obs, jnp.zeros((2, 3))))
    assert not moved(state.random_network_params, after.random_network_params)
    assert moved(state.predictor_params, after.predictor_params)


def test_training_the_predictor_drives_the_error_down():
    env = build([])
    key = jax.random.key(0)
    obs = jax.random.normal(key, (2, 3, 4))
    transitions = rollout(2, 3, obs, jnp.zeros((2, 3)))
    state, _ = env.init(key, frame(3, obs[0]))
    before = float(jnp.mean(errors_of(env, state, transitions)))
    for _ in range(200):
        state = env.update(state, key, transitions)
    after = float(jnp.mean(errors_of(env, state, transitions)))
    assert after < before


def test_subsampling_the_batch_still_trains_the_predictor():
    env = build([], proportion=0.25)
    key = jax.random.key(0)
    obs = jax.random.normal(key, (2, 3, 4))
    state, _ = env.init(key, frame(3, obs[0]))
    after = env.update(state, key, rollout(2, 3, obs, jnp.zeros((2, 3))))
    assert moved(state.predictor_params, after.predictor_params)


def test_more_epochs_move_the_predictor_further():
    key = jax.random.key(0)
    obs = jax.random.normal(key, (4, 6, 4))
    transitions = rollout(4, 6, obs, jnp.zeros((4, 6)))

    def travelled(epochs):
        env = build([], epochs=epochs, num_minibatches=2)
        state, _ = env.init(key, frame(6, obs[0]))
        after = env.update(state, key, transitions)
        return sum(
            float(jnp.sum(jnp.abs(a - b)))
            for a, b in zip(
                jax.tree.leaves(state.predictor_params),
                jax.tree.leaves(after.predictor_params),
            )
        )

    assert travelled(8) > travelled(1)


def test_minibatches_must_divide_the_environments():
    env = build([], num_minibatches=4)
    key = jax.random.key(0)
    obs = jax.random.normal(key, (2, 6, 4))
    state, _ = env.init(key, frame(6, obs[0]))
    with pytest.raises(AssertionError, match="6 environments"):
        env.update(state, key, rollout(2, 6, obs, jnp.zeros((2, 6))))


def test_the_normaliser_tracks_the_discounted_return_not_the_raw_error():
    env = build([])
    key = jax.random.key(0)
    obs = jax.random.normal(key, (8, 4, 4))
    transitions = rollout(8, 4, obs, jnp.zeros((8, 4)))
    state, _ = env.init(key, frame(4, obs[0]))
    raw = errors_of(env, state, transitions)
    after = env.update(state, key, transitions)

    _, trailed = env.accumulate(state, raw)
    _, spread = after.reward_normalizer.settled()
    assert float(spread) == pytest.approx(float(jnp.var(trailed)), rel=1e-3)
    assert float(jnp.var(trailed)) > float(jnp.var(raw))


def test_the_accumulator_survives_across_updates():
    env = build([])
    key = jax.random.key(0)
    obs = jax.random.normal(key, (4, 3, 4))
    transitions = rollout(4, 3, obs, jnp.zeros((4, 3)))
    state, _ = env.init(key, frame(3, obs[0]))
    assert float(jnp.max(jnp.abs(state.intrinsic_returns))) == 0.0
    once = env.update(state, key, transitions)
    twice = env.update(once, key, transitions)
    assert float(jnp.min(once.intrinsic_returns)) > 0.0
    assert float(jnp.min(twice.intrinsic_returns)) > float(
        jnp.min(once.intrinsic_returns)
    )


def test_observation_statistics_accumulate():
    env = build([])
    key = jax.random.key(0)
    obs = jax.random.normal(key, (2, 3, 4)) + 5.0
    state, _ = env.init(key, frame(3, obs[0]))
    assert float(state.obs_normalizer.steps) == 0.0
    after = env.update(state, key, rollout(2, 3, obs, jnp.zeros((2, 3))))
    assert float(after.obs_normalizer.steps) == 1.0
    centre, _ = after.obs_normalizer.settled()
    assert float(jnp.mean(centre)) > 4.0


def test_state_reads_forward_to_the_wrapped_algorithm():
    env = build([])
    key = jax.random.key(0)
    obs = jax.random.normal(key, (2, 3, 4))
    state, _ = env.init(key, frame(3, obs[0]))
    assert state.params is state.algorithm_state.params


def test_the_step_the_podracer_counts_reaches_the_wrapped_algorithm():
    seen = []
    env = build([])
    env.algorithm.seen = seen
    key = jax.random.key(0)
    obs = jax.random.normal(key, (2, 3, 4))
    state, _ = env.init(key, frame(3, obs[0]))
    state = state.replace(step=state.step + 7)
    env.update(state, key, rollout(2, 3, obs, jnp.zeros((2, 3))))
    assert int(seen[0]) == 7


def exploring(algorithm, environment, num_envs, num_steps):
    explorer = RND(
        algorithm=algorithm,
        random_network=Embed(),
        predictor_network=Embed(),
        critic_network=Scalar(),
        optimizer=optax.adam(1e-3),
        intrinsic_coefficient=0.5,
        update_epochs=2,
        num_minibatches=2,
    )
    return zoo.online(explorer, environment, num_envs, num_steps)


def test_a_real_ppo_trains_under_the_wrapper():
    podracer = zoo.ppo(corridor(), podracer=exploring)
    state = podracer.init(jax.random.key(0))
    before = jax.tree.map(jnp.copy, state.algorithm_state.predictor_params)
    state, _ = podracer.train(state, jax.random.key(1), 3)
    assert moved(before, state.algorithm_state.predictor_params)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(state.algorithm_state.params))
