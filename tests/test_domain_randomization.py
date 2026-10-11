import jax
import jax.numpy as jnp
import numpy as np

from boonta.curricula import domain_randomization
from boonta.environments import gymnax
from boonta.environments.wrappers import SameStepAutoReset
from dummies import Dial


class Blink(Dial):
    def step(self, key, state, action):
        state, timestep = super().step(key, state, action)
        return state, timestep.replace(terminated=state.clock >= 2.0)


def randomized(num_envs=8):
    _, environment, _, _ = domain_randomization(
        None,
        SameStepAutoReset(Blink(), num_envs),
        sample=lambda key: jax.random.uniform(key, minval=1.0, maxval=2.0),
    )
    return environment


def params(state):
    (dial,) = [
        leaf
        for leaf in jax.tree.leaves(state, is_leaf=lambda leaf: hasattr(leaf, "params"))
        if hasattr(leaf, "params")
    ]
    return np.asarray(dial.params)


def test_every_environment_starts_on_its_own_draw():
    environment = randomized()
    state, _ = environment.init(jax.random.key(0))
    drawn = params(state)
    assert np.all((drawn >= 1.0) & (drawn < 2.0))
    assert len(np.unique(drawn)) == len(drawn)


def test_a_draw_lasts_its_episode_and_the_next_episode_draws_again():
    environment = randomized()
    action = jnp.zeros(8, jnp.int32)
    state, _ = environment.init(jax.random.key(0))
    first = params(state)
    state, timestep = environment.step(jax.random.key(1), state, action)
    assert not np.asarray(timestep.done).any()
    np.testing.assert_array_equal(params(state), first)
    state, timestep = environment.step(jax.random.key(2), state, action)
    assert np.asarray(timestep.done).all()
    second = params(state)
    assert np.all((second >= 1.0) & (second < 2.0))
    assert not np.any(second == first)


def test_a_gymnax_draw_sets_the_physics_of_the_next_episode():
    sample = gymnax.params_sampler("CartPole-v1", {"gravity": (5.0, 20.0)})
    environment = gymnax.make("CartPole-v1")
    state, _ = environment.init(jax.random.key(0))
    theta = sample(jax.random.key(1))
    state = environment.update(state, jax.random.key(2), theta=theta)
    assert 5.0 <= float(state.params.gravity) < 20.0
    assert state.params.gravity == theta.gravity
