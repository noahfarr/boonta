import jax
import jax.numpy as jnp
import pytest

from boonta.environments.dune_sea import TMaze


def play(env, key, policy):
    state, timestep = env.init(key)
    first = timestep.obs
    total, steps, done = 0.0, 0, False
    while not done:
        state, timestep = env.step(key, state, jnp.int32(policy(first, timestep.obs)))
        total += float(timestep.reward)
        steps += 1
        done = bool(timestep.terminated)
    return total, steps


@pytest.mark.parametrize("length", [1, 4, 10])
def test_remembering_the_cue_always_pays_and_takes_length_plus_one_steps(length):
    env = TMaze(length=length)
    for seed in range(8):
        total, steps = play(env, jax.random.key(seed), lambda first, obs: first[:2].argmax())
        assert total == 1.0
        assert steps == length + 1 == env.time_limit()


def test_the_corridor_hides_the_cue():
    env = TMaze(length=6)
    seen = {}
    for cue in (0, 1):
        state, _ = env.init(jax.random.key(0))
        state = state.replace(cue=jnp.int32(cue))
        trail = [env.observe(state)]
        for _ in range(env.length):
            state, timestep = env.step(jax.random.key(0), state, jnp.int32(0))
            trail.append(timestep.obs)
        seen[cue] = jnp.stack(trail)
    assert not jnp.array_equal(seen[0][0], seen[1][0])
    assert jnp.array_equal(seen[0][1:], seen[1][1:])


def test_a_memoryless_policy_wins_half_the_time():
    env = TMaze(length=3)
    wins = sum(play(env, jax.random.key(seed), lambda first, obs: 0)[0] for seed in range(200))
    assert 70 < wins < 130
