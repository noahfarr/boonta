import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boonta import environments
from boonta.environments.dune_sea import HideAndSeek, Layout
from boonta.environments.dune_sea.hide_and_seek import (EMPTY, GRAB, HIDER, JUMP, JUMP_SPEED,
                                                        LEFT, LOCK, NOOP, NUM_ACTIONS, RIGHT,
                                                        UNLOCKED)

KEY = jax.random.key(0)
TALL = 2.5


def layout(agent_x, object_x=(), wall_x=(), wall_height=(), mirrored=()):
    floats = lambda values: jnp.array(values, jnp.float32)
    return Layout(
        wall_x=floats(wall_x),
        wall_height=floats(wall_height),
        object_x=floats(object_x),
        mirrored=jnp.array(mirrored, bool),
        agent_x=floats(agent_x),
    )


def arena(agent_x, object_x=(), wall_x=(), wall_height=(), num_hiders=1, **kwargs):
    kwargs = {"preparation": 0.0, **kwargs}
    env = HideAndSeek(
        num_hiders=num_hiders,
        num_seekers=len(agent_x) - num_hiders,
        num_boxes=len(object_x),
        num_ramps=0,
        num_walls=len(wall_x),
        generator=lambda *_: layout(agent_x, object_x, wall_x, wall_height),
        **kwargs,
    )
    state, _ = env.init(KEY)
    return env, state


def play(env, state, actions, steps=1):
    step = jax.jit(env.step)
    for _ in range(steps):
        state, timestep = step(KEY, state, jnp.array(actions, jnp.int32))
    return state, timestep


def test_the_default_game_emits_what_its_spaces_promise():
    env = environments.make("dune_sea", "hide_and_seek", kwargs={})
    state, timestep = env.init(KEY)
    assert timestep.obs.shape == env.observation_space().shape
    assert timestep.action.shape == env.action_space().shape
    assert env.action_mask(state).shape == (env.num_agents, NUM_ACTIONS)
    np.testing.assert_array_equal(env.observe(state), timestep.obs)
    state, timestep = play(env, state, [NOOP] * env.num_agents)
    assert timestep.obs.shape == env.observation_space().shape
    assert timestep.reward.shape == (env.num_agents,)


def test_an_agent_carries_the_box_it_grabs_and_drops_it_on_a_second_grab():
    env, state = arena([2.0, 9.0], [2.55])
    state, _ = play(env, state, [GRAB, NOOP])
    assert int(state.holding[0]) == 0
    start = float(state.sim.polygon.position[env.first_object, 0])
    state, _ = play(env, state, [RIGHT, NOOP], steps=30)
    box, agent = float(state.sim.polygon.position[env.first_object, 0]), float(state.sim.circle.position[0, 0])
    assert box - start > 1.0
    assert abs((box - agent) - 0.55) < 0.1
    state, _ = play(env, state, [GRAB, NOOP])
    assert int(state.holding[0]) == EMPTY
    assert not bool(state.sim.joint.active[0])


def test_a_locked_box_cannot_be_pushed_or_grabbed():
    env, state = arena([2.0, 9.0], [2.55])
    state, _ = play(env, state, [LOCK, NOOP])
    assert int(state.locked_by[0]) == HIDER
    start = state.sim.polygon.position[env.first_object]
    state, _ = play(env, state, [RIGHT, NOOP], steps=30)
    np.testing.assert_allclose(state.sim.polygon.position[env.first_object], start, atol=1e-4)
    state, _ = play(env, state, [GRAB, NOOP])
    assert int(state.holding[0]) == EMPTY


def test_only_the_team_that_locked_a_box_can_unlock_it():
    env, state = arena([2.0, 3.1], [2.55])
    state, _ = play(env, state, [LOCK, NOOP])
    state, _ = play(env, state, [NOOP, LOCK])
    assert int(state.locked_by[0]) == HIDER
    state, _ = play(env, state, [LOCK, NOOP])
    assert int(state.locked_by[0]) == UNLOCKED
    assert float(state.sim.polygon.inverse_mass[env.first_object]) == pytest.approx(float(state.inverse_mass[0]))


def test_a_tall_wall_hides_the_hider_and_open_ground_does_not():
    env, state = arena([6.0, 2.0], wall_x=[4.0], wall_height=[TALL])
    _, timestep = play(env, state, [NOOP, NOOP])
    np.testing.assert_array_equal(timestep.reward, [1.0, -1.0])
    env, state = arena([6.0, 2.0])
    _, timestep = play(env, state, [NOOP, NOOP])
    np.testing.assert_array_equal(timestep.reward, [-1.0, 1.0])


def test_seekers_only_see_the_way_they_face():
    env, state = arena([2.0, 6.0])
    state, timestep = play(env, state, [NOOP, NOOP])
    assert float(timestep.reward[0]) == 1.0
    state, timestep = play(env, state, [NOOP, LEFT])
    assert float(state.facing[1]) == -1.0
    assert float(timestep.reward[0]) == -1.0


def test_seekers_wait_and_nobody_scores_while_the_hiders_prepare():
    env, state = arena([2.0, 6.0], preparation=0.5, time_limit=10)
    start = float(state.sim.circle.position[1, 0])
    for _ in range(5):
        state, timestep = play(env, state, [RIGHT, RIGHT])
        np.testing.assert_array_equal(timestep.reward, [0.0, 0.0])
    assert float(state.sim.circle.position[1, 0]) == pytest.approx(start, abs=1e-3)
    assert float(state.sim.circle.position[0, 0]) > 2.3
    state, timestep = play(env, state, [NOOP, NOOP])
    assert bool(np.all(timestep.reward != 0.0))


def test_agents_jump_from_the_ground_but_not_in_the_air():
    env, state = arena([2.0, 9.0])
    state, _ = play(env, state, [NOOP, NOOP], steps=3)
    assert bool(state.grounded[0])
    state, _ = play(env, state, [JUMP, NOOP])
    assert float(state.sim.circle.velocity[0, 1]) > 0.5 * JUMP_SPEED
    state, _ = play(env, state, [NOOP, NOOP], steps=5)
    falling = float(state.sim.circle.velocity[0, 1])
    state, _ = play(env, state, [JUMP, NOOP])
    assert float(state.sim.circle.velocity[0, 1]) < falling


def test_an_agent_pushing_into_a_wall_slides_down_instead_of_clinging():
    env, state = arena([2.7, 9.0], wall_x=[4.0], wall_height=[TALL])
    state, _ = play(env, state, [NOOP, NOOP])
    state, _ = play(env, state, [JUMP, NOOP])
    state, _ = play(env, state, [NOOP, NOOP], steps=11)
    state, _ = play(env, state, [RIGHT, NOOP], steps=20)
    assert bool(state.grounded[0])
    assert float(state.sim.circle.position[0, 1]) < 0.7


def test_the_episode_is_cut_at_the_time_limit():
    env, state = arena([2.0, 6.0], time_limit=3)
    state, timestep = play(env, state, [NOOP, NOOP], steps=2)
    assert not bool(timestep.truncated.any())
    state, timestep = play(env, state, [NOOP, NOOP])
    assert bool(timestep.truncated.all())


def test_random_play_stays_finite_and_inside_the_world():
    env = HideAndSeek(time_limit=200)
    keys = jax.random.split(KEY, 8)
    states, _ = jax.vmap(env.init)(keys)

    def turn(states, key):
        actions = jax.random.randint(key, (8, env.num_agents), 0, NUM_ACTIONS)
        states, timesteps = jax.vmap(env.step)(jax.random.split(key, 8), states, actions)
        return states, timesteps.obs

    states, observations = jax.lax.scan(jax.jit(turn), states, jax.random.split(KEY, 200))
    assert bool(jnp.isfinite(observations).all())
    positions = jnp.concatenate([states.sim.circle.position, states.sim.polygon.position[:, env.first_object :]], 1)
    assert bool(((positions[..., 0] > 0) & (positions[..., 0] < env.width) & (positions[..., 1] > 0)).all())


def test_render_can_be_cached_on_first_call_under_jit_and_shows_sightlines():
    env, state = arena([6.0, 2.0])
    scale, height = 16, 2.0

    def render(state):
        return env.render(state, scale=scale, height=height, sight=True)

    with jax.checking_leaks():
        visible = jax.jit(render)(state)
    assert visible.shape == (38, 256, 3)
    assert visible.dtype == jnp.uint8
    np.testing.assert_array_equal(render(state), visible)

    plain = env.render(state, scale=scale, height=height)
    midpoint = (visible.shape[0] - 1 - int(float(state.sim.circle.position[0, 1]) * scale), 4 * scale)
    assert visible[midpoint][0] > visible[midpoint][1]
    assert not np.array_equal(visible[midpoint], plain[midpoint])

    blocked_env, blocked_state = arena([6.0, 2.0], wall_x=[4.0], wall_height=[TALL])
    np.testing.assert_array_equal(
        blocked_env.render(blocked_state, scale=scale, height=height, sight=True),
        blocked_env.render(blocked_state, scale=scale, height=height),
    )
