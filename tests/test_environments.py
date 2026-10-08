from functools import partial

import jax
import jax.numpy as jnp
import lox
import numpy as np
import pytest
from flax import struct

import zoo
from boonta.environments.gymnasium import Gymnasium, convert, make
from boonta.environments import wrappers
from boonta.environments.spaces import Space
from boonta.environments.wrappers import (MCP, PBRS, Batched, ClipAction,
                                          ClipReward, DomainRandomization,
                                          FlattenObservation, GroupedAutoReset,
                                          LogAction, LogEnvState, LogInfo,
                                          MaskObservation, NextStepAutoReset,
                                          NormalizeObservation, NormalizeReward,
                                          Opponent, OptimisticAutoReset,
                                          PadAction, PadObservation, Prompt,
                                          Reasoning, RecordEpisodeStatistics,
                                          RecordMultiAgentEpisodeStatistics,
                                          RecordRestartStatistics,
                                          SameStepAutoReset, Stagger, StickyAction,
                                          TimeAwareObservation, TimeLimit,
                                          TransformAction, TransformObservation,
                                          TransformReward, Vectorize, Wrapper)
from boonta.utils import Timestep, sharded

from dummies import Corridor, Dial, Team, recall, reach

NUM_ENVS = 8


def identity(value):
    return value


class Lever(Dial):
    def action_space(self) -> Space:
        return Space((1,), jnp.float32, -1.0, 1.0)


STACKS = [
    pytest.param(lambda: Vectorize(Dial(), NUM_ENVS), id="vectorize"),
    pytest.param(lambda: Vectorize(SameStepAutoReset(Dial()), NUM_ENVS), id="same_step_auto_reset"),
    pytest.param(lambda: Vectorize(NextStepAutoReset(Dial()), NUM_ENVS), id="next_step_auto_reset"),
    pytest.param(lambda: OptimisticAutoReset(Dial(), NUM_ENVS, ratio=2), id="optimistic_auto_reset"),
    pytest.param(
        lambda: GroupedAutoReset(Vectorize(SameStepAutoReset(Dial()), NUM_ENVS), num_steps=4),
        id="grouped_auto_reset",
    ),
    pytest.param(lambda: RecordEpisodeStatistics(Vectorize(Dial(), NUM_ENVS)), id="record_episode_statistics"),
    pytest.param(lambda: Vectorize(TimeLimit(Dial(), 100), NUM_ENVS), id="time_limit"),
    pytest.param(lambda: Vectorize(Stagger(SameStepAutoReset(Dial()), spread=4), NUM_ENVS), id="stagger"),
    pytest.param(lambda: Vectorize(StickyAction(Dial()), NUM_ENVS), id="sticky_action"),
    pytest.param(lambda: Vectorize(TimeAwareObservation(Dial(), 100), NUM_ENVS), id="time_aware_observation"),
    pytest.param(lambda: Vectorize(PBRS(Dial(), jnp.sum, gamma=0.9), NUM_ENVS), id="pbrs"),
    pytest.param(lambda: Vectorize(Reasoning(Dial(), num_tokens=2), NUM_ENVS), id="reasoning"),
    pytest.param(lambda: NormalizeObservation(Vectorize(Dial(), NUM_ENVS)), id="normalize_observation"),
    pytest.param(lambda: NormalizeReward(Vectorize(Dial(), NUM_ENVS)), id="normalize_reward"),
    pytest.param(lambda: Vectorize(ClipAction(Dial()), NUM_ENVS), id="clip_action"),
    pytest.param(lambda: Vectorize(ClipReward(Dial()), NUM_ENVS), id="clip_reward"),
    pytest.param(lambda: Vectorize(FlattenObservation(Dial()), NUM_ENVS), id="flatten_observation"),
    pytest.param(lambda: Vectorize(LogAction(Dial()), NUM_ENVS), id="log_action"),
    pytest.param(lambda: Vectorize(LogEnvState(Dial()), NUM_ENVS), id="log_env_state"),
    pytest.param(lambda: LogInfo(Vectorize(Dial(), NUM_ENVS)), id="log_info"),
    pytest.param(
        lambda: Vectorize(MaskObservation(Dial(), jnp.ones(3)), NUM_ENVS), id="mask_observation"
    ),
    pytest.param(lambda: Vectorize(TransformAction(Dial(), identity), NUM_ENVS), id="transform_action"),
    pytest.param(
        lambda: Vectorize(TransformObservation(Dial(), identity), NUM_ENVS),
        id="transform_observation",
    ),
    pytest.param(lambda: Vectorize(TransformReward(Dial(), identity), NUM_ENVS), id="transform_reward"),
    pytest.param(lambda: Batched(Vectorize(Dial(), NUM_ENVS), NUM_ENVS), id="batched"),
    pytest.param(
        lambda: Vectorize(DomainRandomization(Dial(), lambda key, params: params + 1.0), NUM_ENVS),
        id="domain_randomization",
    ),
    pytest.param(lambda: Vectorize(tool(Dial()), NUM_ENVS), id="mcp"),
    pytest.param(lambda: Vectorize(Prompt(Dial(), np.arange(2), pad=0), NUM_ENVS), id="prompt"),
    pytest.param(lambda: Opponent(Vectorize(Team(Dial(), 2), NUM_ENVS), rival, blank), id="opponent"),
    pytest.param(lambda: Vectorize(PadAction(Lever(), 3), NUM_ENVS), id="pad_action"),
    pytest.param(
        lambda: Vectorize(
            PadObservation(TransformObservation(Dial(), lambda obs: {"dial": obs}), "dial", 5),
            NUM_ENVS,
        ),
        id="pad_observation",
    ),
    pytest.param(
        lambda: RecordMultiAgentEpisodeStatistics(Vectorize(Team(Dial(), 2), NUM_ENVS)),
        id="record_multi_agent_episode_statistics",
    ),
    pytest.param(
        lambda: RecordRestartStatistics(
            Vectorize(TransformObservation(Dial(), lambda obs: obs.astype(jnp.int32)), NUM_ENVS),
            room_byte=0,
        ),
        id="record_restart_statistics",
    ),
    pytest.param(
        lambda: RecordEpisodeStatistics(LogInfo(Vectorize(TimeLimit(Dial(), 100), NUM_ENVS))),
        id="a_mixed_stack",
    ),
]


def idle(environment):
    space = environment.action_space()
    agents = (environment.num_agents,) if environment.num_agents > 1 else ()
    return jnp.zeros((NUM_ENVS, *agents, *space.shape), space.dtype)


def test_the_stacks_cover_every_wrapper():
    covered = set()
    for param in STACKS:
        (build,) = param.values
        environment = build()
        while isinstance(environment, Wrapper):
            covered.add(type(environment))
            environment = environment._env
    exported = {getattr(wrappers, name) for name in wrappers.__all__}
    expected = {kind for kind in exported if isinstance(kind, type) and issubclass(kind, Wrapper)}
    assert expected - covered == {Wrapper}


@pytest.mark.parametrize("build", STACKS)
def test_reconfiguration_reaches_the_game_through_every_wrapper(build):
    environment = build()
    state, _ = environment.init(jax.random.key(0))
    state = environment.update(state, setting=7.0)
    _, timestep = environment.step(jax.random.key(1), state, idle(environment))
    np.testing.assert_array_equal(timestep.info["clock"] > 0, True)
    unwrapped = jax.tree.leaves(state, is_leaf=lambda leaf: hasattr(leaf, "setting"))
    settings = [leaf.setting for leaf in unwrapped if hasattr(leaf, "setting")]
    assert settings
    for setting in settings:
        np.testing.assert_array_equal(setting, 7.0)


@pytest.mark.parametrize("build", STACKS)
def test_every_wrapper_shows_the_action_mask_of_the_game(build):
    environment = build()
    state, _ = environment.init(jax.random.key(0))
    state = environment.update(state, setting=7.0)
    mask = environment.action_mask(state)
    if jnp.issubdtype(environment.action_space().dtype, jnp.integer):
        assert mask.shape[:-1] == idle(environment).shape
    mask = mask[..., :2]
    shown = [True, True] if environment.wraps(MCP) else [True, False]
    np.testing.assert_array_equal(mask, np.broadcast_to(shown, mask.shape))


def test_log_flags_passes_reconfiguration_and_the_action_mask_through():
    from boonta.environments.peanut_gb import pokemon_red

    environment = pokemon_red.LogFlags(Dial())
    state, _ = environment.init(jax.random.key(0))
    state = environment.update(state, setting=7.0)
    assert float(state.setting) == 7.0
    np.testing.assert_array_equal(environment.action_mask(state), [True, False])


def play(environment, state, actions, step=None):
    step = step or environment.step
    timesteps = []
    for index, action in enumerate(actions):
        state, timestep = step(jax.random.key(index + 1), state, action)
        timesteps.append(timestep)
    return state, timesteps


def cue(state):
    return state.cue


def test_same_step_auto_reset_reports_the_end_and_shows_the_next_start():
    environment = SameStepAutoReset(recall())
    state, timestep = environment.init(jax.random.key(0))
    answer = state.cue
    state, (_, _, last) = play(environment, state, [answer] * 3)

    assert bool(last.terminated) and float(last.reward) == 1.0
    assert int(state.clock) == 0
    np.testing.assert_array_equal(last.obs, environment.observe(state))


def test_next_step_auto_reset_spends_one_empty_step_on_the_reset():
    environment = NextStepAutoReset(recall())
    state, _ = environment.init(jax.random.key(0))
    answer = state.env_state.cue
    state, (*_, last, reset) = play(environment, state, [answer] * 4)

    assert bool(last.terminated) and float(last.reward) == 1.0
    assert not bool(reset.terminated) and float(reset.reward) == 0.0
    assert int(state.env_state.clock) == 0


def test_next_step_auto_reset_restarts_a_team_once_every_agent_is_done():
    environment = NextStepAutoReset(TimeLimit(Team(Dial(), 2), 2))
    state, _ = environment.init(jax.random.key(0))
    state, (_, last, reset) = play(environment, state, [jnp.zeros(2, jnp.int32)] * 3)

    assert bool(last.truncated.all())
    assert not bool(reset.done.any())
    np.testing.assert_array_equal(state.env_state.env_state.clock, 0)


def test_optimistic_auto_reset_restarts_every_finished_environment():
    environment = OptimisticAutoReset(recall(), NUM_ENVS, ratio=2)
    state, _ = environment.init(jax.random.key(0))
    state, timesteps = play(environment, state, [jnp.zeros(NUM_ENVS, jnp.int32)] * 3)
    assert bool(timesteps[-1].terminated.all())
    np.testing.assert_array_equal(state.clock, 0)


def test_grouped_auto_reset_restarts_each_group_from_one_start():
    environment = GroupedAutoReset(
        Vectorize(SameStepAutoReset(Dial()), NUM_ENVS), num_steps=4, group_size=2
    )
    state, _ = environment.init(jax.random.key(0))
    state, timesteps = play(environment, state, [jnp.zeros(NUM_ENVS, jnp.int32)] * 4)
    noise = np.asarray(state.env_state.noise).reshape(-1, 2)

    assert bool(timesteps[-1].truncated.all()) and not bool(timesteps[-1].terminated.any())
    np.testing.assert_array_equal(state.env_state.clock, 0)
    np.testing.assert_array_equal(noise[:, 0], noise[:, 1])
    assert len(np.unique(noise[:, 0])) > 1


def statistics(environment, actions):
    def run(key):
        state, _ = environment.init(key)

        def step(state, inputs):
            key, action = inputs
            state, timestep = environment.step(key, state, action)
            return state, timestep.reward

        keys = jax.random.split(key, len(actions))
        return jax.lax.scan(step, state, (keys, jnp.stack(actions)))

    (_, rewards), logs = lox.spool(run)(jax.random.key(0))
    finished = {
        name: np.asarray(values).ravel()[~np.isnan(np.asarray(values).ravel())]
        for name, values in logs.items()
    }
    return np.asarray(rewards), finished


def test_episode_statistics_log_each_finished_episode_once():
    gamma = 0.5
    environment = RecordEpisodeStatistics(
        Vectorize(SameStepAutoReset(Corridor()), NUM_ENVS), gamma=gamma
    )
    forward = [jnp.full(NUM_ENVS, step % 3 % 2, jnp.int32) for step in range(9)]
    rewards, logs = statistics(environment, forward)

    np.testing.assert_array_equal(logs["episode_statistics/episode_return"], 1.0)
    np.testing.assert_array_equal(logs["episode_statistics/episode_length"], 3)
    np.testing.assert_allclose(
        logs["episode_statistics/discounted_episode_return"], gamma**2
    )
    assert len(logs["episode_statistics/episode_return"]) == 3 * NUM_ENVS
    assert logs["episode_statistics/episode_return"].sum() == rewards.sum()


def test_every_grouped_window_cut_is_logged_as_an_episode():
    environment = RecordEpisodeStatistics(
        GroupedAutoReset(Vectorize(SameStepAutoReset(Corridor()), NUM_ENVS), num_steps=4)
    )
    rewards, logs = statistics(environment, [jnp.ones(NUM_ENVS, jnp.int32)] * 16)
    assert len(logs["episode_statistics/episode_return"]) == 4 * NUM_ENVS
    assert logs["episode_statistics/episode_return"].sum() == rewards.sum()


def test_a_time_limit_truncates_once_and_never_overrides_a_termination():
    environment = TimeLimit(Dial(), 3)
    state, _ = environment.init(jax.random.key(0))
    _, timesteps = play(environment, state, [jnp.int32(0)] * 3)
    np.testing.assert_array_equal([bool(t.truncated) for t in timesteps], [False, False, True])

    environment = TimeLimit(recall(), 3)
    state, _ = environment.init(jax.random.key(0))
    _, timesteps = play(environment, state, [jnp.int32(0)] * 3)
    assert bool(timesteps[-1].terminated) and not bool(timesteps[-1].truncated)


def test_stagger_cuts_each_environment_once_inside_its_spread():
    spread = 6
    environment = Vectorize(Stagger(SameStepAutoReset(Dial()), spread=spread), 64)
    state, _ = environment.init(jax.random.key(0))
    budgets = np.asarray(state.budget)
    assert budgets.min() >= 0 and budgets.max() < spread and len(np.unique(budgets)) > 1

    state, timesteps = play(environment, state, [jnp.zeros(64, jnp.int32)] * 3 * spread, jax.jit(environment.step))
    cuts = np.stack([np.asarray(t.truncated) for t in timesteps]).sum(0)
    np.testing.assert_array_equal(cuts, 1)
    assert len(np.unique(np.asarray(state.env_state.clock))) > 1


def test_stagger_never_marks_a_natural_end_or_logs_its_own_cut():
    environment = Vectorize(
        Stagger(RecordEpisodeStatistics(SameStepAutoReset(recall())), spread=6), 64
    )
    _, logs = statistics(environment, [jnp.zeros(64, jnp.int32)] * 12)
    np.testing.assert_array_equal(logs["episode_statistics/episode_length"], 3)


def test_normalized_observations_have_zero_mean_and_unit_variance():
    environment = NormalizeObservation(Vectorize(Dial(), 256))
    state, _ = environment.init(jax.random.key(0))
    state, timesteps = play(environment, state, [jnp.zeros(256, jnp.int32)] * 20)
    noise = np.asarray(timesteps[-1].obs[:, 2])
    assert abs(noise.mean()) < 0.1 and abs(noise.std() - 1.0) < 0.1


def test_normalized_rewards_divide_by_the_spread_of_returns():
    environment = NormalizeReward(Vectorize(SameStepAutoReset(reach()), 256), gamma=0.0)
    state, _ = environment.init(jax.random.key(0))
    state, timesteps = play(environment, state, [jnp.zeros((256, 2))] * 30)
    rewards = np.asarray(timesteps[-1].reward)
    assert abs(rewards.std() - 1.0) < 0.2


def test_each_agent_normalizes_its_rewards_by_its_own_spread():
    scales = jnp.array([1.0, 10.0])
    team = TransformReward(Team(Dial(), 2), lambda reward: reward + scales)
    environment = NormalizeReward(Vectorize(team, NUM_ENVS))
    state, timestep = environment.init(jax.random.key(0))
    state, timesteps = play(environment, state, [jnp.zeros((NUM_ENVS, 2), jnp.int32)] * 5)
    rewards = np.asarray(timesteps[-1].reward)
    assert rewards.shape == (NUM_ENVS, 2)
    np.testing.assert_allclose(rewards[:, 0], rewards[:, 1], rtol=1e-4)


def test_clip_action_clips_what_runs_and_reports_what_was_asked():
    environment = ClipAction(reach())
    state, _ = environment.init(jax.random.key(0))
    _, timestep = environment.step(jax.random.key(1), state, jnp.full(2, 5.0))
    np.testing.assert_array_equal(timestep.action, 5.0)
    np.testing.assert_allclose(timestep.reward, 1.0 - jnp.mean(jnp.abs(1.0 - state.cue)))


def test_a_certain_sticky_action_repeats_the_first_one():
    environment = StickyAction(Corridor(), probability=1.0)
    state, _ = environment.init(jax.random.key(0))
    state, timesteps = play(environment, state, [jnp.int32(0), jnp.int32(1), jnp.int32(0)])
    assert int(state.action) == 0
    np.testing.assert_array_equal([int(t.action) for t in timesteps], [0, 1, 0])
    assert int(state.env_state.position) == 1


def test_time_aware_observations_count_up_and_restart_with_the_episode():
    environment = SameStepAutoReset(TimeAwareObservation(recall(), time_limit=4))
    state, _ = environment.init(jax.random.key(0))
    _, timesteps = play(environment, state, [jnp.int32(0)] * 4)
    np.testing.assert_allclose([float(t.obs[-1]) for t in timesteps], [-0.25, 0.0, -0.5, -0.25])


def test_pbrs_adds_the_discounted_change_in_potential():
    gamma = 0.9
    environment = PBRS(Dial(), lambda obs: obs[0], gamma=gamma)
    state, _ = environment.init(jax.random.key(0))
    _, timesteps = play(environment, state, [jnp.int32(0)] * 3)
    np.testing.assert_allclose(
        [float(t.reward) for t in timesteps],
        [gamma * 1 - 0, gamma * 2 - 1, gamma * 3 - 2],
        rtol=1e-6,
    )


def test_a_reasoning_token_freezes_the_game_and_costs_its_price():
    environment = Reasoning(Dial(), num_tokens=2, cost=0.5)
    state, _ = environment.init(jax.random.key(0))
    state, (thought, action) = play(environment, state, [jnp.int32(2), jnp.int32(0)])
    assert float(thought.reward) == -0.5 and float(thought.info["clock"]) == 0.0
    assert float(action.info["clock"]) == 1.0
    assert environment.action_space().num_actions == 4


def tool(environment, start=0, end=0):
    return MCP(
        environment,
        to_action=lambda arguments, cursor: jnp.int32(0),
        to_tokens=lambda obs: obs.astype(jnp.int32),
        start=start,
        end=end,
        pad=0,
        vocab_size=4,
        capacity=4,
        observation_shape=(3,),
    )


def test_a_tool_call_reports_the_truncation_of_the_step_it_fires():
    environment = tool(TimeLimit(Dial(), 2), start=1, end=2)
    state, _ = environment.init(jax.random.key(0))
    _, timesteps = play(environment, state, [jnp.int32(token) for token in (1, 2, 1, 2)])
    np.testing.assert_array_equal([bool(t.truncated) for t in timesteps], [False, False, False, True])


def test_reading_the_prompt_never_ends_the_episode():
    environment = Prompt(TimeLimit(Dial(), 1), np.arange(7), pad=0)
    state, _ = environment.init(jax.random.key(0))
    _, timesteps = play(environment, state, [jnp.int32(0)] * 3)
    np.testing.assert_array_equal([bool(t.truncated) for t in timesteps], [False, False, True])


def test_every_wrapper_reports_the_time_limit_of_the_game_it_wraps():
    from boonta.environments import gymnax

    cartpole = gymnax.make("CartPole-v1", params={"max_steps_in_episode": 7})
    assert RecordEpisodeStatistics(Vectorize(SameStepAutoReset(cartpole), 2)).time_limit() == 7
    assert Vectorize(TimeLimit(cartpole, 5), 2).time_limit() == 5

    with pytest.raises(NotImplementedError):
        Vectorize(Dial(), 2).time_limit()


def test_prompt_and_tool_calls_stretch_the_time_limit():
    calls = tool(TimeLimit(Dial(), 4))
    assert calls.time_limit() == 4 * 32
    assert Prompt(calls, np.arange(7), pad=0).time_limit() == 4 * 32 + 3


def test_flatten_observation_flattens_the_space_too():
    environment = FlattenObservation(TransformObservation(Dial(), lambda obs: obs.reshape(3, 1)))
    _, timestep = environment.init(jax.random.key(0))
    assert timestep.obs.shape == (3,)
    assert environment.observation_space().shape == (3,)


class Duel:
    num_agents = 2

    def __init__(self, num_envs):
        self.num_envs = num_envs

    def timestep(self, value, action):
        return Timestep(
            obs=jnp.stack([value, -value], axis=-1),
            action=action,
            reward=jnp.stack([action[:, 0] - action[:, 1], action[:, 1] - action[:, 0]], -1),
            terminated=jnp.zeros((self.num_envs, 2), bool),
            truncated=jnp.zeros((self.num_envs, 2), bool),
        )

    def init(self, key):
        value = jnp.arange(self.num_envs, dtype=jnp.float32)
        return (value, jnp.array(0)), self.timestep(value, jnp.zeros((self.num_envs, 2)))

    def step(self, key, state, action):
        value, updates = state
        return (value + 1.0, updates), self.timestep(value + 1.0, action)

    def update(self, state, **kwargs):
        value, updates = state
        return value, updates + 1


def rival(carry, params, joint, key, temperature):
    first, *_ = params
    return carry + 1.0, jnp.broadcast_to(first, joint.obs.shape[:1])


def blank(key, joint):
    return jnp.zeros(1), jnp.zeros(joint.obs.shape[:1])


def duel():
    return Opponent(Duel(4), rival, blank, groups=2)


def test_the_learner_plays_its_seat_against_each_groups_rival():
    environment = duel()
    state, timestep = environment.init(jax.random.key(0))
    assert environment.num_agents == 1 and timestep.reward.shape == (4,)

    state = environment.update(state, opponents=jnp.array([[1.0], [2.0]]))
    state, timestep = environment.step(jax.random.key(1), state, jnp.full(4, 5.0))
    np.testing.assert_allclose(state.joint.action[:, 1], [1, 1, 2, 2])
    np.testing.assert_allclose(timestep.reward, [4, 4, 3, 3])


def test_a_new_rival_starts_from_a_blank_carry_and_other_settings_reach_the_game():
    environment = duel()
    state, _ = environment.init(jax.random.key(0))
    state, _ = play(environment, state, [jnp.zeros(4)] * 3)
    np.testing.assert_allclose(state.carry, 3.0)

    state = environment.update(state, opponents=jnp.array([[1.0], [2.0]]))
    np.testing.assert_allclose(state.carry, 0.0)
    _, updates = environment.update(state, decks=None).env_state
    assert int(updates) == 1


def test_rival_parameters_stay_replicated_inside_a_sharded_state():
    state, _ = duel().init(jax.random.key(0))
    axes = sharded(state, "data")
    assert jax.tree.leaves(axes.params, is_leaf=lambda leaf: leaf is None) == [None]
    assert axes.carry == "data"


def cartpole(num_envs=4):
    import gymnasium as gym
    from gymnasium.vector import AutoresetMode

    return gym.make_vec(
        "CartPole-v1",
        num_envs=num_envs,
        vectorization_mode="sync",
        vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP},
    )


def test_gymnasium_spaces_become_boonta_spaces():
    from gymnasium import spaces

    box = convert(spaces.Box(-1.0, 1.0, (3,), np.float64))
    assert box.shape == (3,) and box.dtype == jnp.float32
    assert convert(spaces.Discrete(5, start=2)).num_actions == 5
    assert convert(spaces.Discrete(5, start=2)).sample(jax.random.key(0)) >= 2
    assert convert(spaces.MultiDiscrete([3, 4])).num_actions == (3, 4)
    with pytest.raises(ValueError, match="Box, Discrete or MultiDiscrete"):
        convert(spaces.Dict({"position": spaces.Box(-1.0, 1.0, (3,))}))


def test_the_gymnasium_adapter_returns_exactly_what_gymnasium_returns():
    adapter = Gymnasium(cartpole)
    key = jax.random.key(0)
    seed = int(jax.random.randint(key, (), 0, jnp.iinfo(jnp.int32).max))
    reference = cartpole()
    expected, _ = reference.reset(seed=seed)

    state, timestep = adapter.init(key)
    np.testing.assert_array_equal(timestep.obs, expected)
    assert bool(timestep.terminated.all()) and not bool(timestep.truncated.any())

    ended = 0
    for step in range(60):
        action = np.full(4, step % 3 == 0, np.int32)
        state, timestep = adapter.step(jax.random.key(step), state, jnp.asarray(action))
        obs, reward, terminated, truncated, _ = reference.step(action)
        np.testing.assert_array_equal(timestep.obs, obs)
        np.testing.assert_array_equal(timestep.reward, reward)
        np.testing.assert_array_equal(timestep.terminated, terminated)
        np.testing.assert_array_equal(timestep.truncated, truncated)
        ended += int(terminated.sum())
    assert ended > 0
    adapter.close(state)
    adapter.shutdown()
    reference.close()


def test_every_init_takes_its_own_pool_and_close_gives_it_back():
    built = []

    def factory():
        built.append(cartpole())
        return built[-1]

    adapter = Gymnasium(factory)
    first, _ = adapter.init(jax.random.key(0))
    second, before = adapter.init(jax.random.key(1))
    assert int(first.handle) != int(second.handle)

    adapter.step(jax.random.key(2), first, jnp.zeros(4, jnp.int32))
    np.testing.assert_array_equal(adapter.busy[int(second.handle)].reset(seed=1)[0].shape, before.obs.shape)
    adapter.close(first)
    adapter.close(second)
    assert not adapter.busy and len(adapter.free) == 2

    for index in range(3):
        state, _ = adapter.init(jax.random.key(index))
        adapter.close(state)
    assert len(built) == 2
    adapter.shutdown()


def test_a_vector_environment_that_resets_on_the_next_step_is_refused():
    import gymnasium as gym

    with pytest.raises(AssertionError, match="SAME_STEP"):
        Gymnasium(lambda: gym.make_vec("CartPole-v1", num_envs=2, vectorization_mode="sync"))


def sokoban(**kwargs):
    pytest.importorskip("jumanji")
    from jumanji.environments.routing.sokoban.generator import SimpleSolveGenerator

    from boonta.environments import jumanji

    return jumanji.make("Sokoban-v0", generator=SimpleSolveGenerator(), **kwargs)


def test_a_jumanji_episode_opens_with_a_start_flag():
    _, timestep = sokoban().init(jax.random.key(0))
    assert bool(timestep.terminated) and not bool(timestep.truncated)


@struct.dataclass
class Grid:
    observation: jax.Array
    reward: jax.Array
    discount: jax.Array
    clock: jax.Array

    def last(self):
        return self.clock >= 2


class MiniGrid:
    def reset(self, params, key):
        return Grid(jnp.zeros((5, 5, 2), jnp.uint8), jnp.float32(0.0), jnp.float32(1.0), jnp.int32(0))

    def step(self, params, grid, action):
        return grid.replace(clock=grid.clock + 1)

    def num_actions(self, params):
        return 6


def test_xland_minigrid_hands_its_wrappers_an_info_dict():
    from boonta.environments.xland_minigrid import XLandMiniGrid

    environment = LogInfo(XLandMiniGrid(MiniGrid(), params=None))
    state, timestep = environment.init(jax.random.key(0))
    assert timestep.info == {}
    _, timestep = environment.step(jax.random.key(1), state, jnp.int32(0))
    assert timestep.info == {}


def statistics_only(environment, num_envs):
    return RecordEpisodeStatistics(environment)


@pytest.mark.parametrize(
    "podracer",
    [
        pytest.param(partial(zoo.online, devices=2), id="anakin"),
        pytest.param(zoo.asynchronous, id="sebulba"),
    ],
)
def test_ppo_learns_cartpole_through_gymnasium_on_every_podracer(podracer):
    environment = make("CartPole-v1", num_envs=16, vectorization_mode="sync")
    podracer = zoo.ppo(
        environment, num_envs=16, num_steps=16, podracer=podracer, wrapper=statistics_only
    )
    state = podracer.init(jax.random.key(0))
    state, _ = podracer.train(state, jax.random.key(1), 300)
    _, logs = podracer.evaluate(state, jax.random.key(2), 500)
    podracer.close(state)
    assert np.nanmean(np.asarray(logs["episode_statistics/episode_return"])) >= 400
