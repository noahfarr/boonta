import importlib.util
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
from boonta.environments.wrappers import (MCP, PBRS, AutoReset, Batched, ChunkAction, ClipAction,
                                          ClipReward, DomainRandomization,
                                          FlattenObservation, Group,
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
                                          TransformReward, UED, Vectorize,
                                          Wrapper)
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
    pytest.param(lambda: SameStepAutoReset(Dial(), NUM_ENVS), id="same_step_auto_reset"),
    pytest.param(
        lambda: UED(SameStepAutoReset(Dial(), NUM_ENVS)), id="ued"
    ),
    pytest.param(lambda: NextStepAutoReset(Dial(), NUM_ENVS), id="next_step_auto_reset"),
    pytest.param(lambda: OptimisticAutoReset(Dial(), NUM_ENVS, ratio=2), id="optimistic_auto_reset"),
    pytest.param(
        lambda: Group(SameStepAutoReset(Dial(), NUM_ENVS), num_steps=4),
        id="group",
    ),
    pytest.param(lambda: RecordEpisodeStatistics(Vectorize(Dial(), NUM_ENVS)), id="record_episode_statistics"),
    pytest.param(lambda: Vectorize(TimeLimit(Dial(), 100), NUM_ENVS), id="time_limit"),
    pytest.param(lambda: Stagger(SameStepAutoReset(Dial(), NUM_ENVS), spread=4), id="stagger"),
    pytest.param(lambda: Vectorize(StickyAction(Dial()), NUM_ENVS), id="sticky_action"),
    pytest.param(lambda: Vectorize(ChunkAction(Dial(), 3), NUM_ENVS), id="chunk_action"),
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
    assert expected - covered == {Wrapper, AutoReset}


@pytest.mark.parametrize("build", STACKS)
def test_reconfiguration_reaches_the_game_through_every_wrapper(build):
    environment = build()
    state, _ = environment.init(jax.random.key(0))
    state = environment.update(state, jax.random.key(2), setting=7.0)
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
    state = environment.update(state, jax.random.key(2), setting=7.0)
    mask = environment.action_mask(state)
    if jnp.issubdtype(environment.action_space().dtype, jnp.integer):
        assert mask.shape[:-1] == idle(environment).shape
    mask = mask[..., :2]
    shown = [True, True] if environment.wraps(MCP) else [True, False]
    np.testing.assert_array_equal(mask, np.broadcast_to(shown, mask.shape))


@pytest.mark.parametrize("build", STACKS)
def test_every_wrapper_observes_what_it_emits(build):
    environment = build()
    if environment.wraps(MCP):
        pytest.skip("MCP shows the game only on the step a call fires")
    if environment.wraps(Prompt):
        pytest.skip("Prompt cannot tell a prompt chunk from the game in its state")
    state, timestep = environment.init(jax.random.key(0))
    observed = jax.tree.leaves(environment.observe(state))
    emitted = jax.tree.leaves(timestep.obs)
    assert len(observed) == len(emitted)
    for seen, shown in zip(observed, emitted):
        np.testing.assert_array_equal(seen, shown)


def missing(package):
    return importlib.util.find_spec(package) is None


ADAPTERS = [
    pytest.param("gymnax", "Breakout-MinAtar", {}, id="minatar"),
    pytest.param("gymnax", "CartPole-v1", {}, id="classic_control"),
    pytest.param("connectx", "connectx", {"rows": 6, "columns": 7, "inarow": 4}, id="connectx"),
    pytest.param("dune_sea", "forks", {"depth": 6}, id="forks"),
    pytest.param(
        "jaxued", "Maze", {}, id="jaxued_maze",
        marks=pytest.mark.skipif(missing("jaxued"), reason="needs jaxued"),
    ),
    pytest.param(
        "jaxued", "CartPole", {}, id="jaxued_cartpole",
        marks=pytest.mark.skipif(missing("jaxued"), reason="needs jaxued"),
    ),
    pytest.param(
        "kinetix", None,
        {"action_type": "multi_discrete", "observation_type": "symbolic_entity"},
        id="kinetix",
        marks=pytest.mark.skipif(missing("kinetix"), reason="needs kinetix"),
    ),
]


@pytest.mark.parametrize("namespace, env_id, kwargs", ADAPTERS)
def test_a_step_returns_exactly_the_types_init_returns(namespace, env_id, kwargs):
    from boonta import environments

    environment = environments.make(namespace, env_id, kwargs=kwargs)
    first = environment.init(jax.random.key(0))
    state, timestep = first
    second = jax.jit(environment.step)(jax.random.key(1), state, timestep.action)
    assert jax.tree.structure(second) == jax.tree.structure(first)
    for after, before in zip(jax.tree.leaves(second), jax.tree.leaves(first)):
        assert jax.typeof(after) == jax.typeof(before)


def test_log_flags_passes_reconfiguration_and_the_action_mask_through():
    from boonta.environments.peanut_gb import pokemon_red

    environment = pokemon_red.LogFlags(Dial())
    state, _ = environment.init(jax.random.key(0))
    state = environment.update(state, jax.random.key(2), setting=7.0)
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
    environment = SameStepAutoReset(recall(), 1)
    state, timestep = environment.init(jax.random.key(0))
    answer = state.cue
    state, (_, _, last) = play(environment, state, [answer] * 3)

    assert bool(last.terminated[0]) and float(last.reward[0]) == 1.0
    assert int(state.clock[0]) == 0
    np.testing.assert_array_equal(last.obs, environment.observe(state))


def test_next_step_auto_reset_spends_one_empty_step_on_the_reset():
    environment = NextStepAutoReset(recall(), 1)
    state, _ = environment.init(jax.random.key(0))
    answer = state.env_state.cue
    state, (*_, last, reset) = play(environment, state, [answer] * 4)

    assert bool(last.terminated[0]) and float(last.reward[0]) == 1.0
    assert not bool(reset.terminated[0]) and float(reset.reward[0]) == 0.0
    assert int(state.env_state.clock[0]) == 0


def test_next_step_auto_reset_restarts_a_team_once_every_agent_is_done():
    environment = NextStepAutoReset(TimeLimit(Team(Dial(), 2), 2), 1)
    state, _ = environment.init(jax.random.key(0))
    state, (_, last, reset) = play(environment, state, [jnp.zeros((1, 2), jnp.int32)] * 3)

    assert bool(last.truncated.all())
    assert not bool(reset.done.any())
    np.testing.assert_array_equal(state.env_state.env_state.clock, 0)


def test_optimistic_auto_reset_restarts_every_finished_environment():
    environment = OptimisticAutoReset(recall(), NUM_ENVS, ratio=2)
    state, _ = environment.init(jax.random.key(0))
    state, timesteps = play(environment, state, [jnp.zeros(NUM_ENVS, jnp.int32)] * 3)
    assert bool(timesteps[-1].terminated.all())
    np.testing.assert_array_equal(state.clock, 0)


def test_a_group_restarts_each_group_from_one_start():
    environment = Group(SameStepAutoReset(Dial(), NUM_ENVS), num_steps=4, group_size=2)
    state, _ = environment.init(jax.random.key(0))
    state, timesteps = play(environment, state, [jnp.zeros(NUM_ENVS, jnp.int32)] * 4)
    noise = np.asarray(state.env_state.noise).reshape(-1, 2)

    assert bool(timesteps[-1].truncated.all()) and not bool(timesteps[-1].terminated.any())
    np.testing.assert_array_equal(state.env_state.clock, 0)
    np.testing.assert_array_equal(noise[:, 0], noise[:, 1])
    assert len(np.unique(noise[:, 0])) > 1


def test_a_batched_environment_restarts_its_groups_from_one_start():
    environment = Group(
        Stagger(Batched(Vectorize(Dial(), NUM_ENVS), NUM_ENVS), spread=100),
        num_steps=4,
        group_size=2,
    )
    state, _ = environment.init(jax.random.key(0))
    noise = np.asarray(state.env_state.env_state.noise).reshape(-1, 2)
    np.testing.assert_array_equal(noise[:, 0], noise[:, 1])
    assert len(np.unique(noise[:, 0])) > 1


def test_stagger_cuts_each_environment_of_a_batch_once():
    environment = Stagger(Batched(Vectorize(Dial(), NUM_ENVS), NUM_ENVS), spread=4)
    state, _ = environment.init(jax.random.key(0))
    budget = np.asarray(state.budget)
    state, timesteps = play(environment, state, [jnp.zeros(NUM_ENVS, jnp.int32)] * 6)
    cuts = np.stack([np.asarray(timestep.truncated) for timestep in timesteps])
    np.testing.assert_array_equal(cuts.sum(axis=0), 1)
    np.testing.assert_array_equal(cuts.argmax(axis=0), budget)
    for index, timestep in enumerate(timesteps):
        clocks = np.asarray(timestep.obs[:, 0])
        np.testing.assert_array_equal(clocks[cuts[index]], 0)


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
    environment = RecordEpisodeStatistics(SameStepAutoReset(Corridor(), NUM_ENVS), gamma=gamma)
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
        Group(SameStepAutoReset(Corridor(), NUM_ENVS), num_steps=4)
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
    environment = Stagger(SameStepAutoReset(Dial(), 64), spread=spread)
    state, _ = environment.init(jax.random.key(0))
    budgets = np.asarray(state.budget)
    assert budgets.min() >= 0 and budgets.max() < spread and len(np.unique(budgets)) > 1

    state, timesteps = play(environment, state, [jnp.zeros(64, jnp.int32)] * 3 * spread, jax.jit(environment.step))
    cuts = np.stack([np.asarray(t.truncated) for t in timesteps]).sum(0)
    np.testing.assert_array_equal(cuts, 1)
    assert len(np.unique(np.asarray(state.env_state.clock))) > 1


def test_stagger_never_marks_a_natural_end_or_logs_its_own_cut():
    environment = Stagger(RecordEpisodeStatistics(SameStepAutoReset(recall(), 64)), spread=6)
    _, logs = statistics(environment, [jnp.zeros(64, jnp.int32)] * 12)
    np.testing.assert_array_equal(logs["episode_statistics/episode_length"], 3)


def test_normalized_observations_have_zero_mean_and_unit_variance():
    environment = NormalizeObservation(Vectorize(Dial(), 256))
    state, _ = environment.init(jax.random.key(0))
    state, timesteps = play(environment, state, [jnp.zeros(256, jnp.int32)] * 20)
    noise = np.asarray(timesteps[-1].obs[:, 2])
    assert abs(noise.mean()) < 0.1 and abs(noise.std() - 1.0) < 0.1


def test_normalized_rewards_divide_by_the_spread_of_returns():
    environment = NormalizeReward(SameStepAutoReset(reach(), 256), gamma=0.0)
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


def test_a_chunk_of_actions_plays_in_order_and_stops_at_the_episode_end():
    environment = ChunkAction(Corridor(length=4), size=3)
    assert environment.action_space().shape == (3,)
    state, _ = environment.init(jax.random.key(0))
    state, timestep = environment.step(jax.random.key(1), state, jnp.array([0, 1, 1]))
    assert int(state.position) == 1
    assert int(state.clock) == 3
    assert float(timestep.reward) == 0.0
    assert not bool(timestep.terminated)
    state, timestep = environment.step(jax.random.key(2), state, jnp.array([1, 0, 1]))
    assert int(state.position) == 3
    assert int(state.clock) == 5
    assert float(timestep.reward) == 1.0
    assert bool(timestep.terminated)
    np.testing.assert_array_equal(timestep.action, [1, 0, 1])


def test_time_aware_observations_count_up_and_restart_with_the_episode():
    environment = SameStepAutoReset(TimeAwareObservation(recall(), time_limit=4), 1)
    state, _ = environment.init(jax.random.key(0))
    _, timesteps = play(environment, state, [jnp.zeros(1, jnp.int32)] * 4)
    np.testing.assert_allclose([float(t.obs[0, -1]) for t in timesteps], [-0.25, 0.0, -0.5, -0.25])


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
    assert RecordEpisodeStatistics(SameStepAutoReset(cartpole, 2)).time_limit() == 7
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

    def update(self, state, key, **kwargs):
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

    state = environment.update(state, jax.random.key(2), opponents=jnp.array([[1.0], [2.0]]))
    state, timestep = environment.step(jax.random.key(1), state, jnp.full(4, 5.0))
    np.testing.assert_allclose(state.joint.action[:, 1], [1, 1, 2, 2])
    np.testing.assert_allclose(timestep.reward, [4, 4, 3, 3])


def test_a_new_rival_starts_from_a_blank_carry_and_other_settings_reach_the_game():
    environment = duel()
    state, _ = environment.init(jax.random.key(0))
    state, _ = play(environment, state, [jnp.zeros(4)] * 3)
    np.testing.assert_allclose(state.carry, 3.0)

    state = environment.update(state, jax.random.key(2), opponents=jnp.array([[1.0], [2.0]]))
    np.testing.assert_allclose(state.carry, 0.0)
    _, updates = environment.update(state, jax.random.key(3), decks=None).env_state
    assert int(updates) == 1


class Flicker(Dial):
    def step(self, key, state, action):
        state, timestep = super().step(key, state, action)
        return state, timestep.replace(terminated=jnp.bool_(True))


THETA = jnp.array([10.0, 20.0, 30.0])


def ued(game):
    environment = UED(SameStepAutoReset(game, NUM_ENVS))
    state, timestep = environment.init(jax.random.key(0))
    state = environment.update(state, jax.random.key(1), theta=THETA, weights=jnp.zeros(3))
    return environment, state, timestep


def games(state):
    return state.env_state.params


def test_without_an_assignment_the_games_own_resets_run():
    environment, state, timestep = ued(Flicker())
    np.testing.assert_array_equal(timestep.info["theta"], -1)
    state, (first, second) = play(environment, state, [idle(environment)] * 2)
    np.testing.assert_array_equal(second.info["theta"], -1)
    np.testing.assert_array_equal(games(state), 0.0)


def test_an_assignment_cuts_every_episode_on_the_next_step_and_starts_each_theta():
    environment, state, _ = ued(Dial())
    state, _ = play(environment, state, [idle(environment)] * 2)
    assignment = jnp.arange(NUM_ENVS) % 3
    state = environment.update(state, jax.random.key(2), assign=assignment)
    state, (cut, after) = play(environment, state, [idle(environment)] * 2)
    np.testing.assert_array_equal(cut.truncated, True)
    np.testing.assert_array_equal(cut.info["theta"], -1)
    np.testing.assert_array_equal(cut.obs[:, 0], 0.0)
    np.testing.assert_array_equal(after.truncated, False)
    np.testing.assert_array_equal(after.info["theta"], assignment)
    np.testing.assert_array_equal(games(state), np.asarray(THETA)[assignment])


def test_an_episode_that_ends_replays_its_theta():
    environment, state, _ = ued(Flicker())
    assignment = jnp.arange(NUM_ENVS) % 3
    state = environment.update(state, jax.random.key(2), assign=assignment)
    state, timesteps = play(environment, state, [idle(environment)] * 4)
    for timestep in timesteps[1:]:
        np.testing.assert_array_equal(timestep.info["theta"], assignment)
        np.testing.assert_array_equal(timestep.obs[:, 0], 0.0)
    np.testing.assert_array_equal(games(state), np.asarray(THETA)[assignment])


def test_only_the_environments_that_finished_are_reset():
    environment, state, _ = ued(Dial())
    assignment = jnp.arange(NUM_ENVS) % 3
    state = environment.update(state, jax.random.key(2), assign=assignment)
    state, (_, second) = play(environment, state, [idle(environment)] * 2)
    np.testing.assert_array_equal(second.obs[:, 0], 1.0)
    np.testing.assert_array_equal(games(state), np.asarray(THETA)[assignment])


def test_a_restart_draws_each_theta_from_the_weights():
    environment, state, _ = ued(Dial())
    state = environment.update(
        state, jax.random.key(2), weights=jnp.array([1.0, 0.0, 3.0]), restart=True
    )
    state, (cut, after) = play(environment, state, [idle(environment)] * 2)
    drawn = np.asarray(after.info["theta"])
    assert set(np.unique(drawn)) <= {0, 2}
    np.testing.assert_array_equal(games(state), np.asarray(THETA)[drawn])


def test_a_restart_without_weights_leaves_every_game_alone():
    environment, state, _ = ued(Dial())
    state = environment.update(state, jax.random.key(2), restart=True)
    state, (cut, after) = play(environment, state, [idle(environment)] * 2)
    np.testing.assert_array_equal(cut.truncated, False)
    np.testing.assert_array_equal(after.info["theta"], -1)
    np.testing.assert_array_equal(after.obs[:, 0], 2.0)


def test_ued_runs_without_a_set_of_theta():
    environment = UED(SameStepAutoReset(Dial(), NUM_ENVS))
    state, _ = environment.init(jax.random.key(0))
    state, (first,) = play(environment, state, [idle(environment)])
    np.testing.assert_array_equal(first.info["theta"], -1)


def test_the_set_of_theta_stays_replicated_inside_a_sharded_state():
    environment, state, _ = ued(Dial())
    axes = sharded(state, "data")
    assert jax.tree.leaves(axes.theta, is_leaf=lambda leaf: leaf is None) == [None]
    assert axes.weights is None
    assert axes.playing == "data"


JAXUED = pytest.mark.skipif(
    importlib.util.find_spec("jaxued") is None, reason="needs jaxued"
)

CORRIDOR = """
.....
#####
.>.G.
#####
.....
"""


def maze(**kwargs):
    from boonta import environments

    return environments.make("jaxued", "Maze", kwargs=kwargs)


def corridor():
    from jaxued.environments.maze import Level

    return Level.from_str(CORRIDOR).pad_to_shape(13, 13)


def same_leaves(left, right):
    left, right = jax.tree.leaves(left), jax.tree.leaves(right)
    assert len(left) == len(right)
    for seen, shown in zip(left, right):
        np.testing.assert_array_equal(seen, shown)


def control(env_id):
    from boonta import environments

    return lambda: environments.make("jaxued", env_id)


GAMES = [
    pytest.param(maze, id="maze"),
    pytest.param(control("CartPole"), id="cartpole"),
    pytest.param(control("Acrobot"), id="acrobot"),
    pytest.param(control("Pendulum"), id="pendulum"),
]

CONTROLS = GAMES[1:]


def recipe_stack(game):
    return lambda: RecordEpisodeStatistics(UED(SameStepAutoReset(game(), NUM_ENVS)))


JAXUED_STACKS = [
    *GAMES,
    *[pytest.param(recipe_stack(*param.values), id=f"{param.id}_recipe_stack") for param in GAMES],
]


@JAXUED
@pytest.mark.parametrize("build", JAXUED_STACKS)
def test_jaxued_games_observe_what_they_emit_through_every_wrapper(build):
    environment = build()
    state, timestep = environment.init(jax.random.key(0))
    same_leaves(environment.observe(state), timestep.obs)
    state, timestep = environment.step(jax.random.key(1), state, jnp.zeros_like(timestep.action))
    same_leaves(environment.observe(state), timestep.obs)


@JAXUED
@pytest.mark.parametrize("build", JAXUED_STACKS)
def test_reconfiguration_reaches_jaxued_games_through_every_wrapper(build):
    environment = build()
    state, _ = environment.init(jax.random.key(0))
    updated = environment.update(state, jax.random.key(1), setting=7.0)
    same_leaves(updated, state)


@JAXUED
@pytest.mark.parametrize("build", GAMES)
def test_jaxued_games_step_into_the_state_they_were_given(build):
    environment = build()
    state, timestep = environment.init(jax.random.key(0))
    stepped, _ = environment.step(jax.random.key(1), state, timestep.action)
    assert jax.tree.structure(stepped) == jax.tree.structure(state)
    for after, before in zip(jax.tree.leaves(stepped), jax.tree.leaves(state)):
        assert after.dtype == before.dtype and after.shape == before.shape


@JAXUED
def test_an_update_with_a_level_restarts_the_maze_at_that_level():
    environment = maze()
    state, _ = environment.init(jax.random.key(0))
    state, _ = environment.step(jax.random.key(1), state, jnp.int32(2))
    level = corridor()
    assert environment.update(state, jax.random.key(2), setting=7.0) is state

    started = environment.update(state, jax.random.key(3), theta=level)
    np.testing.assert_array_equal(started.wall_map, level.wall_map)
    np.testing.assert_array_equal(started.agent_pos, level.agent_pos)
    np.testing.assert_array_equal(started.agent_dir, level.agent_dir)
    np.testing.assert_array_equal(started.goal_pos, level.goal_pos)
    assert int(started.time) == 0 and not bool(started.terminal)
    obs, _ = environment._env.reset_to_level(jax.random.key(2), level, environment._params)
    same_leaves(environment.observe(started), {"image": obs.image, "agent_dir": obs.agent_dir})


@JAXUED
def test_a_level_reaches_the_maze_through_the_recipe_stack():
    from boonta.environments.jaxued import maze_generator

    environment = UED(SameStepAutoReset(maze(), NUM_ENVS))
    state, _ = environment.init(jax.random.key(0))
    levels = jax.vmap(maze_generator())(jax.random.split(jax.random.key(1), 3))
    state = environment.update(state, jax.random.key(2), theta=levels, weights=jnp.zeros(3))
    assignment = jnp.arange(NUM_ENVS) % 3
    state = environment.update(state, jax.random.key(2), assign=assignment)
    turn = jnp.zeros(NUM_ENVS, jnp.int32)
    state, (cut, after) = play(environment, state, [turn, turn])
    np.testing.assert_array_equal(cut.truncated, True)
    np.testing.assert_array_equal(after.info["theta"], assignment)
    games = state.env_state
    np.testing.assert_array_equal(games.wall_map, np.asarray(levels.wall_map)[assignment])
    np.testing.assert_array_equal(games.goal_pos, np.asarray(levels.goal_pos)[assignment])
    np.testing.assert_array_equal(games.time, 1)


@JAXUED
def test_the_maze_terminates_at_the_goal_and_truncates_at_its_time_limit():
    environment = maze(max_steps_in_episode=3)
    level = corridor()
    state = environment.update(
        environment.init(jax.random.key(0))[0], jax.random.key(1), theta=level
    )

    _, (first, second) = play(environment, state, [jnp.int32(2), jnp.int32(2)])
    assert not bool(first.done)
    assert bool(second.terminated) and not bool(second.truncated)
    assert float(second.reward) > 0

    _, timesteps = play(environment, state, [jnp.int32(0)] * 3)
    assert [bool(timestep.done) for timestep in timesteps] == [False, False, True]
    assert bool(timesteps[-1].truncated) and not bool(timesteps[-1].terminated)

    environment = maze(max_steps_in_episode=2)
    _, (_, last) = play(environment, state, [jnp.int32(2), jnp.int32(2)])
    assert bool(last.terminated) and not bool(last.truncated)


@JAXUED
def test_the_maze_generator_makes_levels_the_maze_plays():
    from boonta.environments.jaxued import maze_generator

    level = maze_generator()(jax.random.key(0))
    assert bool(level.is_well_formatted())
    environment = maze()
    state = environment.update(
        environment.init(jax.random.key(2))[0], jax.random.key(3), theta=level
    )
    np.testing.assert_array_equal(state.wall_map, level.wall_map)


@JAXUED
@pytest.mark.parametrize("build", CONTROLS)
def test_a_control_level_starts_from_a_state_its_key_draws(build):
    environment = build()
    state, _ = environment.init(jax.random.key(0))
    level = environment._sample(jax.random.key(1))
    first = environment.update(state, jax.random.key(2), theta=level)
    again = environment.update(state, jax.random.key(2), theta=level)
    other = environment.update(state, jax.random.key(3), theta=level)
    same_leaves(first, again)
    same_leaves(first.level_params, jax.tree.map(lambda leaf: np.float32(leaf), level))
    assert int(first.time) == 0
    assert not np.array_equal(environment.observe(first), environment.observe(other))


@JAXUED
def test_the_maze_starts_a_level_the_same_way_whatever_the_key():
    environment = maze()
    state, _ = environment.init(jax.random.key(0))
    same_leaves(
        environment.update(state, jax.random.key(1), theta=corridor()),
        environment.update(state, jax.random.key(2), theta=corridor()),
    )


@JAXUED
@pytest.mark.parametrize("build", CONTROLS)
def test_a_control_level_reaches_the_game_through_the_recipe_stack(build):
    game = build()
    environment = UED(SameStepAutoReset(game, NUM_ENVS))
    state, _ = environment.init(jax.random.key(0))
    levels = jax.vmap(game._sample)(jax.random.split(jax.random.key(1), 3))
    state = environment.update(state, jax.random.key(2), theta=levels, weights=jnp.zeros(3))
    assignment = jnp.arange(NUM_ENVS) % 3
    state = environment.update(state, jax.random.key(3), assign=assignment)
    idle = jnp.zeros((NUM_ENVS, *game.action_space().shape), game.action_space().dtype)
    state, (cut, after) = play(environment, state, [idle, idle])
    np.testing.assert_array_equal(cut.truncated, True)
    np.testing.assert_array_equal(after.info["theta"], assignment)
    same_leaves(
        state.env_state.level_params,
        jax.tree.map(lambda leaf: np.asarray(leaf)[assignment], levels),
    )
    starts = np.asarray(jax.tree.leaves(state.env_state)[0])
    assert len(np.unique(starts)) == NUM_ENVS


@JAXUED
def test_a_pole_terminates_when_it_falls_and_a_pendulum_truncates_at_its_time_limit():
    pole = control("CartPole")()
    state, _ = pole.init(jax.random.key(0))
    _, timesteps = play(pole, state, [jnp.int32(0)] * 60)
    done = [bool(timestep.done) for timestep in timesteps]
    ended = done.index(True)
    assert bool(timesteps[ended].terminated) and not bool(timesteps[ended].truncated)

    pendulum = control("Pendulum")()
    state, _ = pendulum.init(jax.random.key(0))
    _, timesteps = play(pendulum, state, [jnp.zeros(1)] * pendulum.time_limit())
    assert not any(bool(timestep.done) for timestep in timesteps[:-1])
    assert bool(timesteps[-1].truncated) and not bool(timesteps[-1].terminated)


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
