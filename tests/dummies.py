from types import SimpleNamespace

import jax
import jax.numpy as jnp
import lox
import numpy as np
from flax import struct

from boonta.environments.environment import Environment
from boonta.environments.spaces import Space
from boonta.utils import Timestep, Transition


@struct.dataclass(frozen=True)
class CueState:
    cue: jax.Array
    clock: jax.Array


class Cue(Environment):
    def __init__(self, space: Space, delay: int = 0, solved: float = 0.9):
        self.space = space
        self.delay = delay
        self.solved = solved
        self.shortest = delay + 1
        self.discrete = jnp.issubdtype(space.dtype, jnp.integer)
        if self.discrete:
            self.width = space.num_actions
        else:
            self.width, *_ = space.shape

    def observation_space(self) -> Space:
        return Space((self.width + 1,), jnp.float32, -1.0, 1.0)

    def action_space(self) -> Space:
        return self.space

    def show(self, cue: jax.Array) -> jax.Array:
        if self.discrete:
            return jax.nn.one_hot(cue, self.width)
        return cue

    def observe(self, state: CueState) -> jax.Array:
        cue = jnp.where(state.clock == 0, self.show(state.cue), 0.0)
        asked = (state.clock == self.delay).astype(jnp.float32)
        return jnp.concatenate([cue, asked[None]])

    def score(self, cue: jax.Array, action: jax.Array) -> jax.Array:
        if self.discrete:
            return (action == cue).astype(jnp.float32)
        action = jnp.clip(action, self.space.low, self.space.high)
        return 1.0 - jnp.mean(jnp.abs(action - cue))

    def init(self, key):
        state = CueState(cue=self.space.sample(key), clock=jnp.int32(0))
        return state, Timestep(
            obs=self.observe(state),
            action=jnp.zeros(self.space.shape, self.space.dtype),
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(True),
            truncated=jnp.bool_(False),
        )

    def step(self, key, state, action):
        asked = state.clock == self.delay
        reward = jnp.where(asked, self.score(state.cue, action), 0.0)
        state = state.replace(clock=state.clock + 1)
        return state, Timestep(
            obs=self.observe(state),
            action=action,
            reward=reward,
            terminated=asked,
            truncated=jnp.bool_(False),
        )

    def expert(self, state: CueState) -> jax.Array:
        return state.cue


@struct.dataclass(frozen=True)
class CorridorState:
    position: jax.Array
    clock: jax.Array


class Corridor(Environment):
    def __init__(self, length: int = 4, horizon: int = 12, solved: float = 0.9):
        self.length = length
        self.horizon = horizon
        self.solved = solved
        self.shortest = length - 1

    def observation_space(self) -> Space:
        return Space((self.length,), jnp.float32, 0.0, 1.0)

    def action_space(self) -> Space:
        return Space((), jnp.int32, 0, 1)

    def observe(self, state: CorridorState) -> jax.Array:
        return jax.nn.one_hot(state.position, self.length)

    def init(self, key):
        state = CorridorState(position=jnp.int32(0), clock=jnp.int32(0))
        return state, Timestep(
            obs=self.observe(state),
            action=jnp.int32(0),
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(True),
            truncated=jnp.bool_(False),
        )

    def step(self, key, state, action):
        forward = action == self.expert(state)
        position = jnp.clip(
            state.position + jnp.where(forward, 1, -1), 0, self.length - 1
        )
        clock = state.clock + 1
        arrived = position == self.length - 1
        state = CorridorState(position=position, clock=clock)
        return state, Timestep(
            obs=self.observe(state),
            action=action,
            reward=arrived.astype(jnp.float32),
            terminated=arrived,
            truncated=~arrived & (clock >= self.horizon),
        )

    def expert(self, state: CorridorState) -> jax.Array:
        return state.position % 2


class Team(Environment):
    def __init__(self, environment: Environment, num_agents: int = 3):
        self.environment = environment
        self.num_agents = num_agents
        self.solved = environment.solved

    def observation_space(self) -> Space:
        return self.environment.observation_space()

    def action_space(self) -> Space:
        return self.environment.action_space()

    def init(self, key):
        return jax.vmap(self.environment.init)(jax.random.split(key, self.num_agents))

    def step(self, key, state, action):
        keys = jax.random.split(key, self.num_agents)
        return jax.vmap(self.environment.step)(keys, state, action)

    def update(self, state, key, **kwargs):
        return jax.vmap(lambda state: self.environment.update(state, key, **kwargs))(state)

    def action_mask(self, state):
        return jax.vmap(self.environment.action_mask)(state)

    def observe(self, state):
        return jax.vmap(self.environment.observe)(state)


@struct.dataclass(frozen=True)
class DialState:
    clock: jax.Array
    setting: jax.Array
    noise: jax.Array
    params: jax.Array


class Dial(Environment):
    shortest = 4
    solved = 0.0

    def observation_space(self) -> Space:
        return Space((3,), jnp.float32, -jnp.inf, jnp.inf)

    def action_space(self) -> Space:
        return Space((), jnp.int32, 0, 1)

    def observe(self, state: DialState) -> jax.Array:
        return jnp.stack([state.clock, state.setting, state.noise]).astype(jnp.float32)

    def timestep(self, state: DialState, action: jax.Array) -> Timestep:
        return Timestep(
            obs=self.observe(state),
            action=action,
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(False),
            truncated=jnp.bool_(False),
            info={"clock": state.clock, "warm": jnp.bool_(False)},
        )

    def init(self, key):
        state = DialState(
            clock=jnp.float32(0.0),
            setting=jnp.float32(0.0),
            noise=jax.random.normal(key),
            params=jnp.float32(0.0),
        )
        return state, self.timestep(state, jnp.int32(0))

    def step(self, key, state, action):
        state = state.replace(clock=state.clock + 1.0)
        return state, self.timestep(state, action)

    def action_mask(self, state: DialState) -> jax.Array:
        return jnp.stack([state.setting > 0, state.setting <= 0], axis=-1)

    def update(
        self,
        state: DialState,
        key: jax.Array,
        setting: jax.Array = None,
        theta: jax.Array = None,
    ) -> DialState:
        if setting is not None:
            state = state.replace(setting=jnp.full_like(state.setting, setting))
        if theta is not None:
            state = state.replace(clock=jnp.zeros_like(state.clock), params=theta)
        return state

    def expert(self, state: DialState) -> jax.Array:
        return jnp.int32(0)


@struct.dataclass(frozen=True)
class ProbeState:
    step: jax.Array
    version: jax.Array
    carry: jax.Array = struct.field(metadata={"axis": "data"})


class Probe:
    def init(self, key, timestep) -> ProbeState:
        num_envs, *_ = timestep.obs.shape
        return ProbeState(
            step=jnp.array(0), version=jnp.float32(0.0), carry=jnp.zeros((num_envs,))
        )

    def step(self, state, key, timestep, temperature=1.0):
        num_envs, *_ = timestep.obs.shape
        lox.log({"probe/temperature": temperature})
        return (
            state.replace(carry=state.carry + 1.0),
            jnp.zeros((num_envs,), jnp.int32),
            {"version": jnp.full((num_envs,), state.version)},
        )

    def update(self, state, key, transitions) -> ProbeState:
        logs = {
            "probe/version": state.version,
            "probe/observed": jnp.mean(transitions.first.obs),
        }
        if transitions.aux:
            _, width, *_ = transitions.second.obs.shape
            logs |= {
                "probe/acted": jnp.mean(transitions.aux["version"]),
                "probe/setting": jnp.mean(transitions.second.obs[..., 1]),
                "probe/width": jnp.float32(width),
            }
        lox.log(logs)
        return state.replace(version=state.version + 1.0)


def demonstrations(environment, key: jax.Array, num_episodes: int) -> Transition:
    def episode(key):
        init_key, step_key = jax.random.split(key)
        state, timestep = environment.init(init_key)

        def act(carry, key):
            state, timestep = carry
            state, next_timestep = environment.step(key, state, environment.expert(state))
            return (state, next_timestep), Transition(first=timestep, second=next_timestep)

        keys = jax.random.split(step_key, environment.shortest)
        _, transitions = jax.lax.scan(act, (state, timestep), keys)
        return transitions

    return jax.vmap(episode)(jax.random.split(key, num_episodes))


def flatten(transitions: Transition) -> Transition:
    return jax.tree.map(lambda leaf: leaf.reshape(-1, *leaf.shape[2:]), transitions)


def recordings(transitions: Transition) -> list[SimpleNamespace]:
    def record(episode):
        return SimpleNamespace(
            observations=np.concatenate(
                [np.asarray(episode.first.obs), np.asarray(episode.second.obs)[-1:]]
            ),
            actions=np.asarray(episode.second.action),
            rewards=np.asarray(episode.second.reward),
            terminations=np.asarray(episode.second.terminated),
            truncations=np.asarray(episode.second.truncated),
        )

    count, *_ = transitions.second.reward.shape
    return [
        record(jax.tree.map(lambda leaf: leaf[index], transitions))
        for index in range(count)
    ]


def space(leaf):
    import gymnasium as gym

    if isinstance(leaf, dict):
        return gym.spaces.Dict({name: space(value) for name, value in leaf.items()})
    leaf = np.asarray(leaf)
    bound = np.iinfo(leaf.dtype).max if np.issubdtype(leaf.dtype, np.integer) else np.inf
    return gym.spaces.Box(-bound, bound, leaf.shape[1:], leaf.dtype)


def publish(episodes: list, dataset_id: str = "dummy/dial/expert-v0") -> str:
    import warnings

    import minari
    from minari.data_collector import EpisodeBuffer

    first, *_ = episodes
    buffers = [
        EpisodeBuffer(
            id=index,
            observations=episode.observations,
            actions=episode.actions,
            rewards=episode.rewards,
            terminations=episode.terminations,
            truncations=episode.truncations,
        )
        for index, episode in enumerate(episodes)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        minari.create_dataset_from_buffers(
            dataset_id,
            buffers,
            observation_space=space(first.observations),
            action_space=space(first.actions),
        )
    return dataset_id


@struct.dataclass(frozen=True)
class EpisodesState:
    transitions: Transition = struct.field(metadata={"axis": "data"})


class Episodes:
    def __init__(self, transitions: Transition):
        self.transitions = transitions

    def init(self) -> EpisodesState:
        return EpisodesState(self.transitions)

    def update(self, state, key, sharding) -> EpisodesState:
        return state

    def sample(self, state, key, batch_shape) -> Transition:
        batch_size, *_ = batch_shape
        count, *_ = state.transitions.second.reward.shape
        index = jax.random.randint(key, (batch_size,), 0, count)
        return jax.tree.map(lambda leaf: leaf[index], state.transitions)

    def close(self) -> None:
        pass


def match() -> Cue:
    return Cue(Space((), jnp.int32, 0, 2))


def reach() -> Cue:
    return Cue(Space((2,), jnp.float32, -1.0, 1.0))


def recall() -> Cue:
    return Cue(Space((), jnp.int32, 0, 2), delay=2)


def recall_continuous() -> Cue:
    return Cue(Space((2,), jnp.float32, -1.0, 1.0), delay=2)


def corridor() -> Corridor:
    return Corridor()
