import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial

import jax
import lox
from boonta.algorithms import Algorithm
from boonta.algorithms.wrappers.population import Population
from boonta.algorithms.wrappers.wrapper import Wrapper
from flax import struct
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from boonta.utils import (Key, PyTree, Timestep, Transition, concatenate,
                          sharded, take)
from boonta.utils.typing import Environment, EnvState

from .podracer import Lap, Pit


@struct.dataclass(frozen=True)
class SebulbaConfig:
    num_envs: int
    num_steps: int
    learner: Mesh = struct.field(pytree_node=False)
    actor: Mesh = struct.field(pytree_node=False)


@struct.dataclass(frozen=True)
class ActorState:
    timestep: Timestep
    environment_state: EnvState
    carry: PyTree = None
    lap_state: PyTree = None

    def board(self, algorithm_state, pit_state=None) -> "RolloutState":
        if self.carry is not None:
            algorithm_state = algorithm_state.replace(carry=self.carry)
        return RolloutState(
            timestep=self.timestep,
            environment_state=self.environment_state,
            algorithm_state=algorithm_state,
            pit_state=pit_state,
            lap_state=self.lap_state,
        )

    @classmethod
    def alight(cls, state: "RolloutState") -> "ActorState":
        return cls(
            timestep=state.timestep,
            environment_state=state.environment_state,
            carry=getattr(state.algorithm_state, "carry", None),
            lap_state=state.lap_state,
        )


@struct.dataclass(frozen=True)
class RolloutState:
    timestep: Timestep
    environment_state: EnvState
    algorithm_state: PyTree
    pit_state: PyTree = None
    lap_state: PyTree = None


@struct.dataclass(frozen=True)
class SebulbaState:
    actors: tuple
    algorithm_state: PyTree
    pit_state: PyTree = None


@dataclass
class Sebulba:
    config: SebulbaConfig
    algorithm: Algorithm
    environment: Environment
    pit: Pit | None = None
    lap: Lap | None = None
    shardings: PyTree = None

    def __post_init__(self):
        if self.pit is None:
            self.pit = Pit()
        if self.lap is None:
            self.lap = Lap()

    def rollout(self, state: RolloutState, key, temperature):
        algorithm_key, environment_key, lap_key = jax.random.split(key, 3)

        algorithm_state, action, aux = self.algorithm.step(
            state.algorithm_state, algorithm_key, state.timestep, temperature
        )
        environment_state, timestep = self.environment.step(
            environment_key, state.environment_state, action
        )

        transition = Transition(
            first=state.timestep,
            second=timestep,
            aux=aux,
        )

        algorithm_state = algorithm_state.replace(
            step=algorithm_state.step
            + self.config.num_envs * self.environment.num_agents
        )
        state = self.lap(
            lap_key,
            state.replace(
                algorithm_state=algorithm_state,
                timestep=timestep,
                environment_state=environment_state,
            ),
        )
        return state, transition

    def unroll(self, state: RolloutState, key: Key):
        call_key, rollout_key = jax.random.split(key)
        state = self.pit(call_key, state)
        keys = jax.random.split(rollout_key, self.config.num_steps)
        return jax.lax.scan(partial(self.rollout, temperature=1.0), state, keys)

    def update(self, algorithm_state: PyTree, key: Key, transitions: Transition):
        return self.algorithm.update(algorithm_state, key, transitions)

    def renew(
        self,
        pit_state: PyTree,
        key: Key,
        transitions: Transition,
        algorithm_state: PyTree,
    ):
        return self.pit.update(pit_state, key, transitions, algorithm_state)

    def init(self, key: Key) -> SebulbaState:
        replicated = NamedSharding(self.config.learner, P())
        environment_key, algorithm_key, pit_key, hook_key = jax.random.split(key, 4)
        devices = list(self.config.actor.devices.flat)
        num_devices = len(devices)
        environments = [
            jax.jit(self.environment.init)(
                jax.device_put(jax.random.fold_in(environment_key, index), device)
            )
            for index, device in enumerate(devices)
        ]
        timestep = concatenate(
            [
                jax.device_put(actor_timestep, replicated)
                for _, actor_timestep in environments
            ]
        )
        algorithm_state = jax.jit(self.algorithm.init, out_shardings=self.shardings)(
            jax.device_put(algorithm_key, replicated), timestep
        )
        pit_state = jax.jit(self.pit.init)(
            jax.device_put(pit_key, replicated),
            RolloutState(timestep, None, algorithm_state),
        )
        carry = getattr(algorithm_state, "carry", None)
        start = jax.device_get(algorithm_state.step)

        def begin(key, state):
            lap_key, step_key, call_key = jax.random.split(key, 3)
            state = state.replace(lap_state=self.lap.init(lap_key, state))
            return self.pit(call_key, self.lap(step_key, state))

        actors, num_laps = [], 0
        for index, ((environment_state, actor_timestep), device) in enumerate(
            zip(environments, devices)
        ):
            actor_state = ActorState(
                timestep=actor_timestep,
                environment_state=environment_state,
                carry=take(carry, index, num_devices),
            )
            rollout_state = jax.jit(begin)(
                *jax.device_put(
                    (
                        jax.random.fold_in(hook_key, index),
                        actor_state.board(algorithm_state, pit_state),
                    ),
                    device,
                )
            )
            num_laps += jax.device_get(rollout_state.algorithm_state.step) - start
            actors.append(ActorState.alight(rollout_state))
        algorithm_state = algorithm_state.replace(step=algorithm_state.step + num_laps)
        return SebulbaState(
            actors=tuple(actors), algorithm_state=algorithm_state, pit_state=pit_state
        )

    @property
    def batch_size(self) -> int:
        return (
            self.config.actor.size
            * self.config.num_envs
            * self.config.num_steps
            * self.environment.num_agents
        )

    def train(self, state: SebulbaState, key: Key, num_updates: int) -> PyTree:
        replicated = NamedSharding(self.config.learner, P())
        sharded = NamedSharding(self.config.learner, P(None, "data"))
        devices = list(self.config.actor.devices.flat)
        num_devices = len(devices)
        actor_key, learner_key, pit_key = jax.random.split(key, 3)

        parameter_queues = [queue.Queue() for _ in devices]
        rollout_queues = [queue.Queue() for _ in devices]
        for parameter_queue in parameter_queues:
            parameter_queue.put((state.algorithm_state, state.pit_state))

        def drive(index, device, actor_state, parameter_queue, rollout_queue):
            try:
                for rollout in range(num_updates):
                    if rollout != 1:
                        delivery = parameter_queue.get()
                        if delivery is None:
                            return
                        algorithm_state, pit_state = delivery
                    rollout_state = jax.device_put(
                        actor_state.board(algorithm_state, pit_state), device
                    )
                    before = rollout_state.algorithm_state.step
                    unroll_key = jax.device_put(
                        jax.random.fold_in(
                            jax.random.fold_in(actor_key, index), rollout
                        ),
                        device,
                    )
                    (rollout_state, transitions), log = jax.block_until_ready(
                        self.unroll(rollout_state, unroll_key)
                    )
                    log = jax.device_put(log, replicated)
                    num_laps = rollout_state.algorithm_state.step - before
                    actor_state = ActorState.alight(rollout_state)
                    rollout_queue.put((actor_state, transitions, log, num_laps))
            except BaseException as error:
                rollout_queue.put(error)
                raise

        executor = ThreadPoolExecutor(
            max_workers=num_devices, thread_name_prefix="actor"
        )
        workers = [
            executor.submit(
                drive, index, device, actor_state, parameter_queue, rollout_queue
            )
            for index, (
                device,
                actor_state,
                parameter_queue,
                rollout_queue,
            ) in enumerate(zip(devices, state.actors, parameter_queues, rollout_queues))
        ]

        algorithm_state, pit_state = state.algorithm_state, state.pit_state
        logs = []
        try:
            for update in range(num_updates):
                deliveries = [rollout_queue.get() for rollout_queue in rollout_queues]
                for delivery in deliveries:
                    if isinstance(delivery, BaseException):
                        raise delivery
                actors = [actor_state for actor_state, _, _, _ in deliveries]
                transitions = concatenate(
                    [jax.device_put(pieces, sharded) for _, pieces, _, _ in deliveries],
                    1,
                )
                carries = [
                    jax.device_put(actor_state.carry, replicated)
                    for actor_state in actors
                    if actor_state.carry is not None
                ]
                num_laps = sum(jax.device_get(laps) for _, _, _, laps in deliveries)
                algorithm_state = algorithm_state.replace(
                    step=algorithm_state.step + num_laps
                )
                if carries:
                    algorithm_state = algorithm_state.replace(
                        carry=concatenate(carries)
                    )
                algorithm_state = jax.device_put(algorithm_state, self.shardings)
                update_key = jax.device_put(
                    jax.random.fold_in(learner_key, update), replicated
                )
                algorithm_state, update_log = self.update(
                    algorithm_state, update_key, transitions
                )
                renew_key = jax.device_put(
                    jax.random.fold_in(pit_key, update), replicated
                )
                pit_state, pit_log = self.renew(
                    pit_state, renew_key, transitions, algorithm_state
                )
                for parameter_queue in parameter_queues:
                    parameter_queue.put((algorithm_state, pit_state))
                log = concatenate([log for _, _, log, _ in deliveries])
                logs.append(log | update_log | pit_log)
            for worker in workers:
                worker.result()
        finally:
            for parameter_queue in parameter_queues:
                parameter_queue.put(None)
            executor.shutdown(wait=False)
        return (
            SebulbaState(
                actors=tuple(actors),
                algorithm_state=algorithm_state,
                pit_state=pit_state,
            ),
            concatenate(logs),
        )

    def evaluate(self, state: SebulbaState, key: Key, num_steps: int) -> PyTree:
        device, *_ = self.config.actor.devices.flat
        actor_state, *_ = state.actors
        rollout_state = actor_state.board(state.algorithm_state, state.pit_state)
        rollout_state, key = jax.device_put((rollout_state, key), device)
        return self.assess(rollout_state, key, num_steps)

    def assess(self, state: RolloutState, key: Key, num_steps: int) -> PyTree:
        env_key, rollout_key, lap_key, call_key = jax.random.split(key, 4)
        environment_state, timestep = self.environment.init(env_key)
        state = self.pit(
            call_key,
            self.lap(
                lap_key,
                state.replace(
                    timestep=timestep,
                    environment_state=environment_state,
                ),
            ),
        )
        rollout_keys = jax.random.split(rollout_key, num_steps)
        state, _ = jax.lax.scan(
            partial(self.rollout, temperature=0.0), state, rollout_keys
        )
        self.environment.close(state.environment_state)
        return state

    def close(self, state) -> PyTree:
        for actor_state in getattr(state, "actors", (state,)):
            self.environment.close(actor_state.environment_state)
        return state


def make(
    config: SebulbaConfig,
    algorithm: Algorithm,
    environment: Environment,
    pit: Pit | None = None,
    lap: Lap | None = None,
) -> Sebulba:

    podracer = Sebulba(config, algorithm, environment, pit, lap)

    assert (
        config.num_envs % config.learner.size == 0
    ), "num_envs must be divisible by the number of learner devices"
    assert not (
        isinstance(algorithm, Population)
        or (isinstance(algorithm, Wrapper) and algorithm.wraps(Population))
    ), (
        f"{type(algorithm).__name__} runs a population, which sebulba does not support: "
        f"the actor keeps no recurrent carry for the population's copies, their step "
        f"counts never reach the learner, and the league would credit each rollout to "
        f"the opponent drawn one update after the one that played it. Use anakin."
    )

    replicated = NamedSharding(config.learner, P())

    def initialize(key):
        timesteps = [
            environment.init(jax.random.fold_in(key, index))[1]
            for index in range(config.actor.size)
        ]
        return algorithm.init(key, concatenate(timesteps))

    axes = sharded(jax.eval_shape(initialize, jax.random.key(0)))
    podracer.shardings = jax.tree.map(
        lambda axis: NamedSharding(config.learner, P(axis)) if axis else replicated,
        axes,
    )

    podracer.unroll = jax.jit(
        lox.spool(lox.strip(podracer.unroll, tags=["evaluation"]))
    )
    podracer.update = jax.jit(
        lox.spool(podracer.update), out_shardings=(podracer.shardings, replicated)
    )
    podracer.renew = jax.jit(lox.spool(podracer.renew), out_shardings=replicated)
    podracer.assess = jax.jit(
        lox.spool(lox.strip(podracer.assess, tags=["training"])),
        static_argnums=(2,),
    )
    return podracer
