import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial

import jax
import lox
from boonta.algorithms import Algorithm
from boonta.algorithms.wrappers.ensemble import Ensemble
from boonta.algorithms.wrappers.wrapper import Wrapper
from flax import struct
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from boonta.utils import Key, PyTree, Timestep, Transition, concatenate, take
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

    def board(self, algorithm_state) -> "RolloutState":
        if self.carry is not None:
            algorithm_state = algorithm_state.replace(carry=self.carry)
        return RolloutState(
            timestep=self.timestep,
            environment_state=self.environment_state,
            algorithm_state=algorithm_state,
        )


@struct.dataclass(frozen=True)
class RolloutState:
    timestep: Timestep
    environment_state: EnvState
    algorithm_state: PyTree


@struct.dataclass(frozen=True)
class SebulbaState:
    actors: tuple
    algorithm_state: PyTree


@dataclass
class Sebulba:
    config: SebulbaConfig
    algorithm: Algorithm
    environment: Environment
    pit: Pit[RolloutState] = lambda state: state
    lap: Lap[RolloutState] = lambda state: state

    def rollout(self, state: RolloutState, key, temperature):
        algorithm_key, environment_key = jax.random.split(key)

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
            state.replace(
                algorithm_state=algorithm_state,
                timestep=timestep,
                environment_state=environment_state,
            )
        )
        return state, transition

    def unroll(self, state: RolloutState, key: Key):
        state = self.pit(state)
        keys = jax.random.split(key, self.config.num_steps)
        return jax.lax.scan(partial(self.rollout, temperature=1.0), state, keys)

    def update(self, algorithm_state: PyTree, key: Key, transitions: Transition):
        return self.algorithm.update(algorithm_state, key, transitions)

    def init(self, key: Key) -> SebulbaState:
        replicated = NamedSharding(self.config.learner, P())
        environment_key, algorithm_key = jax.random.split(key)
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
        algorithm_state = jax.jit(self.algorithm.init, out_shardings=replicated)(
            jax.device_put(algorithm_key, replicated), timestep
        )
        carry = getattr(algorithm_state, "carry", None)
        start = jax.device_get(algorithm_state.step)
        actors, num_laps = [], 0
        for index, ((environment_state, actor_timestep), device) in enumerate(
            zip(environments, devices)
        ):
            actor_state = ActorState(
                timestep=actor_timestep,
                environment_state=environment_state,
                carry=take(carry, index, num_devices),
            )
            rollout_state = jax.jit(lambda state: self.pit(self.lap(state)))(
                jax.device_put(actor_state.board(algorithm_state), device)
            )
            num_laps += jax.device_get(rollout_state.algorithm_state.step) - start
            actors.append(
                ActorState(
                    timestep=rollout_state.timestep,
                    environment_state=rollout_state.environment_state,
                    carry=getattr(rollout_state.algorithm_state, "carry", None),
                )
            )
        algorithm_state = algorithm_state.replace(step=algorithm_state.step + num_laps)
        return SebulbaState(actors=tuple(actors), algorithm_state=algorithm_state)

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
        actor_key, learner_key = jax.random.split(key)

        parameter_queues = [queue.Queue() for _ in devices]
        rollout_queues = [queue.Queue() for _ in devices]
        for parameter_queue in parameter_queues:
            parameter_queue.put(state.algorithm_state)

        def drive(index, device, actor_state, parameter_queue, rollout_queue):
            try:
                for rollout in range(num_updates):
                    if rollout != 1:
                        algorithm_state = parameter_queue.get()
                        if algorithm_state is None:
                            return
                    rollout_state = jax.device_put(
                        actor_state.board(algorithm_state), device
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
                    actor_state = ActorState(
                        timestep=rollout_state.timestep,
                        environment_state=rollout_state.environment_state,
                        carry=getattr(rollout_state.algorithm_state, "carry", None),
                    )
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

        algorithm_state = state.algorithm_state
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
                algorithm_state, update_key = jax.device_put(
                    (algorithm_state, jax.random.fold_in(learner_key, update)),
                    replicated,
                )
                algorithm_state, update_log = self.update(
                    algorithm_state, update_key, transitions
                )
                for parameter_queue in parameter_queues:
                    parameter_queue.put(algorithm_state)
                log = concatenate([log for _, _, log, _ in deliveries])
                logs.append(log | update_log)
            for worker in workers:
                worker.result()
        finally:
            for parameter_queue in parameter_queues:
                parameter_queue.put(None)
            executor.shutdown(wait=False)
        return (
            SebulbaState(actors=tuple(actors), algorithm_state=algorithm_state),
            concatenate(logs),
        )

    def evaluate(self, state: SebulbaState, key: Key, num_steps: int) -> PyTree:
        device, *_ = self.config.actor.devices.flat
        actor_state, *_ = state.actors
        rollout_state = actor_state.board(state.algorithm_state)
        rollout_state, key = jax.device_put((rollout_state, key), device)
        return self.assess(rollout_state, key, num_steps)

    def assess(self, state: RolloutState, key: Key, num_steps: int) -> PyTree:
        env_key, rollout_key = jax.random.split(key)
        environment_state, timestep = self.environment.init(env_key)
        state = self.pit(
            self.lap(
                state.replace(
                    timestep=timestep,
                    environment_state=environment_state,
                )
            )
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
    pit: Pit[RolloutState] = lambda state: state,
    lap: Lap[RolloutState] = lambda state: state,
    **kwargs,
) -> Sebulba:

    podracer = Sebulba(config, algorithm, environment, pit, lap)

    assert (
        config.num_envs % config.learner.size == 0
    ), "num_envs must be divisible by the number of learner devices"
    assert not (
        isinstance(algorithm, Ensemble)
        or (isinstance(algorithm, Wrapper) and algorithm.wraps(Ensemble))
    ), (
        f"{type(algorithm).__name__} runs an ensemble, which sebulba does not support: "
        f"the actor keeps no recurrent carry for the ensemble's copies, their step "
        f"counts never reach the learner, and PSRO would credit each rollout to the "
        f"opponent drawn one update after the one that played it. Use anakin."
    )

    replicated = NamedSharding(config.learner, P())

    podracer.unroll = jax.jit(
        lox.spool(lox.strip(podracer.unroll, tags=["evaluation"]))
    )
    podracer.update = jax.jit(lox.spool(podracer.update), out_shardings=replicated)
    podracer.assess = jax.jit(
        lox.spool(lox.strip(podracer.assess, tags=["training"])),
        static_argnums=(2,),
    )
    return podracer
