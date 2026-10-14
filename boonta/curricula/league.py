from dataclasses import dataclass
from typing import Callable

import jax

from boonta.algorithms import Algorithm
from boonta.algorithms.wrappers.psro import PSRO, Member
from boonta.environments.wrappers import Opponent, OpponentState
from boonta.podracers.podracer import Lap, Pit
from boonta.utils.typing import Environment


@dataclass
class Seat(Pit):
    psro: PSRO
    environment: Opponent
    dress: Callable

    def __call__(self, key, state):
        opponents = self.psro.opponents(state.algorithm_state)

        def seat(node):
            if isinstance(node, OpponentState):
                return self.environment.update(node, key, opponents=opponents)
            return node

        seated = jax.tree.map(
            seat,
            state.environment_state,
            is_leaf=lambda node: isinstance(node, OpponentState),
        )
        return self.dress(state.replace(environment_state=seated))


def league(
    algorithm: Algorithm,
    environment: Environment,
    learners,
    play,
    initialize,
    capacity,
    decay,
    load=lambda checkpoint: checkpoint,
    pool=(),
    deal=lambda environment, learners, lineages, **table: lambda state: state,
    **table,
) -> tuple[PSRO, Environment, Pit, Lap]:
    learners = list(learners)
    psro = PSRO(
        algorithm=algorithm,
        learners=learners,
        capacity=capacity,
        decay=decay,
        pool=tuple(
            Member(lineage=spec.lineage, params=load(spec.checkpoint))
            for spec in pool
        ),
        warmstarts={
            index: load(learner.warmstart)
            for index, learner in enumerate(learners)
            if learner.warmstart is not None
        },
    )
    environment = Opponent(environment, play, initialize, groups=len(learners))
    dress = deal(environment, learners, psro.names, **table)
    return psro, environment, Seat(psro, environment, dress), Lap()
