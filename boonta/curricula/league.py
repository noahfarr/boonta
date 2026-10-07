from boonta.algorithms import Algorithm
from boonta.algorithms.wrappers.psro import PSRO, Member
from boonta.environments.wrappers import Opponent
from boonta.podracers.podracer import Lap, Pit
from boonta.utils.typing import Environment


def league(
    algorithm: Algorithm,
    environment: Environment,
    learners,
    play,
    initialize,
    capacity,
    decay,
    load=lambda checkpoint: checkpoint,
    population=(),
    deal=lambda environment, learners, lineages, **table: lambda state: state,
    **table,
) -> tuple[PSRO, Environment, Pit, Lap]:
    learners = list(learners)
    psro = PSRO(
        algorithm=algorithm,
        learners=learners,
        capacity=capacity,
        decay=decay,
        population=tuple(
            Member(lineage=spec.lineage, params=load(spec.checkpoint))
            for spec in population
        ),
        warmstarts={
            index: load(learner.warmstart)
            for index, learner in enumerate(learners)
            if learner.warmstart is not None
        },
    )
    environment = Opponent(environment, play, initialize, groups=len(learners))
    dress = deal(environment, learners, psro.names, **table)

    def pit(state):
        seated = environment.update(
            state.environment_state, opponents=psro.opponents(state.algorithm_state)
        )
        return dress(state.replace(environment_state=seated))

    return psro, environment, pit, lambda state: state
