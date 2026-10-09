import jax

from boonta.algorithms import Algorithm
from boonta.algorithms.wrappers.psro import PSRO, Member
from boonta.environments.wrappers import Opponent, OpponentState
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

    def pit(state):
        opponents = psro.opponents(state.algorithm_state)
        key = jax.random.fold_in(jax.random.key(0), state.algorithm_state.step)

        def seat(node):
            if isinstance(node, OpponentState):
                return environment.update(node, key, opponents=opponents)
            return node

        seated = jax.tree.map(
            seat,
            state.environment_state,
            is_leaf=lambda node: isinstance(node, OpponentState),
        )
        return dress(state.replace(environment_state=seated))

    return psro, environment, pit, lambda state: state
