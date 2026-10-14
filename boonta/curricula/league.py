from dataclasses import replace

from boonta.algorithms import Algorithm
from boonta.algorithms.wrappers.roster import Roster
from boonta.environments.wrappers import Opponent
from boonta.podracers.podracer import Lap
from boonta.utils.typing import Environment

from .psro import PSRO, Member


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
) -> tuple[Roster, Environment, PSRO, Lap | None]:
    learners = list(learners)
    roster = Roster(
        algorithm,
        count=len(learners),
        warmstarts={
            index: load(learner.warmstart)
            for index, learner in enumerate(learners)
            if learner.warmstart is not None
        },
        kl_coefficients=tuple(learner.kl_coefficient for learner in learners),
    )
    environment = Opponent(environment, play, initialize, groups=len(learners))
    psro = PSRO(
        roster=roster,
        opponent=environment,
        learners=learners,
        capacity=capacity,
        decay=decay,
        pool=tuple(
            Member(lineage=spec.lineage, params=load(spec.checkpoint))
            for spec in pool
        ),
    )
    psro = replace(psro, dress=deal(environment, learners, psro.names, **table))
    return roster, environment, psro, None
