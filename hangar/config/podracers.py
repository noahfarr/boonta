import operator
import statistics

from boonta import curricula
from boonta.podracers import anakin, quadinaros, sebulba
from boonta.utils import mesh

from .store import builds, store

podracer = store(group="podracer")
scoring = store(group="scoring")

podracer(
    builds(
        anakin.make,
        zen_partial=True,
        config=builds(
            anakin.AnakinConfig,
            num_envs="${environment.num_envs}",
            num_steps="${rollout.num_steps}",
            mesh=builds(mesh, count=1),
        ),
    ),
    name="anakin",
)
podracer(
    builds(
        quadinaros.make,
        zen_partial=True,
        config=builds(
            quadinaros.QuadinarosConfig,
            num_envs="${environment.num_envs}",
            batch_shape=["${algorithm.batch_size}"],
            mesh=builds(mesh, count=1),
        ),
    ),
    name="quadinaros",
)
podracer(
    builds(
        sebulba.make,
        zen_partial=True,
        config=builds(
            sebulba.SebulbaConfig,
            num_envs="${environment.num_envs}",
            num_steps="${rollout.num_steps}",
            learner=builds(mesh, count=1),
            actor=builds(mesh, start=0, count=1),
        ),
    ),
    name="sebulba",
)

store(builds(curricula.default, zen_partial=True), group="curriculum", name="default")

scoring(builds(max, zen_partial=True), name="best")
scoring(builds(operator.itemgetter, -1), name="final")
scoring(builds(statistics.fmean, zen_partial=True), name="mean")
