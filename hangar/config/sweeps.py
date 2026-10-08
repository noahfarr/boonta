from . import algorithms, hyperparameters
from .store import place, store

store(
    dict(
        defaults=[{"override /hydra/sweeper": "carbs"}],
        hydra=dict(mode="MULTIRUN", sweeper=dict(metric="score", cost="cost", direction="maximize")),
    ),
    group="sweep",
    name="carbs",
    package="_global_",
)

for algorithm, space in algorithms.spaces.items():
    place(space, f"search/{algorithm}", package="hydra.sweeper")
for (algorithm, environment), space in hyperparameters.spaces.items():
    wide = [f"/search/{algorithm}"] if algorithm in algorithms.spaces else []
    place(
        dict(defaults=[*wide, "_self_"], **space),
        f"search/{algorithm}/{environment}",
        package="hydra.sweeper",
    )
