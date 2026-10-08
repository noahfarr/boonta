from .algorithms import spaces
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

for (algorithm, environment), space in spaces.items():
    if environment is None:
        place(space, f"search_space/{algorithm}", package="hydra.sweeper")
        continue
    wide = [f"/search_space/{algorithm}"] if (algorithm, None) in spaces else []
    place(
        dict(defaults=[*wide, "_self_"], **space),
        f"search_space/{algorithm}/{environment}",
        package="hydra.sweeper",
    )
