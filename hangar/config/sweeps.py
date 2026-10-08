from .algorithms import spaces
from .store import place

for (algorithm, environment), space in spaces.items():
    if environment is None:
        place(space, f"search_space/{algorithm}", package="search_space")
        continue
    wide = [f"/search_space/{algorithm}"] if (algorithm, None) in spaces else []
    place(
        dict(defaults=[*wide, "_self_"], **space),
        f"search_space/{algorithm}/{environment}",
        package="search_space",
    )
