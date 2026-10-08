from ..sections import Environment
from . import environment

for name in ("find_return", "king_hill", "prey", "scouts", "snake", "traveling_salesman"):
    environment(Environment(namespace="mapox", suite="mapox", env_id=name), name=f"mapox/{name}")
