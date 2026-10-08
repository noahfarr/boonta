from ..sections import Exploration
from ..store import place, store

algorithm = store(group="algorithm", package="_global_")

table = {}
spaces = {}

recurrent = [{"/torso": "default"}, {"/cell": "gru"}, {"/stack": "default"}]
offline = [{"/dataset": "minari/mujoco/expert"}, {"override /podracer": "quadinaros"}]
epsilon = Exploration(start=1.0, end=0.01, fraction=0.2)


def special(algorithm, environment, /, **node):
    table[algorithm, environment] = node
    place(node, f"hyperparameters/{algorithm}/{environment}", package="_global_")


def search(algorithm, environment=None, /, **space):
    spaces[algorithm, environment] = space


def lookup(table, algorithm, environment):
    parts = environment.split("/")
    while parts:
        path = "/".join(parts)
        if (algorithm, path) in table:
            return f"{algorithm}/{path}"
        parts.pop()
    return algorithm


from . import (  # noqa: E402, F401
    bc,
    dqn,
    grpo,
    ippo,
    iql,
    mappo,
    mmd,
    ppo,
    pqn,
    recurrent_bc,
    recurrent_dqn,
    recurrent_grpo,
    recurrent_ppo,
    recurrent_pqn,
    recurrent_pupo,
    recurrent_sac,
    reppo,
    sac,
)
