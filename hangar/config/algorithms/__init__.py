from ..sections import Exploration
from ..store import place, store

table = {}
spaces = {}

recurrent = [{"/torso": "default"}, {"/cell": "gru"}, {"/stack": "default"}]
offline = [{"/dataset": "minari/mujoco/expert"}, {"override /podracer": "quadinaros"}]
epsilon = Exploration(start=1.0, end=0.01, fraction=0.2)


class Algorithm:
    def __init__(self, node, name):
        self.node = node
        self.name = name

    def hyperparameters(self, environment, /, **node):
        table[self.name, environment] = node
        place(node, f"hyperparameters/{self.name}/{environment}", package="_global_")

    def search_space(self, environment=None, /, **space):
        spaces[self.name, environment] = space


def algorithm(node, /, name):
    store(node, group="algorithm", name=name, package="_global_")
    return Algorithm(node, name)


def cascade(table, algorithm, environment):
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
