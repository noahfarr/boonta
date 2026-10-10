from .forks import Forks, ForksState

registry = {"forks": Forks}


def make(env_id: str, **kwargs):
    return registry[env_id](**kwargs)


__all__ = ["Forks", "ForksState", "make"]
