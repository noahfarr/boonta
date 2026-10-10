from .chain_of_forks import ChainOfForks, ChainOfForksState

registry = {"chain_of_forks": ChainOfForks}


def make(env_id: str, **kwargs):
    return registry[env_id](**kwargs)


__all__ = ["ChainOfForks", "ChainOfForksState", "make"]
