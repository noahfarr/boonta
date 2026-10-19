registry = {}


def make(env_id: str, **kwargs):
    return registry[env_id](**kwargs)


__all__ = [
    "make",
]
