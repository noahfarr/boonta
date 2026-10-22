def make(env_id: str, **kwargs):
    from .walk import Walk

    registry = {
        "walk": Walk,
    }
    return registry[env_id](**kwargs)
