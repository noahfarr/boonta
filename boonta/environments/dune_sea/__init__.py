from .dune_sea import DuneSea, DuneSeaState
from .nca import NCA, measure_liveliness, render_field, symbol_palette

registry = {
    "dune_sea": DuneSea,
}


def make(env_id: str, **kwargs):
    return registry[env_id](**kwargs)


__all__ = [
    "DuneSea",
    "DuneSeaState",
    "NCA",
    "make",
    "measure_liveliness",
    "render_field",
    "symbol_palette",
]
