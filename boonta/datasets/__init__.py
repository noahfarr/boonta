from . import agent_trove, fineweb, kinetix, minari
from .dataset import Dataset, DatasetState

registry = {
    "agent_trove": agent_trove.AgentTrove,
    "fineweb": fineweb.FineWeb,
    "kinetix": kinetix.make,
    "minari": minari.Minari,
}


def make(namespace, dataset_id, **kwargs):
    dataset_kwargs = kwargs.get("kwargs") or {}
    return registry[namespace](dataset_id, **dataset_kwargs)


__all__ = [
    "Dataset",
    "DatasetState",
    "agent_trove",
    "fineweb",
    "kinetix",
    "make",
    "minari",
    "registry",
]
