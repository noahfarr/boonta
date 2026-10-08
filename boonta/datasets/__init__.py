from . import kinetix, minari
from .dataset import Dataset, DatasetState

registry = {
    "kinetix": kinetix.make,
    "minari": minari.make,
}


def make(namespace, dataset_id, **kwargs):
    dataset_kwargs = kwargs.get("kwargs") or {}
    return registry[namespace](dataset_id, **dataset_kwargs)


__all__ = [
    "Dataset",
    "DatasetState",
    "kinetix",
    "make",
    "minari",
    "registry",
]
