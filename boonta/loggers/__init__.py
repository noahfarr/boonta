from .file import FileLogger
from .logger import Logger, MultiLogger
from .orbax import OrbaxLogger

__all__ = [
    "DashboardLogger",
    "FileLogger",
    "Logger",
    "MultiLogger",
    "OrbaxLogger",
    "WandbLogger",
]


def __getattr__(name):
    # wandb and rich are optional dependencies, so these load on first use
    if name == "DashboardLogger":
        from .dashboard import DashboardLogger

        return DashboardLogger
    if name == "WandbLogger":
        from .wandb import WandbLogger

        return WandbLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
