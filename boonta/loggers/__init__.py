from .file import FileLogger
from .logger import Logger, MultiLogger
from .orbax import OrbaxLogger
from .xdg import XdgLogger

__all__ = [
    "DashboardLogger",
    "FileLogger",
    "Logger",
    "MultiLogger",
    "OrbaxLogger",
    "WandbLogger",
    "XdgLogger",
]


def __getattr__(name):
    if name == "DashboardLogger":
        from .dashboard import DashboardLogger

        return DashboardLogger
    if name == "WandbLogger":
        from .wandb import WandbLogger

        return WandbLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
