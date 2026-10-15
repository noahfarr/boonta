from .artifact import Artifact, Checkpoint, Metrics, Text, Video
from .artisan import Artisan
from .checkpointer import Checkpointer
from .gauntlet import Gauntlet
from .render import Render
from .transcript import Transcript

__all__ = [
    "Artifact",
    "Artisan",
    "Checkpoint",
    "Checkpointer",
    "Gauntlet",
    "Metrics",
    "Render",
    "Text",
    "Transcript",
    "TrueSkill",
    "Video",
]


def __getattr__(name):
    if name == "TrueSkill":
        from .trueskill import TrueSkill

        return TrueSkill
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
