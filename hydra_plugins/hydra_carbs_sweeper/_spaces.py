import math

import attr
from carbs import LinearSpace, LogitSpace, LogSpace, RealNumberSpace

from .config import Distribution


@attr.s(auto_attribs=True, hash=True)
class Pow2Space(RealNumberSpace):
    min: float = 1.0
    max: float = float("+inf")
    is_integer: bool = True

    def basic_from_param(self, value):
        return math.log2(value) / self.scale

    def param_from_basic(self, value, is_rounded: bool = True):
        value = value * self.scale
        if is_rounded:
            value = round(value / self.rounding_factor) * self.rounding_factor
        return 2**value

    def round_tensor_in_basic(self, value):
        import torch

        return (
            torch.round(value * self.scale / self.rounding_factor)
            * self.rounding_factor
            / self.scale
        )

    @property
    def plot_scale(self) -> str:
        return "log"


def space(entry):
    name = (
        entry.distribution.name
        if isinstance(entry.distribution, Distribution)
        else str(entry.distribution)
    )
    bounds = dict(min=float(entry.min), max=float(entry.max))
    rounding = dict(rounding_factor=entry.rounding_factor)

    if name == "uniform":
        return LinearSpace(**bounds, scale=scale(entry, entry.max - entry.min))
    if name == "int_uniform":
        return LinearSpace(
            **bounds,
            scale=scale(entry, entry.max - entry.min),
            is_integer=True,
            **rounding,
        )
    if name == "uniform_pow2":
        width = math.log2(entry.max) - math.log2(entry.min)
        return Pow2Space(**bounds, scale=scale(entry, width), **rounding)
    if name == "log_normal":
        width = math.log(entry.max) - math.log(entry.min)
        return LogSpace(**bounds, scale=scale(entry, width))
    if name == "logit_normal":
        return LogitSpace(**bounds, scale=scale(entry, 1.0))
    raise ValueError(f"unknown distribution {name!r}")


def scale(entry, width: float) -> float:
    if entry.scale is not None:
        return float(entry.scale)
    return max(width / 4.0, 1e-6)


def center(entry) -> float:
    if entry.center is not None:
        return float(entry.center)

    name = (
        entry.distribution.name
        if isinstance(entry.distribution, Distribution)
        else str(entry.distribution)
    )
    if name in ("log_normal", "uniform_pow2"):
        return math.sqrt(float(entry.min) * float(entry.max))
    return (float(entry.min) + float(entry.max)) / 2.0
