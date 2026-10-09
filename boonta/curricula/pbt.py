from boonta.algorithms import Algorithm
from boonta.algorithms.wrappers.pbt import PBT
from boonta.podracers.podracer import Lap, Pit
from boonta.utils.typing import Environment


def pbt(
    algorithm: Algorithm,
    environment: Environment,
    members: int,
    seeds: int,
    interval: int,
    fraction: float,
    threshold: float,
    window: int,
    low: float,
    high: float,
    factors,
    **kwargs,
) -> tuple[PBT, Environment, Pit, Lap]:
    algorithm = PBT(
        algorithm,
        members=members,
        seeds=seeds,
        interval=interval,
        fraction=fraction,
        threshold=threshold,
        window=window,
        low=low,
        high=high,
        factors=tuple(factors),
    )
    return algorithm, environment, lambda state: state, lambda state: state
