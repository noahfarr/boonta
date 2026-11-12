from .chain import Chain, ChainState, Scorecard, cell

__all__ = ["CombinationLock", "CombinationLockState", "Scorecard", "cell"]

CombinationLockState = ChainState


class CombinationLock(Chain):
    """A row of cells, each opened by its own seeded secret action, which pays one.

    A wrong action ends the episode, so an episode from the start reaches a cell only by answering
    every cell before it, and the deep cells stay rare until the shallow ones are learned. An optional
    hazard also ends the episode at random after every answer.
    """

    def __init__(self, length: int = 64, num_actions: int = 4, hazard: float = 0.0, seed: int = 0):
        super().__init__(length=length, num_actions=num_actions, hazard=hazard, seed=seed, lock=True)
