import copy

from ..blocks import Block, Stack


def repeat(block: Block, num_layers: int = 1) -> Stack:
    return Stack(tuple(copy.deepcopy(block) for _ in range(num_layers)))
