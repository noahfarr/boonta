import flax.linen as nn

from boonta.utils.typing import Array, Carry, Key, PyTree

from ..layers import LoRA as LoRAParams
from ..pretrained import init_fn
from .block import Block


class LoRA(Block):
    block: Block
    params: PyTree
    rank: int = 4
    alpha: float = 1.0

    @nn.compact
    def __call__(self, carry: Carry, x: Array, done: Array) -> tuple[Carry, Array]:
        base = self.variable("frozen_params", "base", init_fn(self.params)).value
        params = LoRAParams(rank=self.rank, alpha=self.alpha)(base)
        return self.block.apply({"params": params}, carry, x, done)

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        return self.block.initialize_carry(key, input_shape)
