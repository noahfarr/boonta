import flax.linen as nn

from boonta.utils.typing import Array


class ActorCritic(nn.Module):
    actor: nn.Module
    critic: nn.Module

    @nn.compact
    def __call__(self, x: Array, temperature: float | Array) -> tuple:
        return self.actor(x, temperature=temperature), self.critic(x)
