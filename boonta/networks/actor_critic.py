import flax.linen as nn
import jax
import jax.numpy as jnp

from boonta.utils.typing import Array, Carry, Key


class ActorCritic(nn.Module):
    actor: nn.Module
    critic: nn.Module

    @nn.compact
    def __call__(self, x: Array, temperature: float | Array) -> tuple:
        distribution = self.actor(x, temperature=temperature)
        return distribution, self.critic(x).astype(jnp.float32)


class RecurrentActorCritic(nn.Module):
    """Compose independent recurrent actor and critic Networks and carries."""

    actor: nn.Module
    critic: nn.Module

    @nn.compact
    def __call__(
        self,
        obs,
        action=None,
        reward=None,
        done=None,
        carry: Carry = None,
        temperature: float | Array = 1.0,
        **kwargs,
    ) -> tuple:
        actor_carry, distribution = self.actor(
            obs, action, reward, done,
            carry=None if carry is None else carry["actor"],
            temperature=temperature,
            **kwargs,
        )
        critic_carry, value = self.critic(
            obs, action, reward, done,
            carry=None if carry is None else carry["critic"],
            **kwargs,
        )
        return {"actor": actor_carry, "critic": critic_carry}, (
            distribution, value.astype(jnp.float32)
        )

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        actor_key, critic_key = jax.random.split(key)
        return {
            "actor": self.actor.initialize_carry(actor_key, input_shape),
            "critic": self.critic.initialize_carry(critic_key, input_shape),
        }
