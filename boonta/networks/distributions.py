import distrax
import jax
import jax.numpy as jnp

from boonta.utils.typing import Array, Key


class Categorical(distrax.Categorical):
    def __init__(self, preferences: Array, temperature: float | Array = 1.0):
        greedy = jax.nn.one_hot(
            jnp.argmax(jax.nn.log_softmax(preferences), axis=-1),
            preferences.shape[-1],
            dtype=bool,
        )
        cooled = preferences / jnp.where(temperature > 0, temperature, 1.0)
        super().__init__(
            logits=jnp.where(temperature > 0, cooled, jnp.where(greedy, 0.0, -jnp.inf))
        )
        self.preferences = preferences

    def _sample_n(self, key: Key, n: int) -> Array:
        gumbel = jax.random.gumbel(key, (n, *self.logits.shape), self.logits.dtype)
        return jnp.argmax(self.logits + gumbel, axis=-1).astype(self._dtype)


class Independent(distrax.Independent):
    @property
    def logits(self) -> Array:
        return self.distribution.logits


def categorical(
    logits: Array, temperature: float | Array = 1.0, reinterpreted_batch_ndims: int = 0
) -> Categorical | Independent:
    distribution = Categorical(logits.astype(jnp.float32), temperature)
    if reinterpreted_batch_ndims:
        return Independent(distribution, reinterpreted_batch_ndims)
    return distribution


def epsilon_greedy(
    preferences: Array, temperature: float | Array = 1.0
) -> distrax.EpsilonGreedy:
    return distrax.EpsilonGreedy(
        preferences=preferences.astype(jnp.float32), epsilon=temperature
    )


def split(outputs: Array, temperature: float | Array) -> tuple[Array, Array]:
    mean, scale = jnp.split(outputs.astype(jnp.float32), 2, axis=-1)
    return mean, temperature * (jax.nn.softplus(scale) + 1e-3)


def gaussian(
    outputs: Array, temperature: float | Array = 1.0
) -> distrax.MultivariateNormalDiag:
    return distrax.MultivariateNormalDiag(*split(outputs, temperature))


def squashed_gaussian(
    outputs: Array, temperature: float | Array = 1.0
) -> distrax.Transformed:
    return distrax.Transformed(
        distrax.MultivariateNormalDiag(*split(outputs, temperature)),
        distrax.Block(distrax.Tanh(), 1),
    )
