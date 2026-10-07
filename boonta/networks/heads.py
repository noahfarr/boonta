import distrax
import flax.linen as nn

from boonta.utils.typing import Array

from . import distributions


class Categorical(nn.Module):
    layer: nn.Module
    reinterpreted_batch_ndims: int = 0

    def setup(self):
        nn.share_scope(self, self.layer)

    def __call__(
        self, x: Array, temperature: float | Array, **kwargs
    ) -> distributions.Categorical | distributions.Independent:
        return distributions.categorical(
            self.layer(x, **kwargs), temperature, self.reinterpreted_batch_ndims
        )


class EpsilonGreedy(nn.Module):
    layer: nn.Module

    def setup(self):
        nn.share_scope(self, self.layer)

    def __call__(
        self, x: Array, temperature: float | Array, **kwargs
    ) -> distrax.EpsilonGreedy:
        return distributions.epsilon_greedy(self.layer(x, **kwargs), temperature)


class Gaussian(nn.Module):
    layer: nn.Module

    def setup(self):
        nn.share_scope(self, self.layer)

    def __call__(
        self, x: Array, temperature: float | Array, **kwargs
    ) -> distrax.MultivariateNormalDiag:
        return distributions.gaussian(self.layer(x, **kwargs), temperature)


class SquashedGaussian(nn.Module):
    layer: nn.Module

    def setup(self):
        nn.share_scope(self, self.layer)

    def __call__(
        self, x: Array, temperature: float | Array, **kwargs
    ) -> distrax.Transformed:
        return distributions.squashed_gaussian(self.layer(x, **kwargs), temperature)
