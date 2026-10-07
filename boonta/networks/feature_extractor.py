from collections.abc import Callable

import flax.linen as nn
import jax.numpy as jnp

from boonta.utils.typing import Array


class FeatureExtractor(nn.Module):
    observation_extractor: Callable
    action_extractor: Callable | None = None
    reward_extractor: Callable | None = None
    done_extractor: Callable | None = None

    @nn.compact
    def __call__(
        self,
        obs: Array,
        action: Array | None = None,
        reward: Array | None = None,
        done: Array | None = None,
        **kwargs: Array,
    ) -> Array:
        embeddings = [self.observation_extractor(obs)]
        if self.action_extractor is not None:
            embeddings.append(self.action_extractor(action))
        if self.reward_extractor is not None:
            embeddings.append(self.reward_extractor(reward))
        if self.done_extractor is not None:
            embeddings.append(self.done_extractor(done))
        return jnp.concatenate(embeddings, axis=-1)
