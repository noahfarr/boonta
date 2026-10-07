from dataclasses import dataclass
from functools import partial, reduce

import flax.linen as nn
import jax
import jax.numpy as jnp

from boonta.utils import add_time_axis, get_attention_implementation
from boonta.utils.typing import Array, Dtype

from ..blocks import (GLU, joint_attention_mask, rotary_positional_embedding,
                      sinusoidal_time_embedding)
from .vit import ViT


@dataclass(frozen=True)
class Expert:
    features: int
    expansion_factor: int = 4
    hidden_dim: int | None = None


class JointSelfAttention(nn.Module):
    features: tuple[int, ...]
    num_heads: int
    num_groups: int
    head_dim: int
    max_wavelength: float
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, xs: tuple[Array, ...], mask: Array) -> tuple[Array, ...]:
        projection = partial(
            nn.DenseGeneral,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )

        def step(state, item):
            queries, keys, values, lengths = state
            i, x = item
            query = projection(
                features=(self.num_heads, self.head_dim), name=f"query_{i}"
            )(x)
            key = projection(
                features=(self.num_groups, self.head_dim), name=f"key_{i}"
            )(x)
            value = projection(
                features=(self.num_groups, self.head_dim), name=f"value_{i}"
            )(x)
            return (
                (*queries, query),
                (*keys, key),
                (*values, value),
                (
                    *lengths,
                    x.shape[1],
                ),
            )

        queries, keys, values, lengths = reduce(step, enumerate(xs), ((), (), (), ()))

        query = jnp.concatenate(queries, axis=1)
        key = jnp.concatenate(keys, axis=1)
        value = jnp.concatenate(values, axis=1)

        positions = jnp.arange(query.shape[1])
        query, key, _ = rotary_positional_embedding(
            query, key, positions, positions, max_wavelength=self.max_wavelength
        )

        implementation, attention_dtype = get_attention_implementation(
            query.shape[-1], query.shape[1]
        )
        attention = jax.nn.dot_product_attention(
            query.astype(attention_dtype),
            key.astype(attention_dtype),
            value.astype(attention_dtype),
            mask=mask[:, None],
            implementation=implementation,
        ).astype(query.dtype)

        def step(state, item):
            start, outputs = state
            i, (features, length) = item
            end = start + length
            output = nn.DenseGeneral(
                features,
                axis=(-2, -1),
                use_bias=False,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"output_projection_{i}",
            )(attention[:, start:end])
            return end, (*outputs, output)

        _, outputs = reduce(
            step, enumerate(zip(self.features, lengths, strict=True)), (0, ())
        )
        return outputs


class NoisyActionEmbedding(nn.Module):
    features: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        dense = partial(nn.Dense, dtype=self.dtype, param_dtype=self.param_dtype)
        x = dense(self.features)(x)
        x = nn.silu(x)
        return dense(self.features)(x)


class PI0Layer(nn.Module):
    experts: tuple[Expert, ...]
    num_heads: int
    num_groups: int
    head_dim: int
    max_wavelength: float
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, xs: tuple[Array, ...], mask: Array) -> tuple[Array, ...]:
        rms_norm = partial(nn.RMSNorm, dtype=self.dtype, param_dtype=self.param_dtype)

        skip = xs
        normed = tuple(
            rms_norm(name=f"attention_norm_{i}")(x) for i, x in enumerate(xs)
        )

        ys = JointSelfAttention(
            tuple(expert.features for expert in self.experts),
            self.num_heads,
            self.num_groups,
            self.head_dim,
            self.max_wavelength,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="attention",
        )(normed, mask)

        xs = tuple(x + y for x, y in zip(skip, ys, strict=True))

        def step(outputs, item):
            i, (expert, x) = item
            y = rms_norm(name=f"ge_glu_norm_{i}")(x)
            _, y = GLU(
                expert.features,
                expert.expansion_factor,
                hidden_dim=expert.hidden_dim,
                activation=partial(nn.gelu, approximate=True),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"ge_glu_{i}",
            )(None, y, None)
            return (*outputs, x + y)

        return reduce(step, enumerate(zip(self.experts, xs, strict=True)), ())


class PI0(nn.Module):
    vision: ViT
    experts: tuple[Expert, Expert]
    vocab_size: int
    action_dim: int
    action_horizon: int
    num_layers: int
    num_heads: int
    num_groups: int
    head_dim: int
    max_wavelength: float
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(
        self, obs: dict[str, Array], noisy_actions: Array, time: Array
    ) -> Array:
        image = obs["image"]
        language = obs["language"]
        language_mask = obs["language_mask"]
        state = obs["proprioception"]

        vlm, action_expert = self.experts
        dense = partial(nn.Dense, dtype=self.dtype, param_dtype=self.param_dtype)

        batch_size, *_ = image.shape
        _, image_embeddings = self.vision(
            None, image, jnp.zeros((batch_size,), dtype=bool)
        )
        image_embeddings = dense(vlm.features, name="image_embedding")(image_embeddings)
        language_embeddings = nn.Embed(
            self.vocab_size,
            vlm.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="language_embedding",
        )(language)
        prefix_embeddings = jnp.concatenate(
            [image_embeddings, language_embeddings], axis=1
        )

        state_embedding = dense(action_expert.features, name="state_embedding")(state)[
            :, None, :
        ]

        action_embeddings = dense(action_expert.features, name="action_embedding")(
            noisy_actions
        )
        time_embedding = sinusoidal_time_embedding(time, action_expert.features)
        time_embedding = jnp.broadcast_to(
            add_time_axis(time_embedding), action_embeddings.shape
        )

        noisy_action_embeddings = jnp.concatenate(
            [action_embeddings, time_embedding], axis=-1
        )
        noisy_action_embeddings = NoisyActionEmbedding(
            action_expert.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="noisy_action_embedding",
        )(noisy_action_embeddings)

        _, num_patches, _ = image_embeddings.shape
        _, language_length, _ = language_embeddings.shape
        _, action_horizon, _ = noisy_action_embeddings.shape
        prefix_length = num_patches + language_length

        segment_starts = jnp.concatenate(
            [
                jnp.zeros((batch_size, prefix_length), dtype=bool),
                jnp.ones((batch_size, 1), dtype=bool),
                jnp.concatenate(
                    [
                        jnp.ones((batch_size, 1), dtype=bool),
                        jnp.zeros((batch_size, action_horizon - 1), dtype=bool),
                    ],
                    axis=1,
                ),
            ],
            axis=1,
        )
        input_mask = jnp.concatenate(
            [
                jnp.ones((batch_size, num_patches), dtype=bool),
                language_mask,
                jnp.ones((batch_size, 1 + action_horizon), dtype=bool),
            ],
            axis=1,
        )
        mask = joint_attention_mask(input_mask, segment_starts)

        xs = (
            prefix_embeddings,
            jnp.concatenate([state_embedding, noisy_action_embeddings], axis=1),
        )
        for i in range(self.num_layers):
            xs = PI0Layer(
                self.experts,
                self.num_heads,
                self.num_groups,
                self.head_dim,
                self.max_wavelength,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"layers_{i}",
            )(xs, mask)

        rms_norm = partial(nn.RMSNorm, dtype=self.dtype, param_dtype=self.param_dtype)
        _, actions = tuple(
            rms_norm(name=f"output_norm_{i}")(x) for i, x in enumerate(xs)
        )

        return dense(self.action_dim, name="velocity")(
            actions[:, -self.action_horizon :]
        )
