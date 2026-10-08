import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from boonta.environments.kaggriculture import (BOARD, MARKET_I0, NUM_PRODUCTS,
                                               seat_view)
from boonta.environments.kaggriculture import actions as A
from boonta.utils import load_checkpoint, newest
from boonta.utils.typing import Carry, Dtype, Key

from .distribution import SEATS, JobDistribution, unfold_seats


def none_bias_init(strength):
    def init(key, shape, dtype=jnp.float32):
        index = np.arange(A.ORDERS) * A.NUM_MARKET_ACTIONS + A.MARKET_NONE
        return jnp.zeros(shape, dtype).at[index].set(strength)

    return init


def signed_log(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def both_seats(obs):
    views = [seat_view(obs, seat) for seat in range(SEATS)]
    return {
        name: jnp.stack([view[name] for view in views], axis=1).reshape(
            -1, *views[0][name].shape[1:]
        )
        for name in views[0]
    }


def board_input(obs):
    tiles = obs["tiles"].astype(jnp.float32) / 8.0
    own, rival = tiles[:, 0], tiles[:, 1]
    batch, *_ = own.shape

    pos = obs["pos"][:, 0].astype(jnp.int32)
    live = obs["hands"][:, 0].astype(jnp.int32) + 1
    alive = (jnp.arange(A.UNITS)[None, :] < live[:, None]).astype(jnp.float32)
    flat = jnp.clip(pos[..., 1], 0, BOARD - 1) * BOARD + jnp.clip(pos[..., 0], 0, BOARD - 1)
    density = jax.vmap(lambda index, weight: jnp.zeros((A.TILES,)).at[index].add(weight))(
        flat, alive
    ).reshape(batch, BOARD, BOARD, 1)

    summary = jnp.concatenate(
        [
            signed_log(obs["money"].astype(jnp.float32)),
            obs["prices"].astype(jnp.float32) / 100.0,
            (obs["market"].astype(jnp.float32) - MARKET_I0) / 200.0,
            obs["shops"].astype(jnp.float32),
            obs["shed"][:, 0].astype(jnp.float32) / 20.0,
            obs["seeds"][:, 0].astype(jnp.float32) / 5.0,
            obs["hands"].astype(jnp.float32) / 8.0,
            obs["quadrants"].astype(jnp.float32),
            obs["day"].astype(jnp.float32)[:, None] / 30.0,
            obs["hour"].astype(jnp.float32)[:, None] / 24.0,
        ],
        axis=-1,
    )
    spread = jnp.broadcast_to(summary[:, None, None, :], (batch, BOARD, BOARD, summary.shape[-1]))
    return jnp.concatenate([own, rival, density, spread], axis=-1), summary


class Residual(nn.Module):
    features: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        def conv():
            return nn.Conv(self.features, (3, 3), dtype=self.dtype, param_dtype=self.param_dtype)

        def norm():
            return nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype)

        y = conv()(nn.relu(norm()(x)))
        y = conv()(nn.relu(norm()(y)))
        return x + y


class Encoder(nn.Module):
    features: int = 96
    blocks: int = 4
    dtype: Dtype | None = jnp.bfloat16
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, obs):
        grid, summary = board_input(obs)
        x = nn.Conv(
            self.features, (3, 3), dtype=self.dtype, param_dtype=self.param_dtype, name="stem"
        )(grid)
        for index in range(self.blocks):
            x = Residual(
                self.features, dtype=self.dtype, param_dtype=self.param_dtype, name=f"block_{index}"
            )(x)
        x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="trunk_norm")(x)
        pooled = jnp.concatenate([x.mean((1, 2)), x.max(axis=(1, 2)), summary.astype(x.dtype)], -1)
        return x, pooled


class Outputs(nn.Module):
    dtype: Dtype | None = jnp.bfloat16
    param_dtype: Dtype = jnp.float32
    market_none_bias: float = 0.0

    @nn.compact
    def __call__(self, spatial, state, obs):
        def dense(features):
            return nn.Dense(features, dtype=self.dtype, param_dtype=self.param_dtype)

        h = nn.relu(dense(256)(state))
        h = nn.relu(dense(256)(h))

        jobs = nn.Conv(A.NUM_JOBS, (1, 1), dtype=self.dtype, param_dtype=self.param_dtype)(spatial)
        jobs = jobs.reshape(jobs.shape[0], A.TILES, A.NUM_JOBS).astype(jnp.float32)

        market = nn.Dense(
            A.ORDERS * A.NUM_MARKET_ACTIONS,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.variance_scaling(0.01, "fan_in", "truncated_normal"),
            bias_init=none_bias_init(self.market_none_bias),
        )(h).reshape(-1, A.ORDERS, A.NUM_MARKET_ACTIONS)
        quantity = dense(A.NUM_MARKET_ACTIONS * A.NUM_QUANTITIES)(h)
        quantity = quantity.reshape(-1, A.NUM_MARKET_ACTIONS, A.NUM_QUANTITIES)

        travel = nn.softplus(dense(1)(h))[:, 0].astype(jnp.float32)
        value = dense(1)(h)[:, 0].astype(jnp.float32)
        flow = dense(NUM_PRODUCTS)(h).astype(jnp.float32)

        distance = A.unit_distance(obs)
        biased = jobs - travel[:, None, None] * distance[:, :, None]
        jobs_masked = jnp.where(A.legal_jobs(obs, 0), biased, jnp.finfo(jnp.float32).min)
        market_masked = jnp.where(
            A.legal_market(obs, 0)[:, None, :],
            market.astype(jnp.float32),
            jnp.finfo(jnp.float32).min,
        )
        logits = {
            "jobs_masked": jobs_masked.reshape(jobs_masked.shape[0], -1),
            "market": market_masked,
            "quantity": quantity.astype(jnp.float32),
            "travel": travel,
            "flow": flow,
            "live": obs["hands"][:, 0].astype(jnp.int32) + 1,
        }
        return logits, value


class Farmer(nn.Module):
    features: int = 96
    blocks: int = 4
    dtype: Dtype | None = jnp.bfloat16
    param_dtype: Dtype = jnp.float32
    market_none_bias: float = 0.0

    @nn.compact
    def __call__(self, obs, temperature):
        obs = both_seats(obs)
        spatial, state = Encoder(
            self.features, self.blocks, dtype=self.dtype, param_dtype=self.param_dtype
        )(obs)
        logits, value = Outputs(
            dtype=self.dtype, param_dtype=self.param_dtype, market_none_bias=self.market_none_bias
        )(spatial, state, obs)
        return JobDistribution(logits, temperature), unfold_seats(value)[..., None]


class Trunk(nn.Module):
    features: int = 96
    blocks: int = 4
    dtype: Dtype | None = jnp.bfloat16
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None, **kwargs):
        envs, steps = obs["tiles"].shape[:2]
        flat = jax.tree.map(lambda leaf: leaf.reshape(-1, *leaf.shape[2:]), obs)
        folded = both_seats(flat)
        spatial, state = Encoder(
            self.features, self.blocks, dtype=self.dtype, param_dtype=self.param_dtype
        )(folded)
        return {"state": state, "spatial": spatial, "obs": folded, "shape": (envs, steps)}


class Seated(nn.Module):
    torso: nn.Module

    @nn.compact
    def __call__(self, carry, x, done, *args, **kwargs):
        envs, steps = x["shape"]
        state = x["state"].reshape(envs, steps, SEATS, -1)
        state = jnp.swapaxes(state, 1, 2).reshape(envs * SEATS, steps, -1)

        mask = done
        if mask is not None:
            mask = jnp.broadcast_to(mask.reshape(envs, steps, -1)[:, :, :1], (envs, steps, SEATS))
            mask = jnp.swapaxes(mask, 1, 2).reshape(envs * SEATS, steps)

        folded = jax.tree.map(lambda leaf: leaf.reshape(envs * SEATS, *leaf.shape[2:]), carry)
        folded, state = self.torso(folded, state, mask, *args, **kwargs)

        carry = jax.tree.map(lambda leaf: leaf.reshape(envs, SEATS, *leaf.shape[1:]), folded)
        state = state.reshape(envs, SEATS, steps, -1)
        state = jnp.swapaxes(state, 1, 2).reshape(envs * steps * SEATS, -1)
        return carry, {**x, "state": state}

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        envs, *_ = input_shape
        return self.torso.initialize_carry(key, (envs, SEATS, input_shape[-1]))


def seat_major(leaf, envs, steps):
    leaf = leaf.reshape(envs, steps, SEATS, *leaf.shape[1:])
    return jnp.swapaxes(leaf, 1, 2).reshape(envs * SEATS, steps, *leaf.shape[3:])


class Heads(nn.Module):
    dtype: Dtype | None = jnp.bfloat16
    param_dtype: Dtype = jnp.float32
    market_none_bias: float = 0.0

    @nn.compact
    def __call__(self, x, temperature, **kwargs):
        envs, steps = x["shape"]
        logits, value = Outputs(
            dtype=self.dtype, param_dtype=self.param_dtype, market_none_bias=self.market_none_bias
        )(x["spatial"], x["state"].astype(self.dtype or self.param_dtype), x["obs"])
        value = unfold_seats(value).reshape(envs, steps, SEATS)[..., None]
        self.sow("intermediates", "value", value, reduce_fn=lambda _, value: value)
        logits = jax.tree.map(lambda leaf: seat_major(leaf, envs, steps), logits)
        return JobDistribution(logits, temperature)


class Value(nn.Module):
    dtype: Dtype | None = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x, **kwargs):
        envs, steps = x["shape"]
        h = x["state"].astype(self.dtype or self.param_dtype)

        def dense(features):
            return nn.Dense(features, dtype=self.dtype, param_dtype=self.param_dtype)

        h = nn.relu(dense(256)(h))
        h = nn.relu(dense(256)(h))
        value = dense(1)(h)[:, 0].astype(jnp.float32)
        return unfold_seats(value).reshape(envs, steps, SEATS)[..., None]


class Split(nn.Module):
    actor: nn.Module
    critic: nn.Module

    @nn.compact
    def __call__(self, *args, carry=None, **kwargs):
        walk, watch = carry if carry is not None else (None, None)
        walk, dist = self.actor(*args, carry=walk, **kwargs)
        watch, value = self.critic(*args, carry=watch, **kwargs)
        return (walk, watch), (dist, value)

    @nn.nowrap
    def initialize_carry(self, key: Key, input_shape: tuple[int, ...]) -> Carry:
        walk, watch = jax.random.split(key)
        return (
            self.actor.initialize_carry(walk, input_shape),
            self.critic.initialize_carry(watch, input_shape),
        )


def weights(tree):
    while isinstance(tree, dict):
        if "params" in tree:
            tree = tree["params"]
        elif set(tree) == {"pretrained"}:
            tree = tree["pretrained"]
        elif set(tree) == {"actor", "critic"}:
            tree = tree["actor"]
        else:
            return tree
    return tree


def clone(path):
    return weights(load_checkpoint(newest(path)))
