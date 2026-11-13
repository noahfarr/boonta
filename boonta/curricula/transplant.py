"""Uniform restarts that follow another arm's schedule over game progress.

ERD's sampler (uniform over eligible archive cells), except that each restart first draws a
progress bucket from a schedule of target shares per training epoch, then a cell uniformly among
the eligible cells in that bucket. Built to test whether a restart rule's benefit comes from game
progress alone: the schedule is copied from the rule under test (for Pac-Man, the forecast gain's
share of restarts per pellets-left bucket), and nothing else about which states it picks is kept.

A bucket with no eligible cell yet has its share spread over the buckets that have one, so a seed
that has not reached the late game restarts where it can. The realized shares are what the signal
probe logs as restart/bucket_k.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp

from boonta.algorithms import Algorithm
from boonta.environments.wrappers.archive_auto_reset import (
    ArchiveAutoReset, ArchiveAutoResetState, pick, unwrap)
from boonta.podracers.podracer import Lap, Pit
from boonta.utils import Array, Key
from boonta.utils.typing import Environment


def pac_man_pellet_bucket(bucket_pellets: int = 50, num_buckets: int = 7) -> Callable:
    """The pellets left in an archived PacMan state, in buckets: 0 is 0-49 pellets left."""

    def bucket(snapshot) -> Array:
        state = snapshot
        while not hasattr(state, "pellets"):
            state = state.env_state
        return jnp.clip(state.pellets // bucket_pellets, 0, num_buckets - 1).astype(jnp.int32)

    return bucket


@dataclass
class Transplant(Pit):
    environment: ArchiveAutoReset
    shares: Array  # (epochs, buckets): target share of restarts per bucket in each epoch
    epoch_steps: int
    bucket_fn: Callable
    k: float = 4.0
    restart_at: str = "rollout"
    num_buckets: int = field(init=False)

    def __post_init__(self):
        assert isinstance(self.environment, ArchiveAutoReset), "Transplant restarts from an ArchiveAutoReset"
        assert self.restart_at in ("rollout", "episode"), self.restart_at
        self.num_buckets = int(self.shares.shape[1])

    @property
    def num_placed(self) -> int:
        return int(round((1.0 - 1.0 / self.k) * self.environment.num_envs))

    def select(self, archive_auto_reset_state: ArchiveAutoResetState, key: Key, step: Array) -> Array:
        archive_state = archive_auto_reset_state.archive_state
        eligible = self.environment.archive.eligible(archive_state)
        buckets = self.bucket_fn(archive_state.cell_states.snapshot)
        epoch = jnp.clip(step // self.epoch_steps, 0, self.shares.shape[0] - 1)
        counts = jnp.zeros(self.num_buckets).at[jnp.where(eligible, buckets, self.num_buckets)].add(1.0, mode="drop")
        target = self.shares[epoch] * (counts > 0)
        target = jnp.where(jnp.sum(target) > 0, target / jnp.maximum(jnp.sum(target), 1e-30), (counts > 0) / jnp.maximum(jnp.sum(counts > 0), 1))
        # Each eligible cell gets its bucket's target share split evenly over the bucket's cells.
        weight = target[buckets] / jnp.maximum(counts[buckets], 1.0)
        logits = jnp.where(eligible & (weight > 0), jnp.log(jnp.maximum(weight, 1e-30)), -jnp.inf)
        return pick(key, logits, (self.num_placed,))

    def __call__(self, key: Key, state):
        select_key, assign_key = jax.random.split(key)
        step = state.algorithm_state.step

        def assign(node):
            if isinstance(node, ArchiveAutoResetState):
                slots = self.select(node, select_key, step)
                return self.environment.update(node, assign_key, slots=slots, preempt=self.restart_at == "rollout")
            return node

        environment_state = jax.tree.map(
            assign, state.environment_state, is_leaf=lambda node: isinstance(node, ArchiveAutoResetState)
        )
        return state.replace(environment_state=environment_state)


def make(
    algorithm: Algorithm,
    environment: Environment,
    schedule: str,
    bucket_fn: Callable,
    k: float = 4.0,
    restart_at: str = "rollout",
    **kwargs,
) -> tuple[Algorithm, Environment, Transplant, Lap | None]:
    table = json.loads(Path(schedule).read_text())
    shares = jnp.asarray(table["shares"], jnp.float32)
    pit = Transplant(unwrap(environment), shares, int(table["epoch_steps"]), bucket_fn, k, restart_at)
    return algorithm, environment, pit, None
