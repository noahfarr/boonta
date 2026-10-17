from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
from flax import struct

from boonta.algorithms import Algorithm
from boonta.environments.wrappers.archive_auto_reset import (
    ArchiveAutoReset, ArchiveAutoResetState, pick, unwrap)
from boonta.podracers.podracer import Lap, Pit
from boonta.utils import Array, Key, PyTree, Trajectory
from boonta.utils.typing import Environment


@struct.dataclass
class Outcomes:
    """Discounted returns and lengths of finished episodes, counted per start."""

    episodes: Array
    returns: Array
    squares: Array
    lengths: Array


@struct.dataclass
class NeymanState:
    cells: Outcomes
    rho: Outcomes
    reach: Array
    visits: Array
    origins: Array
    origin: Array
    from_rho: Array
    earned: Array
    discount: Array
    length: Array


def capped(outcomes: Outcomes, episodes: Array, returns: Array, squares: Array, lengths: Array,
           history_length: int) -> Outcomes:
    """Adds new episodes and shrinks the old ones so each start remembers about history_length."""
    room = jnp.maximum(history_length - episodes, 0.0)
    keep = jnp.where(
        outcomes.episodes + episodes > history_length,
        room / jnp.maximum(outcomes.episodes, 1e-12),
        1.0,
    )
    return Outcomes(
        episodes=keep * outcomes.episodes + episodes,
        returns=keep * outcomes.returns + returns,
        squares=keep * outcomes.squares + squares,
        lengths=keep * outcomes.lengths + lengths,
    )


def moments(outcomes: Outcomes, prior_variance: Array, prior_length: Array, prior_count: float):
    """Return variance and episode length, shrunk towards the pooled values by prior_count episodes."""
    count = jnp.maximum(outcomes.episodes, 1e-12)
    deviations = jnp.maximum(outcomes.squares - outcomes.returns**2 / count, 0.0)
    weight = jnp.maximum(outcomes.episodes + prior_count, 1e-12)
    variance = (deviations + prior_count * prior_variance) / weight
    length = jnp.where(
        outcomes.episodes + prior_count > 0.0,
        (outcomes.lengths + prior_count * prior_length) / weight,
        prior_length,
    )
    return variance, length


@dataclass
class Neyman(Pit):
    """Restarts each episode from a start drawn in proportion to W * S / sqrt(L).

    Neyman allocation with costs: S is the spread of the discounted return of episodes from a start,
    L their length in steps, and W the weight J_rho puts on the start. With relevance, W is the share
    of rho's visits the start's cell holds and rho is one more start; without it, every archived start
    weighs the same and rho keeps rho_share of the episodes.
    """

    environment: ArchiveAutoReset
    relevance: bool = True
    rho_share: float = 0.25
    importance_weights: bool = False
    max_weight: float = 10.0
    history_length: float = 100.0
    prior_count: float = 2.0
    rate: float = 0.1
    gamma: float = 0.99

    def __post_init__(self):
        assert isinstance(
            self.environment, ArchiveAutoReset
        ), "Neyman restarts from an ArchiveAutoReset, and the environment wraps none"
        assert 0.0 <= self.rho_share <= 1.0, self.rho_share

    def init(self, key: Key, state: PyTree) -> NeymanState:
        archive_size = self.environment.archive.archive_size
        num_envs = self.environment.num_envs

        def blank(*shape):
            return Outcomes(*(jnp.zeros(shape, jnp.float32) for _ in range(4)))

        return NeymanState(
            cells=blank(archive_size),
            rho=blank(),
            reach=jnp.zeros(archive_size, jnp.float32),
            visits=jnp.zeros(archive_size, jnp.float32),
            origins=jnp.zeros(archive_size, jnp.float32),
            origin=jnp.full(num_envs, -1, jnp.int32),
            from_rho=jnp.zeros(num_envs, bool),
            earned=jnp.zeros(num_envs, jnp.float32),
            discount=jnp.ones(num_envs, jnp.float32),
            length=jnp.zeros(num_envs, jnp.float32),
        )

    def update(
        self, state: NeymanState, key: Key, trajectory: Trajectory, algorithm_state: PyTree
    ) -> NeymanState:
        transitions = trajectory.transitions
        archive_size = self.environment.archive.archive_size
        first, second = transitions.first, transitions.second
        num_steps, num_envs = jnp.shape(second.done)
        reward = jnp.reshape(second.reward, (num_steps, num_envs, -1)).sum(axis=-1)

        def follow(carry, inputs):
            origin, from_rho, earned, discount, length = carry
            reward, done, truncated, cell, rho = inputs
            earned = earned + discount * reward
            length = length + 1.0
            finished = done & ~truncated & (origin >= 0)
            outcome = (finished, origin, from_rho, earned, length)
            carry = (
                jnp.where(done, cell, origin),
                jnp.where(done, rho, from_rho),
                jnp.where(done, 0.0, earned),
                jnp.where(done, 1.0, discount * self.gamma),
                jnp.where(done, 0.0, length),
            )
            return carry, outcome

        (next_origin, next_from_rho, next_earned, next_discount, next_length), (
            finished, origin, from_rho, earned, length
        ) = jax.lax.scan(
            follow,
            (state.origin, state.from_rho, state.earned, state.discount, state.length),
            (reward, second.done, second.truncated.astype(bool), second.info["cell"], second.info["rho"]),
        )

        def tally(mask, value):
            slots = jnp.where(mask, origin, archive_size).reshape(-1)
            return jnp.zeros(archive_size).at[slots].add(jnp.reshape(value, (-1,)), mode="drop")

        from_cells = finished & ~from_rho
        from_start = finished & from_rho
        cells = capped(
            state.cells,
            tally(from_cells, 1.0),
            tally(from_cells, earned),
            tally(from_cells, earned**2),
            tally(from_cells, length),
            self.history_length,
        )
        rho = capped(
            state.rho,
            jnp.sum(from_start, dtype=jnp.float32),
            jnp.sum(jnp.where(from_start, earned, 0.0)),
            jnp.sum(jnp.where(from_start, earned**2, 0.0)),
            jnp.sum(jnp.where(from_start, length, 0.0)),
            self.history_length,
        )

        def counted(mask, slots):
            slots = jnp.where(mask & (slots >= 0), slots, archive_size).reshape(-1)
            return jnp.zeros(archive_size).at[slots].add(1.0, mode="drop")

        decay = 1.0 - self.rate
        visited = first.info["cell"]
        started = second.done & second.info["rho"]
        return NeymanState(
            cells=cells,
            rho=rho,
            reach=decay * state.reach + counted(first.info["rho"], visited),
            visits=decay * state.visits + counted(jnp.ones_like(started), visited),
            origins=decay * state.origins + counted(started, second.info["cell"]),
            origin=next_origin,
            from_rho=next_from_rho,
            earned=next_earned,
            discount=next_discount,
            length=next_length,
        )

    def allocate(self, state: NeymanState, eligible_mask: Array) -> tuple[Array, Array]:
        """Probabilities over the archived starts, and the probability of starting from rho."""
        tried = state.cells.episodes > 0
        pooled = jax.tree.map(
            lambda cell, start: jnp.sum(jnp.where(tried, cell, 0.0)) + start, state.cells, state.rho
        )
        prior_variance, prior_length = moments(pooled, 0.0, 1.0, 0.0)
        variance, length = moments(state.cells, prior_variance, prior_length, self.prior_count)
        rho_variance, rho_length = moments(state.rho, prior_variance, prior_length, self.prior_count)
        candidates = eligible_mask & (state.origins <= 0.0)
        spread = jnp.where(candidates, jnp.sqrt(variance / jnp.maximum(length, 1.0)), 0.0)
        rho_spread = jnp.sqrt(rho_variance / jnp.maximum(rho_length, 1.0))
        uniform = candidates / jnp.maximum(jnp.sum(candidates, dtype=jnp.float32), 1.0)
        if self.relevance:
            reach = state.reach / jnp.maximum(jnp.sum(state.reach), 1e-12)
            weight = jnp.where(candidates, reach, 0.0)
            rho_weight = jnp.sum(jnp.where(state.origins > 0.0, reach, 0.0))
            scores = weight * spread
            rho_score = rho_weight * rho_spread
            total = jnp.sum(scores) + rho_score
            fallback = total <= 0.0
            probabilities = jnp.where(
                fallback, (1.0 - self.rho_share) * uniform, scores / jnp.maximum(total, 1e-30)
            )
            return probabilities, jnp.where(fallback, self.rho_share, rho_score / jnp.maximum(total, 1e-30))
        total = jnp.sum(spread)
        share = jnp.where(total > 0.0, spread / jnp.maximum(total, 1e-30), uniform)
        return (1.0 - self.rho_share) * share, jnp.float32(self.rho_share)

    def slot_weights(self, state: NeymanState) -> Array:
        """d_rho / d_mu per cell, so a batch of restarted episodes estimates the gradient of J_rho."""
        reach = state.reach / jnp.maximum(jnp.sum(state.reach), 1e-12)
        visits = state.visits / jnp.maximum(jnp.sum(state.visits), 1e-12)
        return jnp.where(visits > 0.0, jnp.minimum(reach / jnp.maximum(visits, 1e-30), self.max_weight), 1.0)

    def __call__(self, key: Key, state):
        neyman_state = state.pit_state
        num_envs = self.environment.num_envs

        def assign(node):
            if isinstance(node, ArchiveAutoResetState):
                eligible_mask = self.environment.archive.eligible(node.archive_state)
                probabilities, rho_probability = self.allocate(neyman_state, eligible_mask)
                logits = jnp.log(jnp.append(probabilities, rho_probability))
                drawn = pick(key, jnp.where(jnp.isfinite(logits), logits, -jnp.inf), (num_envs,))
                archive_size, = jnp.shape(probabilities)
                slots = jnp.where(drawn == archive_size, -1, drawn)
                self.report(neyman_state, probabilities, rho_probability)
                return self.environment.update(
                    node,
                    key,
                    slots=slots,
                    preempt=False,
                    slot_weights=self.slot_weights(neyman_state) if self.importance_weights else None,
                )
            return node

        environment_state = jax.tree.map(
            assign,
            state.environment_state,
            is_leaf=lambda node: isinstance(node, ArchiveAutoResetState),
        )
        return state.replace(environment_state=environment_state)

    def report(self, state: NeymanState, probabilities: Array, rho_probability: Array) -> None:
        mass = jnp.append(probabilities, rho_probability)
        entropy = -jnp.sum(jnp.where(mass > 0.0, mass * jnp.log(jnp.maximum(mass, 1e-30)), 0.0))
        tried = state.cells.episodes > 0
        lox.log(
            {
                "mu/rho_share": rho_probability,
                "mu/support": jnp.exp(entropy),
                "mu/peak": jnp.max(probabilities),
                "outcomes/tried": jnp.sum(tried, dtype=jnp.float32),
                "outcomes/rho_episodes": state.rho.episodes,
                "outcomes/max_weight": jnp.max(self.slot_weights(state)),
            }
        )


def make(
    algorithm: Algorithm,
    environment: Environment,
    relevance: bool = True,
    rho_share: float = 0.25,
    importance_weights: bool = False,
    max_weight: float = 10.0,
    history_length: float = 100.0,
    prior_count: float = 2.0,
    rate: float = 0.1,
    gamma: float = 0.99,
    **kwargs,
) -> tuple[Algorithm, Environment, Neyman, Lap | None]:
    pit = Neyman(
        unwrap(environment),
        relevance=relevance,
        rho_share=rho_share,
        importance_weights=importance_weights,
        max_weight=max_weight,
        history_length=history_length,
        prior_count=prior_count,
        rate=rate,
        gamma=gamma,
    )
    return algorithm, environment, pit, None
