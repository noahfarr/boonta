import jax
import jax.numpy as jnp
from flax import struct

from boonta.environments.environment import Environment
from boonta.environments.spaces import Space
from boonta.utils import Array, Key, Timestep

HIGHER, LOWER, CORRECT, INVALID, START = 0, 1, 2, 3, 4


@struct.dataclass
class GuessNumberState:
    secret: Array
    guesses: Array
    feedback: Array


class GuessNumber(Environment):
    def __init__(self, low: int = 1, high: int = 100, num_guesses: int = 7):
        self.low = low
        self.high = high
        self.num_guesses = num_guesses

    def init(self, key: Key) -> tuple[GuessNumberState, Timestep]:
        state = GuessNumberState(
            secret=jax.random.randint(key, (), self.low, self.high + 1),
            guesses=jnp.int32(0),
            feedback=jnp.int32(START),
        )
        timestep = Timestep(
            obs=state.feedback,
            action=jnp.int32(0),
            reward=jnp.float32(0.0),
            terminated=jnp.bool_(True),
            truncated=jnp.bool_(False),
            info={"won": jnp.float32(0.0), "guesses": jnp.float32(0.0)},
        )
        return state, timestep

    def step(self, key: Key, state: GuessNumberState, action: Array) -> tuple[GuessNumberState, Timestep]:
        guess = jnp.asarray(action, jnp.int32)
        valid = (guess >= self.low) & (guess <= self.high)
        won = valid & (guess == state.secret)
        feedback = jnp.where(
            ~valid, INVALID, jnp.where(won, CORRECT, jnp.where(guess < state.secret, HIGHER, LOWER))
        )
        guesses = state.guesses + 1
        state = state.replace(guesses=guesses, feedback=feedback.astype(jnp.int32))
        timestep = Timestep(
            obs=state.feedback,
            action=guess,
            reward=won.astype(jnp.float32),
            terminated=won | (guesses >= self.num_guesses),
            truncated=jnp.bool_(False),
            info={"won": won.astype(jnp.float32), "guesses": guesses.astype(jnp.float32)},
        )
        return state, timestep

    def observation_space(self) -> Space:
        return Space(shape=(), dtype=jnp.int32, low=0, high=START)

    def action_space(self) -> Space:
        return Space(shape=(), dtype=jnp.int32, low=self.low, high=self.high)

    def time_limit(self) -> int:
        return self.num_guesses
