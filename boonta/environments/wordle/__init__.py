from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from boonta.utils import Array, Key, Timestep

from ..environment import Environment
from ..spaces import Space

WORD_LENGTH = 5
NUM_LETTERS = 26

ABSENT, PRESENT, CORRECT = 0, 1, 2

INVALID_PENALTY = 0.0
YELLOW_REWARD = 0.0
GREEN_REWARD = 0.0


@struct.dataclass
class EnvState:
    secret_answer: Array
    guess_feedback: Array
    guess_revealed: Array
    guess_count: Array
    matched_secret: Array


def feedback(guess: Array, secret: Array) -> Array:
    correct = guess == secret
    counts = (
        jnp.zeros(NUM_LETTERS, jnp.int32).at[secret].add((~correct).astype(jnp.int32))
    )

    def mark(counts, x):
        letter, is_correct = x
        present = ~is_correct & (counts[letter] > 0)
        return counts.at[letter].add(-present.astype(jnp.int32)), present

    _, present = jax.lax.scan(mark, counts, (guess, correct))
    return jnp.where(correct, CORRECT, jnp.where(present, PRESENT, ABSENT))


def encode(letters: Array) -> Array:
    weights = NUM_LETTERS ** jnp.arange(WORD_LENGTH, dtype=jnp.int32)
    return jnp.sum(letters * weights, dtype=jnp.int32)


def is_allowed(word_id: Array, allowed_guesses: Array) -> Array:
    index = jnp.clip(
        jnp.searchsorted(allowed_guesses, word_id), 0, allowed_guesses.shape[0] - 1
    )
    return allowed_guesses[index] == word_id


def get_obs(state: EnvState) -> Array:
    cells = jax.nn.one_hot(state.guess_feedback, 3, dtype=jnp.float32)
    return jnp.where(state.guess_revealed, cells.reshape(-1), 0.0)


class Wordle(Environment):

    def __init__(
        self,
        answer_pool: np.ndarray,
        guess_pool: np.ndarray,
        num_guesses: int = 6,
        invalid_penalty: float = INVALID_PENALTY,
        yellow_reward: float = YELLOW_REWARD,
        green_reward: float = GREEN_REWARD,
    ):
        self.answer_pool = jnp.asarray(answer_pool, dtype=jnp.int32)
        self.guess_pool = jnp.asarray(guess_pool, dtype=jnp.int32)
        self.num_guesses = num_guesses
        self.invalid_penalty = invalid_penalty
        self.yellow_reward = yellow_reward
        self.green_reward = green_reward

    def init(
        self, key: Key
    ) -> tuple[EnvState, Timestep]:
        index = jax.random.randint(key, (), 0, self.answer_pool.shape[0])
        state = EnvState(
            secret_answer=self.answer_pool[index],
            guess_feedback=jnp.zeros(WORD_LENGTH, jnp.int32),
            guess_revealed=jnp.bool_(False),
            guess_count=jnp.int32(0),
            matched_secret=jnp.bool_(False),
        )
        action_space = self.action_space()
        action = jnp.zeros(action_space.shape, action_space.dtype)
        _, shapes = jax.eval_shape(self.step, key, state, action)
        timestep = Timestep(
            obs=get_obs(state),
            action=action,
            reward=jnp.zeros(shapes.reward.shape, shapes.reward.dtype),
            terminated=jnp.ones(shapes.terminated.shape, shapes.terminated.dtype),
            truncated=jnp.zeros(shapes.truncated.shape, shapes.truncated.dtype),
            info=jax.tree.map(
                lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shapes.info
            ),
        )
        return state, timestep

    def step(
        self,
        key: Key,
        state: EnvState,
        action: Array,
    ) -> tuple[EnvState, Timestep]:
        guess = jnp.asarray(action, jnp.int32)
        legal_guesses = is_allowed(encode(guess), self.guess_pool)
        matched_secret = jnp.all(guess == state.secret_answer)

        guess_feedback = feedback(guess, state.secret_answer)
        revealed_feedback = jnp.where(legal_guesses, guess_feedback, ABSENT)

        state = state.replace(
            guess_feedback=guess_feedback,
            guess_revealed=legal_guesses,
            guess_count=state.guess_count + legal_guesses.astype(jnp.int32),
            matched_secret=matched_secret,
        )
        reward = (
            matched_secret.astype(jnp.float32)
            + self.green_reward
            * (revealed_feedback == CORRECT).sum().astype(jnp.float32)
            + self.yellow_reward
            * (revealed_feedback == PRESENT).sum().astype(jnp.float32)
            - self.invalid_penalty * (~legal_guesses).astype(jnp.float32)
        )
        terminated = state.matched_secret | (state.guess_count >= self.num_guesses)
        info = {
            "matched_secret": matched_secret.astype(jnp.float32),
            "guess_count": state.guess_count.astype(jnp.float32),
            "illegal_guesses": (~legal_guesses).astype(jnp.float32),
        }
        timestep = Timestep(
            obs=get_obs(state),
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=jnp.zeros_like(terminated),
            info=info,
        )
        return state, timestep

    def observation_space(self) -> Space:
        return Space(shape=(WORD_LENGTH * 3,), dtype=jnp.float32, low=0.0, high=1.0)

    def action_space(self) -> Space:
        return Space(
            shape=(WORD_LENGTH,),
            dtype=jnp.int32,
            low=0,
            high=NUM_LETTERS - 1,
        )

    def time_limit(self) -> int:
        return self.num_guesses

    def render(self, state: EnvState, cell: int = 32, pad: int = 2) -> Array:
        palette = jnp.array(
            [[58, 58, 60], [181, 159, 59], [83, 141, 78]], dtype=jnp.uint8
        )
        colors = jnp.where(
            state.guess_revealed, palette[state.guess_feedback], palette[0]
        )
        border = jnp.arange(cell)
        inside = (border >= pad) & (border < cell - pad)
        tile = inside[:, None] & inside[None, :]
        cells = jnp.where(tile[None, :, :, None], colors[:, None, None, :], 0)
        return cells.transpose(1, 0, 2, 3).reshape(cell, WORD_LENGTH * cell, 3)


def load_words(path: Path) -> list[str]:
    return sorted(
        {
            word
            for line in path.read_text().splitlines()
            if (word := line.strip().lower()).isascii()
            and word.isalpha()
            and len(word) == WORD_LENGTH
        }
    )


def encode_words(words: list[str]) -> np.ndarray:
    letters = np.array(
        [[ord(c) - ord("a") for c in word] for word in words], dtype=np.int64
    )
    weights = (NUM_LETTERS ** np.arange(WORD_LENGTH)).astype(np.int64)
    return np.unique((letters * weights).sum(axis=1)).astype(np.int32)


def make(
    env_id,
    answer_pool_path=None,
    guess_pool_path=None,
    num_guesses=6,
    invalid_penalty=INVALID_PENALTY,
    yellow_reward=YELLOW_REWARD,
    green_reward=GREEN_REWARD,
    **kwargs,
):
    data = Path(__file__).parent
    answer_pool_path = answer_pool_path or data / "answer_pool.txt"
    guess_pool_path = guess_pool_path or data / "guess_pool.txt"

    answer_pool = load_words(answer_pool_path)
    guess_words = load_words(guess_pool_path)

    missing_words = set(answer_pool) - set(guess_words)
    assert not missing_words, (
        f"guess pool must be a superset of the answer pool; "
        f"{len(missing_words)} answer(s) missing (e.g. {sorted(missing_words)[:5]})"
    )

    letters = np.array(
        [[ord(c) - ord("a") for c in answer] for answer in answer_pool], dtype=np.int32
    )
    guess_pool = encode_words(guess_words)

    env = Wordle(
        answer_pool=letters,
        guess_pool=guess_pool,
        num_guesses=num_guesses,
        invalid_penalty=invalid_penalty,
        yellow_reward=yellow_reward,
        green_reward=green_reward,
    )
    return env
