import jax
import jax.numpy as jnp

from boonta.environments.kaggriculture import PLAYERS
from boonta.environments.kaggriculture import actions as A

SEATS = PLAYERS
NEG = jnp.finfo(jnp.float32).min


def fold_seats(x):
    return x.reshape(-1, *x.shape[2:])


def unfold_seats(x):
    return x.reshape(-1, SEATS, *x.shape[1:])


def categorical_log_prob(logits, index, keep=None):
    log_p = jax.nn.log_softmax(logits, axis=-1)
    picked = jnp.take_along_axis(log_p, index[..., None], axis=-1)[..., 0]
    if keep is not None:
        picked = jnp.where(keep, picked, 0.0)
    return picked.sum(-1) if picked.ndim > 1 else picked


def categorical_entropy(logits):
    log_p = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(jnp.exp(log_p) * log_p, axis=-1)


class JobDistribution:
    def __init__(self, logits, temperature=1.0):
        self.steps = logits["live"].shape[1] if logits["live"].ndim > 1 else None
        if self.steps is not None:
            logits = jax.tree.map(lambda leaf: leaf.reshape(-1, *leaf.shape[2:]), logits)
        self.logits = logits
        self.temperature = temperature

    def fold(self, x):
        if self.steps is None:
            return fold_seats(x)
        envs, steps, seats = x.shape[:3]
        return jnp.swapaxes(x, 1, 2).reshape(envs * seats * steps, *x.shape[3:])

    def unfold(self, x):
        if self.steps is None:
            return unfold_seats(x)
        rows = x.shape[0] // self.steps
        x = x.reshape(rows // SEATS, SEATS, self.steps, *x.shape[1:])
        return jnp.swapaxes(x, 1, 2)

    def cool(self, leaf):
        if isinstance(self.temperature, (int, float)) and self.temperature == 1:
            return leaf
        scale = jnp.where(self.temperature > 0, self.temperature, 1.0)
        return jnp.where(leaf > NEG, leaf / scale, NEG)

    def quantity_logits(self, market, table=None):
        table = self.logits["quantity"] if table is None else table
        picked = jnp.take_along_axis(table[:, None], market[:, :, None, None], axis=2)
        return picked[:, :, 0, :]

    def draw(self, seed):
        key_jobs, key_market, key_qty = jax.random.split(seed, 3)
        chosen, job_lp, _ = A.sample_jobs(
            self.cool(self.logits["jobs_masked"]), self.logits["live"], key_jobs
        )
        market_logits = self.cool(self.logits["market"])
        market = jax.random.categorical(key_market, market_logits, axis=-1)
        quantity = self.quantity_logits(market, self.cool(self.logits["quantity"]))
        qty = jax.random.categorical(key_qty, quantity, axis=-1)
        action = {
            "chosen": self.unfold(chosen),
            "market": self.unfold(market),
            "qty": self.unfold(qty),
            "travel": self.unfold(self.logits["travel"]),
        }
        log_prob = (
            job_lp
            + categorical_log_prob(market_logits, market)
            + categorical_log_prob(quantity, qty)
        )
        return action, self.unfold(log_prob)

    def sample_and_log_prob(self, seed):
        if isinstance(self.temperature, (int, float)):
            if self.temperature > 0:
                return self.draw(seed)
            action = self.mode()
            return action, self.log_prob(action)
        drawn, log_prob = self.draw(seed)
        mode = self.mode()
        greedy = self.temperature <= 0
        action = jax.tree.map(lambda best, sampled: jnp.where(greedy, best, sampled), mode, drawn)
        return action, jnp.where(greedy, self.log_prob(mode), log_prob)

    def sample(self, seed):
        action, _ = self.sample_and_log_prob(seed)
        return action

    def parts(self, action):
        live = self.logits["live"]
        if "filled" in action:
            live = self.fold(action["filled"])
        allowed = self.fold(action["allowed"]) if "allowed" in action else None
        job_lp, _ = A.score_jobs(self.logits["jobs_masked"], live, self.fold(action["chosen"]))
        market = self.fold(action["market"])
        return {
            "jobs": self.unfold(job_lp),
            "market": self.unfold(categorical_log_prob(self.logits["market"], market, allowed)),
            "qty": self.unfold(
                categorical_log_prob(
                    self.quantity_logits(market), self.fold(action["qty"]), allowed
                )
            ),
        }

    def log_prob(self, action):
        parts = self.parts(action)
        return parts["jobs"] + parts["market"] + parts["qty"]

    def mode(self):
        chosen = A.greedy_jobs(self.logits["jobs_masked"], self.logits["live"])
        market = jnp.argmax(self.logits["market"], axis=-1)
        qty = jnp.argmax(self.quantity_logits(market), axis=-1)
        return {
            "chosen": self.unfold(chosen),
            "market": self.unfold(market),
            "qty": self.unfold(qty),
            "travel": self.unfold(self.logits["travel"]),
        }

    def entropy(self):
        rows = self.logits["jobs_masked"].shape[0]
        blank = jnp.zeros((rows, A.UNITS), jnp.int32)
        _, job_entropy = A.score_jobs(self.logits["jobs_masked"], self.logits["live"], blank)
        share = jax.nn.softmax(self.logits["market"], axis=-1)
        conditional = categorical_entropy(self.logits["quantity"])
        return self.unfold(
            job_entropy
            + categorical_entropy(self.logits["market"]).sum(-1)
            + jnp.sum(share * conditional[:, None, :], axis=-1).sum(-1)
        )
