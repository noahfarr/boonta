import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate

from boonta import datasets
from boonta.algorithms.recurrent_bc import RecurrentBC
from boonta.networks import Categorical, Network, Unembed
from boonta.utils import Timestep

from . import schedules


END_OF_TEXT_ID = 50256
EMBEDDING_STD = 0.02


class TiedTokenEmbedding(nn.Module):
    embed: nn.Embed

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None):
        return self.embed(obs)



class TokenInitialization:
    num_agents = 1

    def __init__(self, num_envs, pad_id):
        self.num_envs, self.pad_id = num_envs, pad_id

    def init(self, key):
        return None, Timestep(
            obs=jnp.full((self.num_envs,), self.pad_id, jnp.int32),
            action=jnp.zeros(self.num_envs, jnp.int32),
            reward=jnp.zeros(self.num_envs, jnp.float32),
            terminated=jnp.ones(self.num_envs, bool),
            truncated=jnp.zeros(self.num_envs, bool),
        )

    def step(self, key, state, action):
        raise NotImplementedError("pretraining data is offline")

    def close(self, state):
        pass

def make_network(vocab_size, features, torso, dtype):
    embed = nn.Embed(
        vocab_size,
        features,
        embedding_init=nn.initializers.normal(EMBEDDING_STD),
        dtype=dtype,
    )
    return Network(
        feature_extractor=TiedTokenEmbedding(embed=embed),
        torso=torso,
        head=Categorical(Unembed(embed=embed)),
    )


def decayed_parameters(params):
    return jax.tree_util.tree_map_with_path(
        lambda path, _: path[-1].key in ("kernel", "embedding"), params
    )


def make(cfg):
    dataset = datasets.make(**cfg.dataset)
    batch_positions = cfg.algorithm.batch_size * dataset.length
    learning_rate = schedules.learning_rate(cfg, batch_size=batch_positions)
    network = make_network(
        cfg.network.vocab_size, cfg.cell.features, instantiate(cfg.stack), cfg.network.dtype
    )
    algorithm = RecurrentBC(
        cfg=instantiate(cfg.algorithm),
        network=network,
        optimizer=optax.chain(
            optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
            optax.adamw(
                learning_rate,
                b1=cfg.optimizer.beta1,
                b2=cfg.optimizer.beta2,
                weight_decay=cfg.optimizer.weight_decay,
                mask=decayed_parameters,
            ),
        ),
    )
    environment = TokenInitialization(cfg.environment.num_envs, END_OF_TEXT_ID)
    algorithm, environment, pit, lap = instantiate(cfg.curriculum)(algorithm, environment)
    return {
        "algorithm": algorithm,
        "environment": environment,
        "dataset": dataset,
        "pit": pit,
        "lap": lap,
    }
