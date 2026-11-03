import flax.linen as nn
import jax
import optax
from hydra.utils import instantiate

from boonta import datasets
from boonta.algorithms.recurrent_bc import RecurrentBC
from boonta.networks import Categorical, Network, Unembed

from .recurrent_bc_agent_trove import TokenInitialization

END_OF_TEXT_ID = 50256
EMBEDDING_STD = 0.02


class TiedTokenEmbedding(nn.Module):
    embed: nn.Embed

    @nn.compact
    def __call__(self, obs, action=None, reward=None, done=None):
        return self.embed(obs)


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
    updates = int(cfg.num_steps) // batch_positions
    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg.optimizer.lr,
        warmup_steps=cfg.optimizer.warmup_updates,
        decay_steps=updates,
        end_value=cfg.optimizer.final_lr,
    )
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
