import flax.linen as nn

from boonta.networks import (RNN, SSM, Highway, Identity, MinGRUCell, RTUCell,
                             SelfAttention, Tower, causal_attention_mask, llama,
                             repeat)

from .store import builds, store

cell = store(group="cell", package="_global_")
torso = store(group="torso")
stack = store(group="stack", package="_global_")

cell(
    dict(defaults=[{"override /torso": "rnn"}], cell=builds(nn.GRUCell, features=128, dtype=None)),
    name="gru",
)
cell(
    dict(defaults=[{"override /torso": "ssm"}], cell=builds(MinGRUCell, features=128, dtype=None)),
    name="min_gru",
)
cell(
    dict(defaults=[{"override /torso": "rnn"}], cell=builds(RTUCell, features=128, dtype=None)),
    name="rtu",
)
cell(
    dict(
        defaults=[{"override /torso": "default"}],
        cell=builds(
            SelfAttention,
            features=128,
            num_heads=4,
            context_length="${rollout.num_steps}",
            attention_mask=builds(causal_attention_mask, zen_partial=True),
            dtype=None,
        ),
    ),
    name="self_attention",
)

torso(dict(torso="${cell}"), name="default", package="_global_")
torso(builds(Identity), name="identity")
torso(builds(RNN, cell="${cell}"), name="rnn")
torso(builds(SSM, cell="${cell}"), name="ssm")

stack(dict(stack="${torso}"), name="default")
stack(
    dict(
        num_layers=1,
        stack=builds(
            Tower,
            block=builds(Highway, blocks="${torso}", dtype="${oc.select:network.dtype,null}"),
            num_layers="${num_layers}",
        ),
    ),
    name="highway",
)
stack(
    dict(
        num_layers=2,
        stack=builds(
            llama,
            block="${torso}",
            num_layers="${num_layers}",
            dtype="${oc.select:network.dtype,null}",
            features="${cell.features}",
            expansion_factor=4,
        ),
    ),
    name="llama",
)
stack(
    dict(num_layers=1, stack=builds(repeat, block="${torso}", num_layers="${num_layers}")),
    name="repeat",
)
