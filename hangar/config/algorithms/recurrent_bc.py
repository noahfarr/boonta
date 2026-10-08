from boonta.algorithms.recurrent_bc import RecurrentBCConfig

from ..sections import Optimizer
from ..store import fbuilds
from . import algorithm, recurrent

recurrent_bc = algorithm(
    dict(
        defaults=[*recurrent, {"override /podracer": "quadinaros"}],
        algorithm=fbuilds(RecurrentBCConfig, batch_size=32, entropy_coefficient=0.0),
        optimizer=Optimizer(lr=3e-4, max_grad_norm=0.5),
    ),
    name="recurrent_bc",
)

recurrent_bc.hyperparameters(
    "kinetix",
    defaults=[{"/dataset": "kinetix/offline_m"}],
    algorithm=dict(batch_size="${dataset.kwargs.batch_size}"),
    network=dict(features=128, num_layers=2, num_heads=8, actor_depth=5, actor_width=128),
    podracer=dict(config=dict(batch_shape=["${dataset.kwargs.batch_size}", 256])),
)
