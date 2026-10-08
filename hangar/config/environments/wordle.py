from hydra_zen import make_config

from ..sections import Environment
from . import environment

environment(
    make_config(
        hydra_defaults=["/artisan/transcript", "_self_"],
        namespace="wordle",
        suite="wordle",
        env_id="Wordle-v0",
        bases=(Environment,),
    ),
    name="wordle/wordle",
)
