from . import algorithm
from .ippo import ippo

mappo = algorithm(ippo.node, name="mappo")

mappo.hyperparameters("jaxmarl", total_timesteps=20_000_000)
