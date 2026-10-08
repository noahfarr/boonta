from . import algorithm, special
from .ippo import ippo

algorithm(ippo, name="mappo")

special("mappo", "jaxmarl", total_timesteps=20_000_000)
