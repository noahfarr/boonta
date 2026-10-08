from ..sections import Craftax
from . import environment

environment(
    Craftax(namespace="craftax", suite="craftax_classic", env_id="Craftax-Classic-Symbolic-v1", reset_ratio=64),
    name="craftax/craftax_classic/symbolic",
)
environment(
    Craftax(namespace="craftax", suite="craftax", env_id="Craftax-Symbolic-v1", reset_ratio=64),
    name="craftax/craftax/symbolic",
)
