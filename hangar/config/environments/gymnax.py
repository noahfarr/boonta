from ..sections import Environment, Minatar
from . import environment

deep_sea = Environment(namespace="gymnax", suite="bsuite", env_id="DeepSea-bsuite", kwargs=dict(size=20))
environment(deep_sea, name="gymnax/bsuite")
environment(deep_sea, name="gymnax/bsuite/deep_sea")
environment(
    Environment(namespace="gymnax", suite="classic_control", env_id="CartPole-v1"),
    name="gymnax/classic_control/cartpole",
)
for name, env_id in (
    ("asterix", "Asterix-MinAtar"),
    ("breakout", "Breakout-MinAtar"),
    ("freeway", "Freeway-MinAtar"),
    ("seaquest", "Seaquest-MinAtar"),
    ("space_invaders", "SpaceInvaders-MinAtar"),
):
    environment(Minatar(namespace="gymnax", suite="minatar", env_id=env_id), name=f"gymnax/minatar/{name}")
