from ..sections import Environment
from . import environment

for scenario in (
    "10m_vs_11m",
    "25m",
    "27m_vs_30m",
    "2s3z",
    "3m",
    "3s5z_vs_3s6z",
    "3s5z",
    "3s_vs_5z",
    "5m_vs_6m",
    "6h_vs_8z",
    "8m",
    "smacv2_10_units",
    "smacv2_20_units",
    "smacv2_5_units",
):
    environment(
        Environment(
            namespace="jaxmarl", suite="smax", env_id="HeuristicEnemySMAX", kwargs=dict(scenario=scenario)
        ),
        name=f"jaxmarl/smax/{scenario}",
    )
