import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="sphere")
def sphere(cfg: DictConfig) -> dict:
    value = -((cfg.x - 0.3) ** 2) - (cfg.y - 0.7) ** 2
    return {"score": float(value), "cost": 1.0}


if __name__ == "__main__":
    sphere()
