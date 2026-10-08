from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING


@dataclass
class Training:
    num_epochs: int = 10


@dataclass
class Evaluation:
    num_steps: int = 0


@dataclass
class Rollout:
    num_steps: int = MISSING


@dataclass
class Replay:
    capacity: int = MISSING


@dataclass
class Exploration:
    start: float = MISSING
    end: float = MISSING
    fraction: float = MISSING


@dataclass
class Optimizer:
    lr: float = MISSING
    max_grad_norm: Optional[float] = None
    anneal: bool = False
    min_lr_ratio: float = 0.0
    name: Optional[str] = None
    beta: Optional[float] = None
    weight_decay: Optional[float] = None
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8


@dataclass
class Optimizers:
    actor_lr: float = MISSING
    critic_lr: float = MISSING
    alpha_lr: Optional[float] = None
    value_lr: Optional[float] = None
    lagrangian_lr: Optional[float] = None


@dataclass
class Network:
    features: Optional[int] = None
    layers: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    actor_depth: Optional[int] = None
    actor_width: Optional[int] = None
    encoder: Optional[str] = None
    prior: bool = False
    repo_id: Optional[str] = None
    dtype: Optional[str] = None
    param_dtype: Optional[str] = None


@dataclass
class Environment:
    namespace: str = MISSING
    suite: str = MISSING
    env_id: Any = MISSING
    num_envs: int = MISSING
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Minatar(Environment):
    sticky_action_prob: float = 0.0


@dataclass
class Craftax(Environment):
    reset_ratio: int = MISSING


@dataclass
class Kinetix(Environment):
    holdout_levels: bool = False
    levels_from: Optional[str] = None


@dataclass
class Libero(Environment):
    horizon: int = MISSING


@dataclass
class Dataset:
    namespace: str = MISSING
    dataset_id: str = MISSING
    kwargs: dict[str, Any] = field(default_factory=dict)
