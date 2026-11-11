from dataclasses import dataclass, fields

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


class NoisyJointPositionAction(JointPositionAction):
    cfg: "NoisyJointPositionActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.executed = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions):
        self.executed = actions + self.cfg.noise * torch.randn_like(actions)
        super().process_actions(self.executed)


@dataclass(kw_only=True)
class NoisyJointPositionActionCfg(JointPositionActionCfg):
    noise: float = 0.0

    @classmethod
    def derive(cls, base: JointPositionActionCfg, **fields_):
        return cls(**{field.name: getattr(base, field.name) for field in fields(base) if field.init}, **fields_)

    def build(self, env):
        return NoisyJointPositionAction(self, env)
