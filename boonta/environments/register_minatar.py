"""Registers the official MinAtar games with Gymnasium.

MinAtar registers them through a plugin entry point, which Gymnasium stopped loading in
1.0. An environment id of the form "boonta.environments.register_minatar:MinAtar/Breakout-v0"
makes Gymnasium import this module first.
"""

from minatar.gym import register_envs

register_envs()
