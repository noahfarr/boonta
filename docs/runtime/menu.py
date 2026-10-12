PODRACERS = [
    {
        "id": "anakin",
        "name": "Anakin",
        "blurb": "Rollout and update compile into one on-policy program that runs in one worker.",
    },
    {
        "id": "sebulba",
        "name": "Sebulba",
        "blurb": "Actor and learner in two workers. Parameters cross by message with a fixed one-update lag, as in Cleanba.",
    },
    {
        "id": "quadinaros",
        "name": "Quadinaros",
        "blurb": "Learns offline from a bundled dataset and never steps the environment while training.",
    },
]

PILOTS = [
    {"id": "ppo", "name": "PPO", "blurb": "Proximal policy optimization clips the policy gradient and learns a value baseline."},
    {"id": "pqn", "name": "PQN", "blurb": "Parallel Q-network learns Q(λ) on fresh rollouts with LayerNorm, without a replay buffer or a target network."},
    {"id": "dqn", "name": "DQN", "blurb": "Deep Q-network acts ε-greedily and learns from a replay buffer against a Polyak-averaged target network."},
    {"id": "bc", "name": "BC", "blurb": "Behavior cloning maximizes the likelihood of a trained policy's actions."},
    {"id": "iql", "name": "IQL", "blurb": "Implicit Q-learning regresses values to an expectile and clones actions weighted by their advantage."},
]

TRACKS = [
    {"id": "cartpole", "name": "CartPole", "blurb": "Balance a pole on a cart for 500 steps. Gymnax CartPole-v1.", "frame": "cartpole", "fps": 50, "range": [0, 500]},
    {"id": "breakout", "name": "Breakout", "blurb": "MinAtar Breakout is a 10×10 brick breaker with four object channels.", "frame": "grid", "fps": 15},
    {"id": "acrobot", "name": "Acrobot", "blurb": "Swing a two-link pendulum's tip above the bar. Gymnax Acrobot-v1, −500 to 0.", "frame": "acrobot", "fps": 30, "range": [-500, 0]},
    {"id": "kinetix", "name": "Kinetix", "blurb": "Three small Kinetix physics levels, where the green shape has to reach the blue goal without touching red. Offline only in the browser.", "frame": "pixels", "fps": 30},
]

HYPERPARAMETERS = {
    ("ppo", "cartpole"): {"num_envs": 128, "num_steps": 32, "num_minibatches": 8, "update_epochs": 4,
                          "learning_rate": 1e-3, "anneal": 1_500_000, "width": 64, "entropy": 0.0},
    ("ppo", "breakout"): {"num_envs": 64, "num_steps": 32, "num_minibatches": 4, "update_epochs": 4,
                          "learning_rate": 5e-4, "width": 64, "entropy": 0.01},
    ("ppo", "acrobot"): {"num_envs": 64, "num_steps": 32, "num_minibatches": 4, "update_epochs": 4,
                         "learning_rate": 1e-3, "anneal": 600_000, "width": 64, "entropy": 0.01,
                         "activation": "tanh"},
    ("pqn", "cartpole"): {"num_envs": 32, "num_steps": 32, "num_minibatches": 4, "update_epochs": 4,
                          "learning_rate": 1e-3, "width": 64, "explore": 100_000, "q_lambda": 0.9},
    ("pqn", "breakout"): {"num_envs": 32, "num_steps": 32, "num_minibatches": 4, "update_epochs": 4,
                          "learning_rate": 1e-3, "width": 64, "explore": 100_000, "q_lambda": 0.9},
    ("pqn", "acrobot"): {"num_envs": 32, "num_steps": 32, "num_minibatches": 4, "update_epochs": 4,
                         "learning_rate": 1e-3, "width": 64, "explore": 100_000, "q_lambda": 0.9},
    ("dqn", "cartpole"): {"num_envs": 16, "num_steps": 1, "batch_size": 64, "capacity": 20_000,
                          "updates_per_step": 1, "tau": 0.01, "learning_rate": 5e-4, "width": 64,
                          "explore": 40_000, "warmup": 1_000},
    ("dqn", "breakout"): {"num_envs": 16, "num_steps": 8, "batch_size": 64, "capacity": 2_500,
                          "updates_per_step": 8, "tau": 0.01, "learning_rate": 3e-4, "width": 128,
                          "explore": 100_000, "warmup": 1_000},
    ("ppo", "kinetix"): {"num_envs": 256, "num_steps": 64, "num_minibatches": 8, "update_epochs": 4,
                         "learning_rate": 3e-4, "width": 128, "entropy": 0.01},
    ("dqn", "acrobot"): {"num_envs": 16, "num_steps": 1, "batch_size": 64, "capacity": 20_000,
                         "updates_per_step": 1, "tau": 0.01, "learning_rate": 5e-4, "width": 64,
                         "explore": 40_000, "warmup": 1_000},
}

THRESHOLDS = {
    "cartpole": {"baseline": 22, "learned": 150, "solved": 475},
    "breakout": {"baseline": 0.5, "learned": 3, "solved": 5},
    "acrobot": {"baseline": -480, "learned": -200, "solved": -100},
    "kinetix": {"baseline": 0.0, "learned": 0.5, "solved": 0.8},
}

SEBULBA_STEPS = {"dqn": 8}


def online(podracer, pilot, track):
    h = dict(HYPERPARAMETERS[pilot, track])
    if podracer == "sebulba" and pilot in SEBULBA_STEPS:
        scale = SEBULBA_STEPS[pilot] // h["num_steps"]
        h.update(num_steps=SEBULBA_STEPS[pilot], updates_per_step=h["updates_per_step"] * scale)
    calls = 1 if podracer == "sebulba" else 100 // h["num_steps"] if pilot == "dqn" else 2
    return {
        "podracer": podracer, "pilot": pilot, "track": track, "h": h,
        "updates_per_call": calls, **THRESHOLDS[track],
    }


COMBOS = [
    online(podracer, pilot, track)
    for podracer in ("anakin", "sebulba")
    for pilot in ("ppo", "pqn", "dqn")
    for track in ("cartpole", "breakout", "acrobot")
]

EVALUATION_STEPS = {"cartpole": 500, "breakout": 1000, "acrobot": 500, "kinetix": 256}

OFFLINE = {
    "bc": {"num_envs": 16, "batch_size": 64, "learning_rate": 3e-4, "width": 64},
    "iql": {"num_envs": 16, "batch_size": 64, "learning_rate": 3e-4, "width": 64},
}


OFFLINE_SOLVED = {"cartpole": {"solved": 475}, "breakout": {"solved": 15}}

OFFLINE_OVERRIDES = {("iql", "breakout"): {"learning_rate": 1e-3}}


def offline(pilot, track):
    return {
        "podracer": "quadinaros", "pilot": pilot, "track": track,
        "h": {**OFFLINE[pilot], **OFFLINE_OVERRIDES.get((pilot, track), {})},
        "updates_per_call": 25, "evaluation_steps": EVALUATION_STEPS[track],
        "evaluation_every": 1500, **THRESHOLDS[track], **OFFLINE_SOLVED.get(track, {}),
        **({"loss": "actor/loss"} if pilot == "bc" else {}),
    }


COMBOS += [offline(pilot, track) for pilot in ("bc", "iql") for track in ("cartpole", "breakout", "acrobot", "kinetix")]

DATASETS = {
    "cartpole": {"pilot": "ppo", "train_steps": 1_500_000, "size": 20_000},
    "breakout": {"pilot": "pqn", "train_steps": 3_000_000, "size": 20_000},
    "acrobot": {"pilot": "pqn", "train_steps": 1_000_000, "size": 20_000},
    "kinetix": {"pilot": "ppo", "train_steps": 6_000_000, "size": 20_000},
}
