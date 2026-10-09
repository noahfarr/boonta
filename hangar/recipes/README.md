# recipes

A recipe turns a config into the parts of a run. It is the only place where the packages meet. Algorithms never import environments, environments never import networks, and podracers only call the methods of what they are given. A recipe builds an environment, a network, an optimizer and an algorithm, and hands them to a podracer.

There is one file per algorithm and suite, named `<algorithm>_<suite>.py`, such as [`ppo_minatar.py`](ppo_minatar.py). Each file is plain and readable on its own. There is no shared helper layer.

## How a recipe is picked

[`__init__.py`](__init__.py) holds `register`, a dict from `(algorithm, namespace, suite)` to a recipe's `make`:

```python
("ppo", "gymnax", "minatar"): ppo_minatar.make,
```

- `algorithm` is the Hydra group choice, the name you pass as `algorithm=ppo`. It is read from `HydraConfig`, not from the config body.
- `namespace` and `suite` come from the environment config. `suite` defaults to `namespace`.

`recipes.make(cfg)` looks up the recipe, calls it, and passes what it returns to the podracer:

```python
def make(cfg):
    namespace = cfg.environment.namespace
    suite = cfg.environment.get("suite", namespace)
    name = HydraConfig.get().runtime.choices["algorithm"]
    components = register[(name, namespace, suite)](cfg)
    return instantiate(cfg.podracer)(**components)
```

## What make(cfg) returns

A dict that is splatted into the podracer's `make`. `algorithm`, `environment`, `pit` and `lap` are always there; `pit` and `lap` come from the curriculum. Any other key becomes a podracer argument, such as `dataset` for `quadinaros`. See [podracers](../../boonta/podracers/README.md).

## A recipe, step by step

[`ppo_minatar.py`](ppo_minatar.py) is a whole recipe in 50 lines.

1. Build the environment from its config, add the auto-reset and vectorize it. The action count is read before `Vectorize`, which adds a batch axis.

   ```python
   env = environments.make(**cfg.environment)
   env = SameStepAutoReset(env)

   num_actions = env.action_space().num_actions

   env = Vectorize(env, num_envs=cfg.environment.num_envs)
   ```

2. Build the network. The head is the policy, here a `Categorical` over the actions next to a value head.

   ```python
   network = Network(
       feature_extractor=FeatureExtractor(
           observation_extractor=nn.Sequential(
               [
                   lambda obs: obs.astype(jnp.float32),
                   nn.Conv(16, (3, 3), padding="VALID"),
                   nn.relu,
                   Flatten(start_dim=-3),
                   nn.Dense(128),
                   nn.relu,
               ]
           ),
       ),
       head=ActorCritic(
           actor=Categorical(nn.Dense(num_actions)),
           critic=nn.Dense(1),
       ),
   )
   ```

3. Build the algorithm. `instantiate(cfg.algorithm)` makes the config dataclass named in `hangar/config/algorithm/ppo.yaml`. The optimizer is built here from `cfg.optimizer`.

   ```python
   algorithm = PPO(
       cfg=instantiate(cfg.algorithm),
       network=network,
       optimizer=optax.chain(
           optax.clip_by_global_norm(cfg.optimizer.max_grad_norm),
           optax.adam(cfg.optimizer.lr),
       ),
   )
   ```

4. Apply the curriculum, then the wrappers that record or reshape what the agent experiences. The curriculum always gets a vectorized, auto-resetting environment and only wraps the outside, so the statistics see the levels it chose.

   ```python
   algorithm, env, pit, lap = instantiate(cfg.curriculum)(algorithm, env)
   env = RecordEpisodeStatistics(env, gamma=cfg.algorithm.gamma)
   return {"algorithm": algorithm, "environment": env, "pit": pit, "lap": lap}
   ```

A learning rate that anneals comes from `learning_rate(cfg, batch_size)` in [`schedules.py`](schedules.py). Recipes do not build their own cosine schedule; a test checks this.

## Add a recipe

1. Copy the closest recipe to `<algorithm>_<suite>.py` and change what differs.
2. Import it in [`__init__.py`](__init__.py) and add its `(algorithm, namespace, suite)` key to `register`. One recipe can serve several suites, as `ppo_isaac_lab.make` does.
3. Add a smoke case to `CASES` in [`tests/test_recipes.py`](../../tests/test_recipes.py):

   ```python
   ("ppo", "gymnax/minatar/breakout", None, False),
   ```

   The third field is a skip reason, such as `installed("brax")`, and the fourth builds a dataset for offline recipes. `test_every_recipe_builds_and_runs_one_update` builds the recipe with 32 environments and runs one update. `test_every_recipe_has_a_smoke_case` fails if a registered recipe has no case.

## Hyperparameters

`hangar/config/hyperparameters/<algorithm>/<environment path>.yaml` is applied on top of the algorithm, environment and curriculum configs. Each file starts with `# @package _global_`, so it can set any key. The `cascade` resolver in [`resolvers.py`](../resolvers.py) walks up the environment path until it finds a file, and falls back to `hyperparameters/<algorithm>.yaml`. With `environment=gymnax/minatar/breakout` it tries

1. `hyperparameters/ppo/gymnax/minatar/breakout.yaml`
2. `hyperparameters/ppo/gymnax/minatar.yaml`
3. `hyperparameters/ppo/gymnax.yaml`
4. `hyperparameters/ppo.yaml`

and uses the first that exists. Here it is the second, which sets `total_timesteps: 20_000_000` for every MinAtar game. A missing file is not an error.

Before each environment file it tries the same path with the curriculum appended, so `curriculum=plr` with `environment=kinetix/kinetix` finds `hyperparameters/ppo/kinetix/plr.yaml`. That file pulls in `hyperparameters/ppo/kinetix.yaml` through its defaults and sets what PLR needs on Kinetix, the level `sample` factory. A recipe never passes curriculum arguments: it calls `instantiate(cfg.curriculum)(algorithm, env)`, and Hydra builds the factories when the curriculum is instantiated. The default curriculum has no such files, so it resolves exactly as before.

Run the recipe with

```bash
uv run boonta algorithm=ppo environment=gymnax/minatar/breakout
```
