import jax


def rollout(algorithm, environment, algorithm_state, key, num_steps, record):
    init_key, key = jax.random.split(key)
    environment_state, timestep = environment.init(init_key)

    def step(carry, key):
        algorithm_state, environment_state, timestep = carry
        algorithm_key, environment_key = jax.random.split(key)
        algorithm_state, action, _ = algorithm.step(
            algorithm_state, algorithm_key, timestep, temperature=0.0
        )
        environment_state, timestep = environment.step(
            environment_key, environment_state, action
        )
        return (algorithm_state, environment_state, timestep), record(
            environment_state, action
        )

    (_, environment_state, _), recorded = jax.lax.scan(
        step,
        (algorithm_state, environment_state, timestep),
        jax.random.split(key, num_steps),
    )
    environment.close(environment_state)
    return recorded
