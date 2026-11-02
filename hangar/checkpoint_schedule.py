"""Exact, step-aligned checkpoint cuts for Quadinaros; the optimizer is unchanged."""
import jax
import jax.numpy as jnp


def checkpoint_steps(budget, batch_size, fractions):
    if budget % batch_size:
        raise ValueError('Allocated budget must be divisible by the global batch positions')
    updates=budget//batch_size
    if not fractions or any(not 0 <= f <= 1 for f in fractions):
        raise ValueError('Checkpoint fractions must be in [0,1]')
    return sorted({round(f*updates)*batch_size for f in fractions})


def offline_epoch(podracer,state,key,num_updates,stops,callback,continuing=False):
    """Stage one pool, consume it across cuts, and keep global PRNG indexing."""
    if not getattr(podracer.config,'indexed_keys',False):
        raise ValueError('Scheduled offline checkpoints require indexed_keys=true')
    start=int(state.algorithm_state.step)
    finish=start+num_updates*podracer.batch_size
    cuts=[s for s in stops if start<s<=finish]
    if not cuts or cuts[-1]!=finish: cuts.append(finish)
    logs=[]
    first=True
    for stop in cuts:
        count=(stop-int(state.algorithm_state.step))//podracer.batch_size
        if first and not continuing:
            state,log=podracer.train(state,key,count)
        else:
            state,log=podracer.fit(state,key,count)
        logs.append(log)
        first=False
        if stop in stops:
            callback(state,log,stop)
    merged=jax.tree.map(lambda *xs:jnp.concatenate(xs,axis=0),*logs)
    return state,merged
