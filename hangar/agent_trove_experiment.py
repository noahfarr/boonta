"""Shared planning and read-only inspection of native Boonta experiment runs."""
import hashlib
import json
from pathlib import Path


def compose_experiment(cache,tokenizer,split,objective,stage='pilot',seed=42):
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from hangar import resolvers  # noqa: F401
    preset='agent_trove_pilot' if stage=='pilot' else 'agent_trove_192m'
    overrides=['algorithm=recurrent_bc','environment=agent_trove',
        f'hyperparameters=recurrent_bc/{preset}','scoring=final','algorithm.batch_size=1','dataset.kwargs.num_devices=1','podracer.config.mesh.count=1',
        f'seed={seed}',f'dataset.kwargs.cache_dir={cache}',f'dataset.kwargs.tokenizer_id={tokenizer}',
        f'dataset.kwargs.split_manifest={split}',f'dataset.kwargs.assistant_only={str(objective=="blm").lower()}']
    with initialize_config_dir(config_dir=str(Path(__file__).parent/'config'),version_base=None):
        cfg=compose('train',overrides=overrides,return_hydra_config=True)
        HydraConfig.instance().set_config(cfg)
    return cfg,overrides


def plan(cache,tokenizer,split,objective,stage,seed):
    import jax
    from hangar.checkpoint_schedule import checkpoint_steps
    cache=Path(cache); tokenizer=Path(tokenizer); split=Path(split)
    metadata=json.loads((cache/'manifest.json').read_text()); grouping=json.loads(split.read_text())
    if not metadata.get('complete'): raise ValueError('Complete preprocessing cache required')
    if hashlib.sha256(tokenizer.read_bytes()).hexdigest()!=metadata['tokenizer_sha256']:
        raise ValueError('Tokenizer file checksum mismatch')
    cfg,overrides=compose_experiment(cache,tokenizer,split,objective,stage,seed)
    context=int(cfg.cell.context_length); budget=int(cfg.num_steps)
    cohort=grouping['budgets'][str(context)]
    if budget//context>cohort['train']['trajectories']:
        raise ValueError('Grouped train split is too small for the requested without-replacement budget')
    return cfg,dict(stage=stage,objective=objective,seed=seed,model_parameters=191772672,
        context=context,allocated_positions=budget,trajectory_draws=budget//context,
        pool_size=int(cfg.dataset.kwargs.pool_size),refreshes=int(cfg.num_epochs),global_batch=1,
        checkpoint_positions=checkpoint_steps(budget,context,cfg.checkpoint_fractions),
        nominal_checkpoint_fractions=list(cfg.checkpoint_fractions),
        actual_checkpoint_fractions=[x/budget for x in checkpoint_steps(budget,context,cfg.checkpoint_fractions)],
        grouped_train=cohort['train'],grouped_validation=cohort['validation'],
        task_disjoint_certified=bool(grouping['task_disjoint_certified']),
        evaluation_scope='task/repository-disjoint' if grouping['task_disjoint_certified'] else 'ENGINEERING ONLY: automatic grouping is not a task/repository certificate',
        cache_revision=metadata['revision'],tokenizer_sha256=metadata['tokenizer_sha256'],
        split_manifest_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
        requires_x64=budget>2**31-1,x64_enabled=bool(jax.config.x64_enabled),
        overrides=overrides)


def read_preflight(directory):
    """Read a normal train.py run; no model initialization or optimizer calls."""
    import numpy as np
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(directory/'.hydra'/'config.yaml')
    context=int(cfg.cell.context_length)
    width=int(cfg.cell.features)
    head_dim=int(cfg.cell.head_dim)
    parameters=(2*int(cfg.network.vocab_size)*width + int(cfg.num_layers)*(
        2*width*head_dim*(int(cfg.cell.num_heads)+int(cfg.cell.num_groups))
        + 3*width*int(cfg.stack.hidden_dim) + 2*width + 2*head_dim) + width)
    with np.load(directory/'metrics.npz') as metrics:
        steps=metrics['steps'].tolist()
        value=lambda name: float(np.asarray(metrics[name][-1]).mean())
        loss=value('actor/loss')
        throughput=value('monitor/SPS')
        peak=value('gpu/peak_memory_consumption')
    split=Path(cfg.dataset.kwargs.split_manifest)
    tokenizer=Path(cfg.dataset.kwargs.tokenizer_id)
    passed=(steps==[0,context,2*context] and np.isfinite(loss) and np.isfinite(throughput) and throughput>0)
    result=dict(passed=bool(passed),objective='blm' if cfg.dataset.kwargs.assistant_only else 'clm',
        context=context,model_parameters=parameters,seed=int(cfg.seed),not_a_pilot_run=True,
        executed_allocated_positions=steps[-1],loss=loss,allocated_tokens_per_second=throughput,
        peak_allocator_memory_gib=peak if np.isfinite(peak) else None,
        tokenizer_sha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
        split_manifest_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
        measurement_scope='Normal train.py epoch timing, including host staging and logging; epoch 1 includes compilation. Peak is the JAX allocator high-water mark, not total device-wide NVML usage.')
    return result


