import json
from types import SimpleNamespace

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

from boonta.datasets.agent_trove import AgentTrove, CachedRows
from boonta.networks import Network, Qwen3, Categorical
from boonta.networks.chunked_categorical import ChunkedCategorical
from boonta.networks.language_policy import LanguagePolicy, generate
from hangar.checkpoint_schedule import checkpoint_steps
from hangar.recipes.recurrent_bc_agent_trove import TokenEmbedding
from hangar.scripts.preprocess_agent_trove import record
from hangar.scripts.group_agent_trove import group_cache


@pytest.fixture
def token_cache(tmp_path):
    alphabet=sorted(pre_tokenizers.ByteLevel.alphabet())
    t=Tokenizer(models.BPE(vocab={v:i for i,v in enumerate(alphabet)},merges=[]))
    t.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)
    t.decoder=decoders.ByteLevel()
    t.add_special_tokens(['<|endoftext|>','<|im_start|>','<|im_end|>'])
    t.save(str(tmp_path/'tokenizer.json'))
    import hashlib
    offset=0;items=[]
    with (tmp_path/'tokens.bin').open('wb') as f,(tmp_path/'mask.bin').open('wb') as m:
        for i in range(32):
            row={'messages':[{'role':'user','content':f'find value {i}'},{'role':'assistant','content':'{"commands":[],"task_complete":true}'}],
                 'task':f'task-{i//2}'+('_copy1' if i%2 else ''),'original_source':'fixture'}
            tokens,mask,item=record(row,t)
            tokens.astype('<i4').tofile(f);mask.astype('u1').tofile(m)
            # Keep this fixture entirely in train to test order across pools.
            items.append({**item,'index':i,'offset':offset,'cached':True,'split':'train'})
            offset+=len(tokens)
    (tmp_path/'records.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in items))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(format_version=1,complete=True,cache_written=True,
        cached_tokens=offset,cache_max_length=32768,revision='fixture',rows=32,
        tokenizer_runtime_sha256=hashlib.sha256(t.to_str().encode()).hexdigest())))
    return tmp_path,t


def test_chunked_dense_values_gradients_and_parameter_identity():
    x=jax.random.normal(jax.random.key(1),(2,7,4));target=jnp.arange(14).reshape(2,7)%11
    heads=[Categorical(nn.Dense(11,use_bias=False)),ChunkedCategorical(nn.Dense(11,use_bias=False),3)]
    params=[head.init(jax.random.key(2),x,temperature=1.) for head in heads]
    for a,b in zip(jax.tree.leaves(params[0]),jax.tree.leaves(params[1])): np.testing.assert_array_equal(a,b)
    def loss(head,p):
        d=head.apply(p,x,temperature=1.)
        return -d.log_prob(target).mean()+.01*d.entropy().mean()
    results=[jax.value_and_grad(lambda p:loss(h,p))(p) for h,p in zip(heads,params)]
    np.testing.assert_allclose(results[0][0],results[1][0],rtol=1e-6)
    for a,b in zip(jax.tree.leaves(results[0][1]),jax.tree.leaves(results[1][1])):np.testing.assert_allclose(a,b,atol=1e-6)
    d=heads[1].apply(params[1],x,temperature=1.)
    np.testing.assert_allclose(d.kl_divergence(d),0,atol=1e-6)


def test_without_replacement_is_paired_and_resumable(token_cache):
    path,t=token_cache
    ids=[];saved=None
    for mode in (True,False):
        d=AgentTrove(cache_dir=path,tokenizer=t,length=128,pool_size=4,sampling='permutation',assistant_only=mode)
        state=d.init();sequence=[]
        for pool in range(4):
            state=d.update(state,jax.random.key(pool),None)
            for offset in (0,2):
                batch=jax.jit(lambda s,step:d.sample_at(s,jax.random.key(999),(2,128),step))(state,jnp.int32((pool*4+offset)*128))
                sequence.extend(np.asarray(batch.aux['record_index']).tolist())
            if pool==1: saved=serialization.to_bytes(state)
        ids.append(sequence)
        restored=serialization.from_bytes(d.init(),saved)
        next_state=d.update(restored,jax.random.key(9999),None)
        next_ids=d.sample_at(next_state,jax.random.key(0),(2,128),8*128).aux['record_index']
        np.testing.assert_array_equal(next_ids,sequence[8:10])
    assert ids[0]==ids[1] and len(set(ids[0]))==16


def test_grouped_split_links_copies_and_does_not_claim_repo_certification(token_cache,tmp_path):
    path,t=token_cache
    r=group_cache(path,tmp_path/'grouped',t,max_length=16384)
    split=np.fromfile(tmp_path/'grouped'/'splits.bin',dtype='u1')
    assert all(split[i]==split[i+1] for i in range(0,32,2))
    assert not r['task_disjoint_certified']
    assert r['unresolved']['repository_identity_unknown']==32
    rows=CachedRows(path,'train',tmp_path/'grouped'/'split.json')
    list(rows.sample_rows(128,1,42))
    assert len(rows.index)==r['budgets']['4096']['train']['trajectories']


def test_generation_cache_matches_teacher_forced_probability(token_cache):
    _,t=token_cache
    native=Network(feature_extractor=TokenEmbedding(t.get_vocab_size(),8,jnp.float32),
        torso=Qwen3(features=8,num_layers=1,num_heads=2,num_groups=1,head_dim=4,max_wavelength=1e6,
                    context_length=128,hidden_dim=16,dtype=jnp.float32),
        head=ChunkedCategorical(nn.Dense(t.get_vocab_size(),use_bias=False),16))
    policy=LanguagePolicy(native,t.token_to_id('<|endoftext|>'),t.token_to_id('<|im_end|>'),4)
    prefix=t.encode('<|im_start|>user\nfind<|im_end|>\n<|im_start|>assistant\n',add_special_tokens=False).ids
    array=np.full((1,1,128),policy.pad_id,np.int32);array[0,0,:len(prefix)]=prefix
    obs={'tokens':jnp.array(array),'length':jnp.array([[len(prefix)]],jnp.int32)}
    params=policy.init(jax.random.key(0),obs)
    _,dist=policy.apply(params,obs)
    actions,old_logp=dist.sample_and_log_prob(jax.random.key(1))
    np.testing.assert_allclose(dist.log_prob(actions),old_logp,atol=2e-5)


def test_checkpoint_targets_are_common_and_step_aligned():
    stops=checkpoint_steps(2**28,8192,[0,.1,.25,.5,.75,1])
    assert len(stops)==6 and stops[0]==0 and stops[-1]==2**28
    assert all(x%8192==0 for x in stops)
    assert abs(stops[1]/2**28-.1)<=1/32768


def test_terminal_objectives_real_outputs_and_group_isolation():
    from boonta.environments.terminal_tools import TerminalEpisode
    for seed in range(12):
        a=TerminalEpisode(seed,'train');b=TerminalEpisode(seed,'train')
        try:
            assert a.messages==b.messages and a.work!=b.work
            assert 'check_task.py' not in a.execute('ls')
            if a.kind=='find':
                marker=a.messages[-1]['content'].split('containing ')[1].split('.')[0]
                result=a.execute(f'grep -l {marker} *.txt')
                assert a.expected in result
                command=f"printf '%s\\n' '{a.expected}' > answer.txt"
            elif a.kind=='read':
                assert f'value={a.expected}' in a.execute('cat settings.cfg')
                command=f"printf '%s\\n' '{a.expected}' > answer.txt"
            else:command=f"printf '%s\\n' 'value={a.expected}' > settings.cfg"
            action=json.dumps({'commands':[{'keystrokes':command}]})
            reward,_,_,_=a.step(action)
            assert not b.goal()
            if a.kind=='modify':
                assert reward==0
                reward,_,_,_=a.step('{"commands":[{"keystrokes":"check"}]}')
            assert reward==1 and a.success
            for forbidden in ('cat /etc/passwd','cat ../check_task.py','python -c "print(1)"','rm settings.cfg',
                              "printf '%s\\n' hacked > ../outside"):
                with pytest.raises(ValueError):b.execute(forbidden)
        finally:a.close();b.close()


def test_terminal_callback_groups_have_separate_handles(token_cache):
    from boonta.environments.terminal_tools import TerminalTools
    _,t=token_cache
    env=TerminalTools(t,num_envs=2,group_size=2,context=2048,max_tokens=8,horizon=2)
    try:
        state,ts=jax.jit(env.init)(jax.random.key(1))
        np.testing.assert_array_equal(ts.obs['tokens'][0],ts.obs['tokens'][1])
        assert len(set(np.asarray(state.handles).tolist()))==2
        action=jnp.full((2,8),env.pad,jnp.int32).at[:,0].set(env.end)
        state,ts=jax.jit(env.step)(jax.random.key(2),state,action)
        assert np.asarray(ts.reward).sum()==0
        state,ts=jax.jit(env.step)(jax.random.key(3),state,action)
        assert np.asarray(ts.done).all()
        assert len(env.episodes)==2
    finally:env.close()


def test_native_grpo_composite_action_update_has_finite_nonzero_gradient(token_cache):
    import optax
    from boonta.algorithms.recurrent_grpo import RecurrentGRPO,RecurrentGRPOConfig
    from boonta.utils import Timestep,Transition
    _,t=token_cache
    native=Network(feature_extractor=TokenEmbedding(t.get_vocab_size(),8,jnp.float32),
        torso=Qwen3(features=8,num_layers=1,num_heads=2,num_groups=1,head_dim=4,max_wavelength=1e6,
                    context_length=128,hidden_dim=16,dtype=jnp.float32),
        head=ChunkedCategorical(nn.Dense(t.get_vocab_size(),use_bias=False),16))
    policy=LanguagePolicy(native,t.token_to_id('<|endoftext|>'),t.token_to_id('<|im_end|>'),4)
    prefix=t.encode('<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n',add_special_tokens=False).ids
    tokens=np.full((2,128),policy.pad_id,np.int32);tokens[:,:len(prefix)]=prefix
    timestep=Timestep(obs={'tokens':jnp.asarray(tokens),'length':jnp.full(2,len(prefix),jnp.int32)},
        action=jnp.full((2,4),policy.pad_id,jnp.int32),reward=jnp.zeros(2),terminated=jnp.ones(2,bool),truncated=jnp.zeros(2,bool))
    algorithm=RecurrentGRPO(RecurrentGRPOConfig(2,1,1,.2,.04,1.),policy,optax.sgd(1e-3))
    state=algorithm.init(jax.random.key(0),timestep)
    stepped,actions,aux=algorithm.step(state,jax.random.key(1),timestep)
    second=timestep.replace(action=actions,reward=jnp.array([1.,0.]),terminated=jnp.ones(2,bool))
    transitions=jax.tree.map(lambda x:x[None],Transition(first=timestep,second=second,aux=aux))
    updated=jax.jit(algorithm.update)(stepped,jax.random.key(2),transitions)
    diffs=[np.max(np.abs(np.asarray(a)-np.asarray(b))) for a,b in zip(jax.tree.leaves(state.params),jax.tree.leaves(updated.params))]
    assert max(diffs)>0 and np.isfinite(diffs).all()


def test_checkpoint_cut_and_midpool_resume_match_uninterrupted_training(token_cache):
    import optax
    from jax.sharding import Mesh
    from boonta.algorithms.recurrent_bc import RecurrentBC,RecurrentBCConfig
    from boonta.podracers.quadinaros import make,QuadinarosConfig
    from hangar.checkpoint_schedule import offline_epoch
    from hangar.recipes.recurrent_bc_agent_trove import TokenInitialization
    path,t=token_cache
    def build():
        network=Network(feature_extractor=TokenEmbedding(t.get_vocab_size(),8,jnp.float32),
            torso=Qwen3(features=8,num_layers=1,num_heads=2,num_groups=1,head_dim=4,max_wavelength=1e6,
                        context_length=128,hidden_dim=16,dtype=jnp.float32),
            head=ChunkedCategorical(nn.Dense(t.get_vocab_size(),use_bias=False),16))
        dataset=AgentTrove(cache_dir=path,tokenizer=t,length=128,pool_size=4,sampling='permutation')
        return make(QuadinarosConfig((1,128),Mesh(np.array(jax.devices()[:1]),('data',)),True),
                    RecurrentBC(RecurrentBCConfig(1),network,optax.sgd(.001)),
                    TokenInitialization(1,dataset.pad_id),dataset)
    pod=build();state=pod.init(jax.random.key(0))
    for epoch in range(4):state,_=pod.train(state,jax.random.fold_in(jax.random.key(42),epoch),4)
    expected=jax.device_get(state.algorithm_state.params)
    pod=build();state=pod.init(jax.random.key(0));saved=[]
    def callback(state,logs,step):saved.append(serialization.to_bytes(jax.device_get(state)))
    state,_=offline_epoch(pod,state,jax.random.fold_in(jax.random.key(42),0),4,[256],callback)
    assert len(saved)==1
    restored=serialization.from_bytes(pod.init(jax.random.key(0)),saved[0])
    state,_=offline_epoch(pod,restored,jax.random.fold_in(jax.random.key(42),0),2,[],callback,continuing=True)
    for epoch in range(1,4):state,_=pod.train(state,jax.random.fold_in(jax.random.key(42),epoch),4)
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(state.algorithm_state.params)):
        np.testing.assert_allclose(a,b,atol=1e-7,rtol=1e-6)


def test_large_main_budget_requires_int64_before_model_initialization():
    from hangar.recipes.recurrent_bc_agent_trove import make
    with jax.enable_x64(False):
        with pytest.raises(ValueError,match='JAX_ENABLE_X64'):
            make(SimpleNamespace(num_steps=2**34))
