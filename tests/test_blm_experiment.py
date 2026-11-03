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
from boonta.networks import Categorical
from boonta.networks.chunked_categorical import ChunkedCategorical
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


def test_without_replacement_is_paired_and_resumable(token_cache):
    path,t=token_cache
    ids=[];saved=None
    for mode in (True,False):
        d=AgentTrove(cache_dir=path,tokenizer=t,length=128,pool_size=4,sampling='permutation',assistant_only=mode)
        state=d.init();sequence=[]
        for pool in range(4):
            state=d.update(state,jax.random.key(pool),None)
            for offset in (0,2):
                batch=jax.jit(lambda s,step:d.sample(s,jax.random.key(999),(2,128),step=step))(state,jnp.int32((pool*4+offset)*128))
                sequence.extend(np.asarray(batch.aux['record_index']).tolist())
            if pool==1: saved=serialization.to_bytes(state)
        ids.append(sequence)
        restored=serialization.from_bytes(d.init(),saved)
        next_state=d.update(restored,jax.random.key(9999),None)
        next_ids=d.sample(next_state,jax.random.key(0),(2,128),step=8*128).aux['record_index']
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


def test_large_main_budget_requires_int64_before_model_initialization():
    from hangar.recipes.recurrent_bc_agent_trove import make
    with jax.enable_x64(False):
        with pytest.raises(ValueError,match='JAX_ENABLE_X64'):
            make(SimpleNamespace(num_steps=2**34))
