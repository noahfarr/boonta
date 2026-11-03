import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
from tokenizers import Tokenizer

def prompt_key(text):
    canonical = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().casefold()
    return hashlib.sha256(canonical.encode()).hexdigest()

def task_key(source, value):
    if not isinstance(value, str) or not value.strip():
        return None
    value = re.sub(r'(?:_copy\d+)+$', '', value.strip())
    if re.fullmatch(r'(?:task|problem|instance)[_-]?\d+', value, re.I):
        value = source.casefold() + ':' + value
    return value


def initial_prompt(tokens, assistant, tokenizer, headers, end_id, newline):
    active = np.flatnonzero(assistant)
    if not len(active):
        raise ValueError('No assistant span')
    prefix = tokens[:active[0]]
    position = 0
    while position < len(prefix):
        role = next((r for r,h in headers.items() if np.array_equal(prefix[position:position+len(h)],h)),None)
        if role not in ('system','user'):
            raise ValueError('No unambiguous initial user task')
        start = position+len(headers[role])
        boundary = None
        for stop in np.flatnonzero(prefix[start:] == end_id)+start:
            after = int(stop)+1+len(newline)
            if np.array_equal(prefix[int(stop)+1:after],newline) and any(np.array_equal(prefix[after:after+len(h)],h) for h in headers.values()):
                boundary = int(stop),after
                break
        if boundary is None:
            raise ValueError('Ambiguous initial prompt boundary')
        stop,position = boundary
        if role == 'user':
            return tokenizer.decode(prefix[start:stop].tolist(),skip_special_tokens=False)
    raise ValueError('No initial user task')


def group_cache(cache, output, tokenizer, seed=42, trusted_path=None, max_length=16384):
    manifest=json.loads((cache/'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('Complete cache required')
    if hashlib.sha256(tokenizer.to_str().encode()).hexdigest()!=manifest['tokenizer_runtime_sha256']:
        raise ValueError('Tokenizer mismatch')
    output.mkdir(parents=True,exist_ok=False)
    n=manifest['rows']
    parent=np.arange(n,dtype=np.int32)
    included=np.zeros(n,bool)
    lengths=np.zeros(n,np.int32)
    blm=np.zeros(n,np.int32)
    clm=np.zeros(n,np.int32)
    old=np.zeros(n,np.uint8)
    keys={}
    old_cross=Counter()
    unresolved=Counter()
    trusted={}
    if trusted_path:
        for line in trusted_path.open():
            item=json.loads(line)
            if item['index'] in trusted: raise ValueError('Duplicate trusted identity row')
            trusted[item['index']]=item
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=int(parent[i])
        return i
    def link(index,key,kind):
        digest=hashlib.sha256((kind+':'+key).encode()).digest()
        if digest in keys:
            other=keys[digest]
            if old[index]!=old[other]: old_cross[kind]+=1
            a,b=root(index),root(other)
            parent[max(a,b)]=min(a,b)
        else: keys[digest]=index
    tokens=np.memmap(cache/'tokens.bin',mode='r',dtype='<i4')
    mask=np.memmap(cache/'mask.bin',mode='r',dtype='u1')
    headers={r:tokenizer.encode(f'<|im_start|>{r}\n',add_special_tokens=False).ids for r in ('system','user','assistant','tool')}
    newline=tokenizer.encode('\n',add_special_tokens=False).ids
    end_id=tokenizer.token_to_id('<|im_end|>')
    trusted_count=0
    with (cache/'records.jsonl').open() as records:
        for line in records:
            row=json.loads(line); i=row['index']
            old[i]=int(row['split']=='validation')
            if not row['cached'] or row['length']>max_length: continue
            included[i]=True; lengths[i]=row['length']; blm[i]=row['blm_targets']; clm[i]=row['length']-1
            link(i,row['identity'],'trajectory')
            key=task_key(row['source'],row['metadata'].get('task_id') or row['metadata'].get('task'))
            if key: link(i,key,'source_task')
            else: unresolved['missing_source_task_id']+=1
            if i in trusted:
                identity=trusted[i]
                if identity['identity']!=row['identity']: raise ValueError('Trusted identity is bound to a different trajectory')
                required=('task_id','prompt_hash','repository_applicable','repository_id')
                if any(k not in identity for k in required) or not identity['task_id'] or not identity['prompt_hash']:
                    raise ValueError('Incomplete trusted task/prompt/repository identity')
                if not isinstance(identity['repository_applicable'],bool): raise ValueError('Repository applicability must be explicit')
                if identity['repository_applicable'] and not identity['repository_id']: raise ValueError('Repository identity required')
                link(i,identity['task_id'],'trusted_task'); link(i,identity['prompt_hash'],'prompt')
                if identity['repository_applicable']: link(i,identity['repository_id'].casefold(),'repository')
                trusted_count+=1
            else:
                offset=row['offset']; stop=offset+row['length']
                try:
                    prompt=initial_prompt(tokens[offset:stop],mask[offset:stop],tokenizer,headers,end_id,newline)
                    link(i,prompt_key(prompt),'prompt')
                    repos=re.findall(r'https?://github\.com/([\w.-]+/[\w.-]+)',prompt,re.I)
                    for repo in repos: link(i,re.sub(r'\.git$','',repo.rstrip('.')).casefold(),'repository')
                    if not repos: unresolved['repository_identity_unknown']+=1
                except ValueError:
                    unresolved['prompt_layout_unresolved']+=1
            if i and i%100000==0: print(f'grouped metadata through row {i}',flush=True)
    assignment=np.full(n,2,np.uint8)
    group_sizes=Counter()
    for i in np.flatnonzero(included):
        r=root(int(i)); group_sizes[r]+=1
        assignment[i]=int(int(hashlib.sha256(f'{seed}:{r}'.encode()).hexdigest()[:16],16)%100==0)
    assignment.tofile(output/'splits.bin')
    digest=hashlib.sha256((output/'splits.bin').read_bytes()).hexdigest()
    budgets={}
    for c in (4096,8192,16384,32768):
        if c>max_length: continue
        budgets[str(c)]={}
        for code,name in ((0,'train'),(1,'validation')):
            selected=(assignment==code)&(lengths<=c)&included
            budgets[str(c)][name]=dict(trajectories=int(selected.sum()),blm_targets=int(blm[selected].sum(dtype=np.int64)),
                clm_targets=int(clm[selected].sum(dtype=np.int64)),allocated_positions=int(selected.sum())*c)
    certificate=trusted_count==int(included.sum()) and bool(trusted_path)
    result=dict(format_version=1,cache_revision=manifest['revision'],cache_rows=n,
        tokenizer_sha256=manifest['tokenizer_runtime_sha256'],max_length=max_length,seed=seed,
        split_file='splits.bin',split_sha256=digest,code_meaning={'0':'train','1':'validation','2':'excluded'},
        grouping='connected components of trajectory, normalized initial prompt, source task (copy suffix removed), and known repository IDs',
        old_split_crossings=dict(old_cross),unresolved=dict(unresolved),trusted_identity_rows=trusted_count,
        included_rows=int(included.sum()),groups=len(group_sizes),largest_group=max(group_sizes.values(),default=0),
        identity_coverage_complete=certificate,task_disjoint_certified=certificate,
        certification_scope='Disjoint supplied identity keys; semantic correctness of trusted source identities is an external provenance assumption. Automatic keys alone are NOT a certificate.',
        trusted_identity_sha256=hashlib.sha256(trusted_path.read_bytes()).hexdigest() if trusted_path else None,
        budgets=budgets)
    (output/'split.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--trusted-identities',type=Path);p.add_argument('--max-length',type=int,default=16384)
    a=p.parse_args()
    r=group_cache(a.cache,a.output,Tokenizer.from_file(str(a.tokenizer)),a.seed,a.trusted_identities,a.max_length)
    print(json.dumps(r,indent=2))

if __name__=='__main__': main()
