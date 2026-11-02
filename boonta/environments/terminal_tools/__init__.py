"""Procedural file tasks with real subprocess outputs and a closed tool grammar.

Generated commands never enter a shell. Only listed file-reading verbs, literal
printf redirection to designated data files, and the trusted checker can run.
Each GRPO group has identical task contents in independent temporary directories.
"""
from dataclasses import dataclass
import glob
import hashlib
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.experimental import io_callback

from boonta.datasets.agent_trove_actions import parse_action
from boonta.datasets.agent_trove import encode_messages, serialize
from boonta.utils import Timestep
from ..spaces import Space

SYSTEM='''You control a terminal through JSON tool calls. Use at most one command per turn.
Respond with {"commands":[{"keystrokes":"COMMAND","duration":0.1}],"task_complete":false}.
Finish with {"commands":[],"task_complete":true}. Available commands: ls, cat FILE,
find . -name PATTERN, grep -l PATTERN *.txt, grep -n PATTERN FILE, check,
and printf '%s\\n' 'TEXT' > answer.txt (or settings.cfg). No shell, pipes or other commands.
All paths are relative to the task workspace.'''


class TerminalEpisode:
    def __init__(self,seed,split='train',max_turns=5):
        if split not in ('train','evaluation'): raise ValueError('Unknown task split')
        self.temporary=tempfile.TemporaryDirectory(prefix='boonta-terminal-')
        self.root=Path(self.temporary.name);self.work=self.root/'work';self.work.mkdir()
        rng=random.Random(f'{split}:{seed}')
        self.task_id=hashlib.sha256(f'{split}:{seed}'.encode()).hexdigest()
        self.kind=rng.choice(('find','read','modify'))
        self.max_turns=max_turns;self.turns=0;self.done=False;self.success=False;self.checked=False
        value=rng.randint(0,99) if split=='train' else rng.randint(100,199)
        self.expected=str(value)
        if self.kind=='find':
            names=[f'{split}_data_{rng.getrandbits(48):012x}.txt' for _ in range(6)]
            self.expected=rng.choice(names)
            marker=f'{split}_MARK_{rng.getrandbits(48):012x}'
            for name in names:(self.work/name).write_text((marker if name==self.expected else 'other')+'\n')
            task=f'Find the text file containing {marker}. Write its basename into answer.txt.'
        elif self.kind=='read':
            (self.work/'settings.cfg').write_text(f'value={value}\n')
            task='Read the value from settings.cfg. Write only its integer value into answer.txt.'
        else:
            (self.work/'settings.cfg').write_text('value=-1\n')
            task=f'Set settings.cfg to value={value}, then run check.'
        self.messages=[{'role':'system','content':SYSTEM},{'role':'user','content':task}]
        target='settings.cfg' if self.kind=='modify' else 'answer.txt'
        expected=f'value={self.expected}' if self.kind=='modify' else self.expected
        self.checker=self.root/'check_task.py'
        self.checker.write_text('from pathlib import Path\nimport sys\ntry:\n'+f'    ok=Path({target!r}).read_text().strip()=={expected!r}\n'+"except OSError:\n    ok=False\nprint('PASS' if ok else 'FAIL')\nsys.exit(0 if ok else 1)\n")

    def path(self,text,writable=False):
        if writable and text not in ('answer.txt','settings.cfg'):
            raise ValueError('Only task data files are writable')
        p=(self.work/text).resolve()
        if not p.is_relative_to(self.work.resolve()) or p.is_symlink(): raise ValueError('Path outside workspace')
        return p

    def execute(self,command):
        args=shlex.split(command)
        if not args: raise ValueError('Empty command')
        destination=None
        if args[0]=='printf':
            if len(args)!=5 or args[1] not in ('%s\\n','%s\n') or args[3]!='>': raise ValueError('Use literal printf redirection')
            destination=self.path(args[4],True)
            argv=['printf',args[1],args[2]]
        elif args[0]=='ls' and args[1:] in ([],['.']): argv=['ls']
        elif args[0]=='cat' and len(args)==2:
            self.path(args[1]);argv=args
        elif args[0]=='find' and len(args)==4 and args[1:3]==['.','-name']:
            if '/' in args[3]:raise ValueError('Filename pattern required')
            argv=['find','.', '-name',args[3]]
        elif args[0]=='grep' and len(args)==4 and args[1] in ('-n','-l'):
            paths=[Path(p) for p in sorted(glob.glob(str(self.work/args[3])))]
            if not paths: paths=[self.path(args[3])]
            for p in paths:self.path(str(p))
            argv=['grep',args[1],'--',args[2],*[str(p.relative_to(self.work)) for p in paths]]
        elif args==['check']:
            self.checked=True;argv=[sys.executable,'-I',str(self.checker)]
        else: raise ValueError('Command outside the common tool grammar')
        result=subprocess.run(argv,cwd=self.work,stdin=subprocess.DEVNULL,capture_output=True,timeout=2,
                              env={'PATH':'/usr/bin:/bin','LANG':'C'})
        if destination is not None:destination.write_bytes(result.stdout)
        return (result.stdout+result.stderr).decode(errors='replace').replace(str(self.root),'/task')[:4096]+f'\nExit code: {result.returncode}'

    def goal(self):
        target=self.work/('settings.cfg' if self.kind=='modify' else 'answer.txt')
        expected=f'value={self.expected}' if self.kind=='modify' else self.expected
        return target.exists() and target.read_text().strip()==expected and (self.kind!='modify' or self.checked)

    def step(self,text):
        if self.done:return 0.,True,False,{'valid_call':0.,'success':float(self.success),'executed_commands':0.}
        self.turns+=1;valid=False;executed=0;complete=False
        try:
            action=parse_action(text)
            if len(action.commands)>1:raise ValueError('One command per turn')
            valid=True;complete=action.task_complete
            outputs=[]
            for command in action.commands:
                outputs.append(self.execute(command));executed+=1
            output='\n'.join(outputs) or 'No command executed.'
        except (ValueError,subprocess.TimeoutExpired,OSError) as error:
            output=f'Tool error: {type(error).__name__}: {error}'.replace(str(self.root),'/task')
        self.messages += [{'role':'assistant','content':text},{'role':'tool','content':output}]
        self.success=self.goal();terminated=self.success or complete
        truncated=self.turns>=self.max_turns and not terminated
        self.done=terminated or truncated
        return float(self.success),terminated,truncated,dict(valid_call=float(valid),success=float(self.success),executed_commands=float(executed))

    def close(self):self.temporary.cleanup()


@struct.dataclass
class TerminalState:
    handles: object
    clock: object


class TerminalTools:
    num_agents=1
    def __init__(self,tokenizer,num_envs=8,group_size=8,context=4096,max_tokens=128,horizon=5,split='train'):
        if num_envs%group_size:raise ValueError('Whole GRPO groups required')
        self.tokenizer=tokenizer;self.num_envs=num_envs;self.group_size=group_size
        self.context=context;self.max_tokens=max_tokens;self.horizon=horizon;self.split=split
        self.pad=tokenizer.token_to_id('<|endoftext|>');self.end=tokenizer.token_to_id('<|im_end|>')
        self.episodes={};self.next_handle=0

    def observation(self,episode):
        tokens,_=serialize(encode_messages({'messages':episode.messages},self.tokenizer))
        header=self.tokenizer.encode('<|im_start|>assistant\n',add_special_tokens=False).ids
        values=np.concatenate([tokens,np.array(header,np.int32)])
        if len(values)+self.max_tokens>self.context:
            raise ValueError('RL history exceeds context; never silently drop observations')
        array=np.full(self.context,self.pad,np.int32);array[:len(values)]=values
        return array,np.int32(len(values))

    def create(self,key):
        digest=hashlib.sha256(np.asarray(key).tobytes()).digest()
        seed=int.from_bytes(digest[:8],'little')
        handles=[];tokens=[];lengths=[]
        for i in range(self.num_envs):
            episode=TerminalEpisode(seed+i//self.group_size,self.split,self.horizon)
            handle=self.next_handle;self.next_handle+=1;self.episodes[handle]=episode
            x,n=self.observation(episode);handles.append(handle);tokens.append(x);lengths.append(n)
        return np.array(handles,np.int32),np.stack(tokens),np.array(lengths,np.int32)

    def shapes(self):
        return (jax.ShapeDtypeStruct((self.num_envs,),jnp.int32),
                jax.ShapeDtypeStruct((self.num_envs,self.context),jnp.int32),
                jax.ShapeDtypeStruct((self.num_envs,),jnp.int32))

    def init(self,key):
        handles,tokens,lengths=io_callback(self.create,self.shapes(),jax.random.key_data(key),ordered=True)
        obs={'tokens':tokens,'length':lengths}
        return TerminalState(handles,jnp.int32(0)),Timestep(obs=obs,action=jnp.full((self.num_envs,self.max_tokens),self.pad,jnp.int32),
            reward=jnp.zeros(self.num_envs,jnp.float32),terminated=jnp.ones(self.num_envs,bool),truncated=jnp.zeros(self.num_envs,bool),
            info={k:jnp.zeros(self.num_envs,jnp.float32) for k in ('valid_call','success','executed_commands')})

    def advance(self,handles,clock,actions,key):
        rewards=[];terminated=[];truncated=[];infos=[];tokens=[];lengths=[]
        for handle,action in zip(handles,actions,strict=True):
            values=list(map(int,action));values=values[:values.index(self.end)] if self.end in values else values
            text=self.tokenizer.decode(values,skip_special_tokens=False)
            episode=self.episodes[int(handle)]
            if self.end not in action:text='INVALID: missing assistant handoff'
            reward,term,trunc,info=episode.step(text)
            x,n=self.observation(episode)
            rewards.append(reward);terminated.append(term);truncated.append(trunc);infos.append(info);tokens.append(x);lengths.append(n)
        if (int(clock)+1)%self.horizon==0:
            for handle in handles:self.episodes.pop(int(handle)).close()
            handles,new_tokens,new_lengths=self.create(key)
            tokens=list(new_tokens);lengths=list(new_lengths)
            truncated=np.asarray(truncated)|~np.asarray(terminated)
        return (np.asarray(handles,np.int32),np.stack(tokens),np.asarray(lengths,np.int32),
                np.asarray(rewards,np.float32),np.asarray(terminated,bool),np.asarray(truncated,bool),
                {k:np.array([x[k] for x in infos],np.float32) for k in infos[0]})

    def step(self,key,state,action):
        shapes=(*self.shapes(),jax.ShapeDtypeStruct((self.num_envs,),jnp.float32),
            jax.ShapeDtypeStruct((self.num_envs,),bool),jax.ShapeDtypeStruct((self.num_envs,),bool),
            {k:jax.ShapeDtypeStruct((self.num_envs,),jnp.float32) for k in ('valid_call','success','executed_commands')})
        handles,tokens,lengths,reward,terminated,truncated,info=io_callback(self.advance,shapes,state.handles,state.clock,action,jax.random.key_data(key),ordered=True)
        return TerminalState(handles,state.clock+1),Timestep(obs={'tokens':tokens,'length':lengths},action=action,
            reward=reward,terminated=terminated,truncated=truncated,info=info)

    def close(self,state=None):
        for episode in self.episodes.values():episode.close()
        self.episodes.clear()

    def time_limit(self):return self.horizon
    def action_space(self):return Space(shape=(self.max_tokens,),dtype=jnp.int32,low=0,high=self.tokenizer.get_vocab_size()-1)
    def observation_space(self):raise NotImplementedError('Structured token histories')


def make(env_id,tokenizer_id,**kwargs):
    from tokenizers import Tokenizer
    if env_id != 'files-v0': raise ValueError('Unknown terminal task suite')
    return TerminalTools(Tokenizer.from_file(str(tokenizer_id)),**kwargs)
