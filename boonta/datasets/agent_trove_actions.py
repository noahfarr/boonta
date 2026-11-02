"""Shared AgentTrove command-payload spans and strict inference interface."""
from dataclasses import dataclass
import json
import math
import re

DECODER = json.JSONDecoder()


def command_fields(content):
    thoughts = [(m.start(),m.end()) for m in re.finditer(r'<think>.*?</think>',content,re.S)]
    return [m for m in re.finditer(r'"commands"\s*:\s*',content)
            if not any(a <= m.start() < b for a,b in thoughts)]


def commands(content):
    spans = []
    for match in command_fields(content)[-1:]:
        try:
            value, end = DECODER.raw_decode(content, match.end())
            if isinstance(value,list):
                spans.append((match.end(),end,value))
        except ValueError:
            pass
    return spans[-1:]


@dataclass(frozen=True)
class ToolAction:
    commands: tuple[str, ...]
    task_complete: bool


def parse_action(content, max_commands=1):
    text = re.sub(r'<think>.*?</think>', '', content, flags=re.S).strip()
    if '<think>' in text or '</think>' in text:
        raise ValueError('Unclosed reasoning block')
    candidates = []
    for match in re.finditer(r'\{',text):
        try:
            value, end = DECODER.raw_decode(text,match.start())
        except ValueError:
            continue
        if isinstance(value,dict) and 'commands' in value:
            candidates.append((value,end))
    if not candidates:
        raise ValueError('No commands object')
    value, end = candidates[-1]
    if text[end:].strip():
        raise ValueError('Text after final commands object')
    entries = value['commands']
    if not isinstance(entries,list) or not isinstance(value.get('task_complete',False),bool):
        raise ValueError('Invalid tool-call schema')
    if len(entries)>max_commands:
        raise ValueError('Too many commands in one tool turn')
    result = []
    for entry in entries:
        if not isinstance(entry,dict):
            raise ValueError('Command must be an object')
        command = entry.get('keystrokes',entry.get('command'))
        duration = entry.get('duration',0.1)
        if not isinstance(command,str) or not command.strip():
            raise ValueError('Command text required')
        if isinstance(duration,bool) or not isinstance(duration,(float,int)) or not math.isfinite(duration) or duration < 0:
            raise ValueError('Invalid duration')
        result.append(command)
    return ToolAction(tuple(result),value.get('task_complete',False))
