"""Recover the adapter's exact ChatML layout, refusing ambiguous/mismatched masks."""
import hashlib
import re
import unicodedata

import numpy as np

from .agent_trove import encode_messages, serialize


def prompt_key(text):
    canonical = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().casefold()
    return hashlib.sha256(canonical.encode()).hexdigest()


def task_key(source, value):
    if not isinstance(value, str) or not value.strip():
        return None
    value = re.sub(r'(?:_copy\d+)+$', '', value.strip())
    # Generic IDs are source-local; named source task IDs also link "unknown"
    # source rows to their named cohort.
    if re.fullmatch(r'(?:task|problem|instance)[_-]?\d+', value, re.I):
        value = source.casefold() + ':' + value
    return value


def layout(tokens, assistant_mask, tokenizer):
    tokens = np.asarray(tokens)
    end_id = tokenizer.token_to_id('<|im_end|>')
    headers = {role: tokenizer.encode(f'<|im_start|>{role}\n', add_special_tokens=False).ids
               for role in ('system', 'user', 'assistant', 'tool')}
    newline = tokenizer.encode('\n', add_special_tokens=False).ids
    messages, spans = [], []
    position = 0
    while position < len(tokens):
        role = next((role for role, header in headers.items()
                     if np.array_equal(tokens[position:position+len(header)], header)), None)
        if role is None:
            raise ValueError('Cache cannot be losslessly parsed as adapter ChatML')
        body_start = position + len(headers[role])
        candidates = np.flatnonzero(tokens[body_start:] == end_id) + body_start
        boundary = None
        for end in candidates:
            stop = int(end) + 1 + len(newline)
            if not np.array_equal(tokens[int(end)+1:stop], newline):
                continue
            if stop == len(tokens) or any(np.array_equal(tokens[stop:stop+len(h)], h) for h in headers.values()):
                boundary = int(end), stop
                break
        if boundary is None:
            raise ValueError('Missing ChatML message boundary')
        end, stop = boundary
        text = tokenizer.decode(tokens[body_start:end].tolist(), skip_special_tokens=False)
        messages.append({'role': role, 'content': text})
        spans.append((role, body_start, stop))
        position = stop
    encoded = list(encode_messages({'messages': messages}, tokenizer))
    reproduced, mask = serialize(encoded)
    if not np.array_equal(reproduced, tokens) or not np.array_equal(mask, assistant_mask):
        raise ValueError('Decoded messages do not reproduce original tokens and role mask')
    return messages, spans, encoded


def masks(tokens, assistant_mask, tokenizer):
    from .agent_trove_actions import commands
    messages, spans, encoded = layout(tokens, assistant_mask, tokenizer)
    command = np.zeros(len(tokens), bool)
    observation = np.zeros(len(tokens), bool)
    for message, (role, start, stop), (_, _, encoding) in zip(messages, spans, encoded, strict=True):
        if role in ('user', 'tool'):
            observation[start:stop] = True
        if role == 'assistant':
            for left, right, values in commands(message['content']):
                if values:
                    command[start:stop] |= np.array([a < right and b > left for a,b in encoding.offsets])
    return {'assistant': np.asarray(assistant_mask[1:], bool),
            'command': command[1:], 'observation': observation[1:],
            'valid': np.ones(max(len(tokens)-1, 0), bool)}, messages, spans
