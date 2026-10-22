#!/usr/bin/env python3
from __future__ import annotations
import json, os, sys, random
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq

SRC = Path('/projects/data/datasets/translation_data/SFT/OUTPUT')
sys.path.insert(0, '/projects/data/datasets/translation_data/SFT/DEDUP_PIPELINE')
import sft_normalize as N

rng = random.Random(7)
chat_files = []
for dirpath, _, filenames in os.walk(SRC):
    for fn in filenames:
        if not fn.endswith('.parquet'):
            continue
        path = Path(dirpath) / fn
        try:
            pf = pq.ParquetFile(path, memory_map=True)
        except Exception:
            continue
        cols = list(pf.schema_arrow.names)
        plan = N.plan(cols)
        if plan.get('mode') != 'chat':
            continue
        rel = str(path.relative_to(SRC))
        chat_files.append((path, rel, plan))

tool_domain = [f for f in chat_files if f[1].startswith('Tool_Use/') or 'tool' in f[1].lower()]
print(f'chat_files={len(chat_files)} toolish_files={len(tool_domain)}', flush=True)

def as_list(raw):
    if raw is None:
        return None
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if isinstance(raw, str):
        t = raw.strip()
        if t[:1] in '[{':
            for fn in (json.loads,):
                try:
                    v = fn(t)
                    if isinstance(v, dict):
                        return [v]
                    if isinstance(v, list):
                        return v
                except Exception:
                    return None
    return None

agg = Counter()
files_id = []
files_tc = []
example_id = None
example_tool_no_id = None
example_tc = None
rows_seen = rows_tool = 0

probe = []
seen = set()
for f in tool_domain[:50] + rng.sample(chat_files, k=min(40, len(chat_files))):
    if f[0] in seen:
        continue
    seen.add(f[0]); probe.append(f)

for path, rel, plan in probe:
    col = plan['col']
    try:
        pf = pq.ParquetFile(path, memory_map=True)
        got = 0
        cols = [c for c in [col, 'tools'] if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(columns=cols, batch_size=128):
            for row in batch.to_pylist():
                rows_seen += 1
                turns = as_list(row.get(col))
                if not turns:
                    continue
                row_toolish = False
                for t in turns:
                    if not isinstance(t, dict):
                        continue
                    role = str(t.get('role') or t.get('from') or '').lower()
                    keys = tuple(sorted(t.keys()))
                    if role in {'tool', 'tool_response', 'function', 'observation', 'tool_result'}:
                        row_toolish = True
                        agg['tool_role'] += 1
                        agg[f'tool_keys:{keys}'] += 1
                        has_id = any(k.lower() in {'tool_call_id', 'tool_callid', 'call_id'} or k.lower()=='id' for k in t)
                        # stricter: tool_call_id specifically
                        if t.get('tool_call_id') is not None:
                            agg['has_tool_call_id'] += 1
                            if rel not in files_id:
                                files_id.append(rel)
                            if example_id is None:
                                example_id = (rel, {k: t.get(k) for k in t})
                        elif has_id and 'id' in {k.lower() for k in t}:
                            agg['has_generic_id'] += 1
                        else:
                            agg['tool_role_no_id'] += 1
                            if example_tool_no_id is None:
                                example_tool_no_id = (rel, {k: t.get(k) for k in list(t)[:8]}, keys)
                    if role in {'assistant', 'gpt', 'model', 'ai'} and t.get('tool_calls'):
                        row_toolish = True
                        agg['asst_tool_calls'] += 1
                        if rel not in files_tc:
                            files_tc.append(rel)
                        if example_tc is None:
                            example_tc = (rel, {k: t.get(k) for k in t if k != 'content'}, str(t.get('tool_calls'))[:400])
                    for k, v in t.items():
                        if 'tool_call' in k.lower() and v not in (None, '', [], {}):
                            agg[f'field:{k}'] += 1
                if row.get('tools'):
                    agg['row_has_tools_col'] += 1
                if row_toolish:
                    rows_tool += 1
                got += 1
                if got >= 200:
                    break
            if got >= 200:
                break
    except Exception as e:
        print('err', rel, type(e).__name__, e, flush=True)

print(f'\nrows_seen={rows_seen} rows_with_toolish_turns={rows_tool}', flush=True)
print('=== counters ===', flush=True)
for k, v in agg.most_common(50):
    print(f'{v:8d}  {k}', flush=True)
print(f'\nfiles with tool_call_id: {len(files_id)}', flush=True)
for x in files_id[:20]:
    print(' ', x, flush=True)
print(f'files with assistant.tool_calls: {len(files_tc)}', flush=True)
for x in files_tc[:20]:
    print(' ', x, flush=True)

print('\n=== example WITH tool_call_id ===', flush=True)
print(json.dumps(example_id, ensure_ascii=False, default=str)[:1200] if example_id else 'NONE', flush=True)
print('\n=== example tool role WITHOUT id ===', flush=True)
print(json.dumps(example_tool_no_id, ensure_ascii=False, default=str)[:1200] if example_tool_no_id else 'NONE', flush=True)
print('\n=== example assistant.tool_calls ===', flush=True)
print(json.dumps(example_tc, ensure_ascii=False, default=str)[:1200] if example_tc else 'NONE', flush=True)
