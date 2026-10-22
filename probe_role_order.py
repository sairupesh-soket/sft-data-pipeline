#!/usr/bin/env python3
"""Phase1-only: count user→tool and agentic role-order issues."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

root = Path('/projects/data/datasets/traces_team/sft-hf-datasets/full')
files = sorted(root.glob('phase1_*_openai_sft_full.jsonl'))

total = 0
with_tool = 0
user_then_tool = 0
tool_before_assistant = 0
starts_with_tool = 0
ends_on_tool = 0
proper = 0
bad = 0
adj = Counter()
examples = []

for path in files:
    print(f'scan {path.name}', flush=True)
    n = 0
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            n += 1
            if '"role": "tool"' not in line and '"role":"tool"' not in line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rs = [m.get('role') for m in (rec.get('messages') or [])]
            if 'tool' not in rs:
                continue
            with_tool += 1
            is_bad = False
            if rs[0] == 'tool':
                starts_with_tool += 1
                is_bad = True
            if rs[-1] == 'tool':
                ends_on_tool += 1
                is_bad = True
            if 'assistant' not in rs or rs.index('tool') < rs.index('assistant'):
                tool_before_assistant += 1
                is_bad = True
            for a, b in zip(rs, rs[1:]):
                adj[f'{a}->{b}'] += 1
                if a == 'user' and b == 'tool':
                    user_then_tool += 1
                    is_bad = True
                    if len(examples) < 6:
                        examples.append({'file': path.name, 'roles': ' > '.join(rs)})
            if is_bad:
                bad += 1
            else:
                proper += 1
            if with_tool % 100000 == 0:
                print(f'  tool_rows={with_tool:,} user→tool={user_then_tool:,}', flush=True)
    print(f'  lines={n:,} running_tool={with_tool:,} user→tool={user_then_tool:,}', flush=True)

out = {
    'phase1_rows': total,
    'with_tool_role': with_tool,
    'user_then_tool': user_then_tool,
    'user_then_tool_pct_of_tool': round(100 * user_then_tool / with_tool, 2) if with_tool else None,
    'tool_before_any_assistant': tool_before_assistant,
    'starts_with_tool': starts_with_tool,
    'ends_on_tool': ends_on_tool,
    'proper_agentic': proper,
    'bad_agentic': bad,
    'tool_rows_ok_no_user_tool': with_tool - user_then_tool,
}
print('\n=== SUMMARY ===', flush=True)
print(json.dumps(out, indent=2), flush=True)
print('\n=== top tool adjacencies ===', flush=True)
for k, v in adj.most_common(20):
    print(f'{v:10,}  {k}', flush=True)
print('\n=== examples ===', flush=True)
print(json.dumps(examples, indent=2), flush=True)
Path('/projects/data/datasets/traces_team/sft-hf-datasets/full/phase1_role_order_report.json').write_text(
    json.dumps({'summary': out, 'adjacencies': adj.most_common(40), 'examples': examples}, indent=2),
    encoding='utf-8',
)
print('wrote report', flush=True)
