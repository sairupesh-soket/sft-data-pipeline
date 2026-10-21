"""Post-pass: make every row strictly plain OpenAI chat format.

Standing user rule (2026-09-23): "all datasets follow the OpenAI format,
period" -- anything a specific chat template needs is a separate adapter
later. Matched against the 521M sft_43_language_wise_clean data, which is
already in this shape. Per message:

  * reasoning_content (not an OpenAI field) -> folded into content as
    "<think>{reasoning}</think>\\n\\n{content}" (the 521M data's own style),
    so no reasoning is lost;
  * tool_calls[].function.arguments -> JSON string; each call keeps only
    id / type / function{name, arguments};
  * tool replies keep only role / content / tool_call_id (no "name");
  * any other non-OpenAI key is dropped (and counted).
Top level keeps only "messages" and "tools".

    python3 to_openai_format.py FOLDER [FOLDER...]
Recursive (skips "_" dirs), streaming, atomic per file, rewrites only files
that change. Prints a count of every change made.
"""

import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

ALLOWED = {
    "system": {"role", "content", "name"},
    "user": {"role", "content", "name"},
    "assistant": {"role", "content", "name", "tool_calls"},
    "tool": {"role", "content", "tool_call_id"},
}


def fix_msg(m, c):
    m = dict(m)
    r = m.get("reasoning_content")
    if r is not None:
        if isinstance(r, str) and r.strip():
            body = m.get("content") or ""
            m["content"] = f"<think>{r}</think>\n\n{body}" if body else f"<think>{r}</think>"
            c["reasoning_folded_into_content"] += 1
        else:
            # empty / whitespace-only reasoning: nothing to keep, but the key
            # removal must still COUNT as a change, or the row is written back
            # unchanged (bug found on 259 nvidia messages)
            c["empty_reasoning_dropped"] += 1
        del m["reasoning_content"]
    if m.get("tool_calls"):
        calls = []
        for tc in m["tool_calls"]:
            fn = tc.get("function") or {}
            a = fn.get("arguments")
            if not isinstance(a, str):
                a = json.dumps(a if a is not None else {}, ensure_ascii=False)
                c["args_to_string"] += 1
            new = {"id": tc.get("id"), "type": "function", "function": {"name": fn.get("name"), "arguments": a}}
            if new != tc:
                c["tool_call_normalised"] += 1
            calls.append(new)
        m["tool_calls"] = calls
    if m.get("content") is None:
        m["content"] = ""
        c["null_content_to_empty"] += 1
    allowed = ALLOWED.get(m.get("role"), set(m))
    for k in [k for k in m if k not in allowed]:
        c[f"dropped_key:{m.get('role')}.{k}"] += 1
        del m[k]
    return m


def fix_file(path):
    c = Counter()
    tmp = path + ".oaitmp"
    changed_rows = 0
    with open(path) as fi, open(tmp, "w") as fo:
        for line in fi:
            d = json.loads(line)
            before = c.copy()
            out = {"messages": [fix_msg(m, c) for m in d["messages"]]}
            if d.get("tools"):
                out["tools"] = d["tools"]
            for k in d:
                if k not in ("messages", "tools"):
                    c[f"dropped_top_key:{k}"] += 1
            if c != before:
                changed_rows += 1
                fo.write(json.dumps(out, ensure_ascii=False) + "\n")
            else:
                fo.write(line)
    if changed_rows:
        os.replace(tmp, path)
        c["rows_changed"] = changed_rows
    else:
        os.remove(tmp)
    return path, dict(c)


def main():
    files = []
    for top in sys.argv[1:]:
        for root, dirs, fs in os.walk(top):
            dirs[:] = [d for d in dirs if not d.startswith("_")]
            files += [os.path.join(root, f) for f in fs if f.endswith(".jsonl")]
    tot = Counter()
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", 48))) as ex:
        futs = [ex.submit(fix_file, f) for f in sorted(files)]
        for i, fu in enumerate(as_completed(futs), 1):
            p, c = fu.result()
            tot.update(c)
            if c:
                print(f"[{i}/{len(files)}] {p} {c}", flush=True)
    print("TOTAL", dict(tot), flush=True)


if __name__ == "__main__":
    main()
