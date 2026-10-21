"""Fix (2026-09-23): two bug classes found in traces_v1_final/ by
hf_agentic_pipeline/scripts/scan_v1_null_args_and_empty_assistant.py.

1. Parquet struct-union nulls in tool-call arguments. In these datasets the
   raw `tool_calls[].function.arguments` column is a parquet STRUCT, so
   every argument name seen anywhere in the column appears as null in every
   call (e.g. a 3-argument find_user_id_by_name_zip call carrying 16 extra
   null arguments). The model would learn to emit them. Confirmed struct-
   typed in the raw schema for every file listed in STRUCT_FILES; null
   argument keys are removed (recursively) there only.
2. Empty assistant messages (no content, no tool_calls, no reasoning) --
   removed everywhere; if that leaves a row not ending on assistant, it is
   trimmed back to its last assistant message (standing rule), or dropped
   if none is left.

Each changed file is backed up first to
sft_traces_v1/data/<name>.backup_before_nulls_empty_fix. Streams line by line.
Files listed in SKIP are not touched (pending a separate decision).
"""

import json
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

DIR = os.environ.get("SFTPIPE_PATH", ".")
BACKUP = os.environ.get("SFTPIPE_PATH", ".")
SCAN = os.environ.get("SFTPIPE_PATH", ".")
STRUCT_FILES = {
    "greghavens__gpt-5.6-sol-coding-and-debugging-traces.jsonl",
    "KermitCO__qwen3.5-9B-tau2bench-retail-traces.jsonl",
    "bobbyzhong__code-orchestrator-synth-traces.jsonl",
    "bobbyzhong__code-orchestrator-synth-traces-fixed.jsonl",
    "r0b0tlab__nemotron-nano-hermes-traces.jsonl",
}
SKIP = set(sys.argv[1:])


def strip_nulls(o):
    if isinstance(o, dict):
        return {k: strip_nulls(v) for k, v in o.items() if v is not None}
    if isinstance(o, list):
        return [strip_nulls(x) for x in o]
    return o


def empty_asst(m):
    return (m["role"] == "assistant" and not m.get("tool_calls")
            and not (m.get("content") or "").strip() and not m.get("reasoning_content"))


def fix(fname):
    path = os.path.join(DIR, fname)
    shutil.copy2(path, os.path.join(BACKUP, fname + ".backup_before_nulls_empty_fix"))
    tmp = path + ".fixtmp"
    c = Counter()
    with open(path) as fi, open(tmp, "w") as fo:
        for line in fi:
            d = json.loads(line)
            msgs = d["messages"]
            changed = False
            if fname in STRUCT_FILES:
                for m in msgs:
                    for tc in m.get("tool_calls") or []:
                        a = tc["function"].get("arguments")
                        if isinstance(a, dict):
                            b = strip_nulls(a)
                            if b != a:
                                tc["function"]["arguments"] = b
                                c["calls_nulls_stripped"] += 1
                                changed = True
            kept = [m for m in msgs if not empty_asst(m)]
            if len(kept) != len(msgs):
                c["empty_assistant_removed"] += len(msgs) - len(kept)
                changed = True
            last = max((i for i, m in enumerate(kept) if m["role"] == "assistant"), default=-1)
            if last < 0:
                c["rows_dropped_no_assistant"] += 1
                continue
            if last < len(kept) - 1:
                kept = kept[: last + 1]
                c["rows_trimmed"] += 1
            if changed:
                c["rows_changed"] += 1
                d["messages"] = kept
                fo.write(json.dumps(d, ensure_ascii=False) + "\n")
            else:
                fo.write(line)
            c["rows_out"] += 1
    os.replace(tmp, path)
    return fname, dict(c)


def main():
    scan = json.load(open(SCAN))["per_file"]
    files = sorted(f for f in scan if f not in SKIP)
    print("fixing", len(files), "files; skipped:", sorted(SKIP))
    tot = Counter()
    with ProcessPoolExecutor(max_workers=16) as ex:
        for name, c in ex.map(fix, files):
            tot.update(c)
            print(name, c, flush=True)
    print("TOTAL", dict(tot))


if __name__ == "__main__":
    main()
