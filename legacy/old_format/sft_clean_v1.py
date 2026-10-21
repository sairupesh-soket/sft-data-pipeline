"""THE cleaner for OpenAI-format SFT data -- companion to sft_audit.py.

Reads a SOURCE folder (never modified) and writes a cleaned copy to an
OUTPUT folder, same relative layout. Every fix found so far is applied, in
this order, to every row:

 1. to_openai_format  -- only OpenAI keys; reasoning_content -> <think>;
                         arguments -> JSON string; tool replies keep only
                         role/content/tool_call_id            (2026-09-23)
 2. shapes            -- flat tool defs wrapped; Kimi tool-declare system
                         message removed; "null" answers with the reply in
                         <think> unwrapped; <answer> moved out of <think>;
                         non-string content -> row dropped        (2026-09-24)
 3. tools             -- tool replies linked to their call id; replies to
                         calls written in TEXT become role "user" (2026-09-23)
 4. think             -- split reasoning/answer turns merged; problem-
                         creation reasoning removed; answer-inside-<think>
                         unwrapped; think-only final turns trimmed   (2026-09-23)
 5. role rules        -- empty assistant turns removed; row trimmed to end
                         on assistant (dropped if none); rows with an
                         assistant right after system dropped  (standing rules)
 6. audit gate        -- any row still failing ANY sft_audit.py rule (every
                         check except INFO_CHECKS) is dropped   (2026-09-24)

New fixes: add them here (and a matching check in sft_audit.py), so every
dataset gets the same treatment. Usage:

    python3 sft_clean.py SOURCE_DIR OUTPUT_DIR
(.jsonl and .jsonl.gz sources; a .gz source is written back as .gz)
Then register OUTPUT_DIR in sft_audit.py DATASETS and run the audit.
"""

import gzip
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

T = os.environ.get("SFTPIPE_PATH", ".")
sys.path.insert(0, f"{T}/hf_agentic_pipeline/scripts")
sys.path.insert(0, f"{T}/sft_audit/fixes")
import to_openai_format as F1            # noqa: E402
import fix_20260924_shapes as F2         # noqa: E402
import fix_countdown_and_tools as F3     # noqa: E402
import fix_think_issues as F4            # noqa: E402
sys.path.insert(0, f"{T}/sft_audit")
import sft_audit as AUDIT                # noqa: E402


def role_rules(msgs, c):
    kept = [m for m in msgs if not (m["role"] == "assistant" and not m.get("tool_calls")
                                    and not (m.get("content") or "").strip())]
    if len(kept) != len(msgs):
        c["empty_assistant_removed"] += len(msgs) - len(kept)
    last = max((i for i, m in enumerate(kept) if m["role"] == "assistant"), default=-1)
    if last < 0:
        c["dropped_no_assistant"] += 1
        return None
    if len(kept) > 1 and kept[0]["role"] == "system" and kept[1]["role"] == "assistant":
        c["dropped_system_then_assistant"] += 1
        return None
    if last < len(kept) - 1:
        c["trimmed_to_last_assistant"] += 1
    return kept[: last + 1]


def clean_row(d, c):
    if not d.get("messages"):
        c["dropped_empty_messages"] += 1
        return None
    row = {"messages": [F1.fix_msg(m, c) for m in d["messages"]]}
    if d.get("tools"):
        row["tools"] = d["tools"]
    for k in d:
        if k not in ("messages", "tools"):
            c[f"dropped_top_key:{k}"] += 1
    row, _ = F2.fix_row(row, c)
    if row is None:
        return None
    F3.fix_tools(row, c)
    msgs, _ = F4.fix_row(row["messages"], c)
    if msgs is None:
        c["dropped_no_answer_after_think_fix"] += 1
        return None
    msgs = role_rules(msgs, c)
    if msgs is None:
        return None
    row["messages"] = msgs
    # 6. final gate: a row that still fails ANY audit rule is dropped, so
    #    the cleaner and the audit always enforce the same list (user, 2026-09-24)
    failed = AUDIT.check_row(row) - AUDIT.INFO_CHECKS
    if failed:
        for f in failed:
            c[f"dropped_fails:{f}"] += 1
        c["dropped_fails_audit_rule"] += 1
        return None
    return row


def clean_file(args):
    src, dst = args
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    c = Counter()
    tmp = dst + ".tmp"
    gz = src.endswith(".gz")
    with (gzip.open(src, "rt") if gz else open(src)) as fi, \
            (gzip.open(tmp, "wt", compresslevel=3) if gz else open(tmp, "w")) as fo:
        for line in fi:
            if not line.strip():
                continue
            c["rows_in"] += 1
            try:
                row = clean_row(json.loads(line), c)
            except Exception:
                c["dropped_bad_json_or_crash"] += 1
                continue
            if row is not None:
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                c["rows_out"] += 1
    os.replace(tmp, dst)
    return src, dict(c)


def main():
    src_top, out_top = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
    if os.path.abspath(src_top) == os.path.abspath(out_top):
        sys.exit("source and output must differ -- the source is never modified")
    jobs = []
    for root, dirs, fs in os.walk(src_top):
        dirs[:] = [x for x in dirs if not x.startswith("_")]
        for f in fs:
            if f.endswith((".jsonl", ".jsonl.gz")):
                s = os.path.join(root, f)
                jobs.append((s, os.path.join(out_top, os.path.relpath(s, src_top))))
    tot = Counter()
    per_file = {}
    # multiprocessing.Pool, not ProcessPoolExecutor: submitting ~30K futures at once deadlocked on Python 3.10 (2026-09-25)
    with Pool(int(os.environ.get("WORKERS", 48))) as pool:
        for i, res in enumerate(pool.imap_unordered(clean_file, sorted(jobs)), 1):
            s, c = res
            tot.update(c)
            per_file[os.path.relpath(s, src_top)] = c
            if i % 50 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] files cleaned", flush=True)
    os.makedirs(out_top, exist_ok=True)
    json.dump({"source": src_top, "totals": dict(tot), "per_file": per_file},
              open(os.path.join(out_top, "_clean_report.json"), "w"), indent=1)
    print("TOTAL", dict(tot), flush=True)


if __name__ == "__main__":
    main()
