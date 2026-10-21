"""Convert Traces_datasets/OUTPUT parquet files into OpenAI chat-format SFT JSONL.

Source (read-only, never modified):
    a Traces_datasets/OUTPUT style folder (--src)
    456 scraped Hugging Face datasets, 42,537 parquet files, 386GB total.
    Only "traces" is in the name -- most are real chat/agent data, but some
    are not conversations at all (RL game states, hardware traces, raw
    logit dumps). This script only converts datasets that already carry a
    'messages' / 'conversations' / 'conversation' column (285 of 456
    datasets, ~72GB) -- the ones that are genuinely chat-shaped. The other
    ~171 datasets are skipped and listed in skipped_datasets_report.json for
    separate, bespoke handling later; forcing them through a generic chat
    converter would silently produce garbage.

Output:
    <out>/<dataset>.jsonl  (--out)
    One line per conversation: {"messages": [{"role": ..., "content": ...}, ...]}
    Extra fields kept on a message when present: tool_calls, tool_call_id,
    name, reasoning_content (kept as its own field, not merged into content,
    since the target GLM chat template handles reasoning content specially).

Known gap this script fixes vs. the teammate pipeline at
an earlier teammate pipeline: that pipeline
silently drops 100% of rows from datasets using the older ShareGPT-style
'from'/'value' keys instead of 'role'/'content' (confirmed on
DJLougen__hermes-agent-traces-filtered and lambda__hermes-agent-reasoning-traces,
26,026 real rows). This script maps 'from' -> role (human->user, gpt->assistant,
system->system, tool/observation->tool) and 'value' -> content, so that data
is kept. Tool calls embedded as raw <tool_call>/<think> XML text inside a
'value' string are left exactly as-is (not parsed into structured tool_calls)
-- safest option, zero risk of corrupting data via imperfect XML parsing.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SRC = Path(os.environ.get("SFTPIPE_SRC", "."))   # raw source folder (set by sft_pipeline.convert --src)
OUT_DIR = Path(os.environ.get("SFTPIPE_OUT", "out"))   # output folder (set by sft_pipeline.convert --out)
LOG_DIR = Path(os.environ.get("SFTPIPE_LOG", str(OUT_DIR).rstrip("/") + "_logs"))   # reports outside the data folder
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = LOG_DIR / "conversion_report.json"
SKIPPED_REPORT_PATH = LOG_DIR / "skipped_datasets_report.json"
LOG_PATH = LOG_DIR / "convert.log"

MSG_COLUMNS = ("messages", "conversations", "conversation")

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "chatgpt": "assistant",
    "bot": "assistant",
    "tool": "tool",
    "function_call": "assistant",
    "function-call": "assistant",
    "observation": "tool",
    "function_response": "tool",
    "function": "tool",
}

CHATML_RE = re.compile(r"<\|im_start\|>\s*(\w+)\s*\n(.*?)(?=<\|im_start\|>|\Z)", re.DOTALL)


def to_plain(o):
    """Recursively convert numpy/pyarrow objects into plain Python types."""
    if isinstance(o, np.ndarray):
        return [to_plain(x) for x in o.tolist()]
    if isinstance(o, dict):
        return {k: to_plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_plain(x) for x in o]
    if isinstance(o, np.generic):
        return o.item()
    if hasattr(o, "items") and not isinstance(o, str):
        try:
            return {k: to_plain(v) for k, v in dict(o).items()}
        except Exception:
            pass
    return o


def normalize_tool_calls(tc):
    tc = to_plain(tc)
    if not tc:
        return None
    if isinstance(tc, dict):
        tc = [tc]
    out = []
    for i, call in enumerate(tc):
        if not isinstance(call, dict):
            continue
        fn = to_plain(call.get("function")) or {}
        if isinstance(fn, dict) and (fn.get("name") or fn.get("arguments") is not None):
            name = fn.get("name")
            args = fn.get("arguments")
        else:
            # some datasets put name/arguments directly on the tool_call entry
            name = call.get("name")
            args = call.get("arguments")
        # NOTE: arguments is kept as a dict/object here, NOT json.dumps'd to a
        # string. The raw OpenAI Chat Completions API spec uses a JSON string,
        # but the actual downstream consumer of this data is the GLM chat
        # template (chat_template.jinja), which does `tc.arguments.items()`
        # directly on the object -- a string there raises
        # "'str object' has no attribute 'items'" and crashes rendering.
        # Confirmed by testing real converted rows through
        # AutoTokenizer.apply_chat_template() against the local GLM-5.2-FP8
        # tokenizer before this data was called done.
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"value": args}
        elif args is None:
            args = {}
        if not isinstance(args, dict):
            args = {"value": args}
        if not name:
            continue
        out.append(
            {
                "id": call.get("id") or f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        )
    return out or None


def convert_message(m):
    m = to_plain(m)
    if not isinstance(m, dict):
        return None

    role = m.get("role")
    if not role:
        role = m.get("from")
    if not role:
        return None
    role = ROLE_MAP.get(str(role).strip().lower(), role)

    if "content" in m and m.get("content") is not None:
        content = m.get("content")
    elif m.get("value") is not None:
        content = m.get("value")
    else:
        content = None

    tool_calls = normalize_tool_calls(m.get("tool_calls"))

    if content is None and not tool_calls:
        return None  # empty placeholder message (e.g. system turn with content=None, no tool_calls) -- drop

    if content is None:
        content = ""

    out = {"role": role, "content": content}
    if tool_calls:
        out["tool_calls"] = tool_calls
    tool_call_id = m.get("tool_call_id")
    if tool_call_id:
        out["tool_call_id"] = tool_call_id
    name = m.get("name")
    if name:
        out["name"] = name
    reasoning = m.get("reasoning_content") or m.get("think")
    if reasoning:
        out["reasoning_content"] = reasoning
    return out


CORE_CHATML_ROLES = {"system", "user", "assistant", "tool"}


def parse_chatml(text):
    matches = CHATML_RE.findall(text)
    if not matches:
        return None
    msgs = []
    for role, content in matches:
        role = role.strip().lower()
        content = content.replace("<|im_end|>", "").strip("\n")
        if role in CORE_CHATML_ROLES:
            msgs.append({"role": role, "content": content})
        elif msgs:
            # Model-specific sub-tags splitting ONE turn into pieces (e.g.
            # QwQ's own <|im_start|>think / <|im_start|>answer convention for
            # separating its reasoning from its final answer within a single
            # assistant response) -- not a real chat role. Merge into the
            # message it belongs to instead of inventing a fake new message:
            # confirmed the real GLM chat template has no branch for an
            # unrecognized role and silently drops it during rendering, so
            # treating "think"/"answer" as real roles doesn't error, it just
            # silently destroys that content (caught via a structural
            # role-sequence audit, not the render-without-error check alone).
            msgs[-1]["content"] = (msgs[-1]["content"] + "\n\n" + content).strip()
        else:
            msgs.append({"role": "assistant", "content": content})
    return msgs


def extract_msg_list(row_val):
    row_val = to_plain(row_val)
    if row_val is None:
        return None, None
    if isinstance(row_val, list):
        return row_val, None
    if isinstance(row_val, str):
        s = row_val.strip()
        if not s:
            return None, None
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed, None
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return parsed, None
        except Exception:
            pass
        if "<|im_start|>" in s:
            parsed = parse_chatml(s)
            if parsed:
                return parsed, "chatml_text"
        return None, None
    return None, None


def convert_row(row_val):
    msgs_raw, _mode = extract_msg_list(row_val)
    if not msgs_raw:
        return None, "skip_empty_or_invalid"
    out_msgs = []
    for m in msgs_raw:
        cm = convert_message(m)
        if cm is not None:
            out_msgs.append(cm)
    if len(out_msgs) == 0:
        return None, "skip_all_messages_dropped"
    if all(mm["role"] == "system" for mm in out_msgs):
        return None, "skip_system_only"
    return {"messages": out_msgs}, "ok"


def group_files():
    files = [f for f in os.listdir(SRC) if f.endswith(".parquet")]
    groups: dict[str, list[str]] = {}
    for f in files:
        prefix = re.sub(r"(_\d+)?\.parquet$", "", f)
        groups.setdefault(prefix, []).append(f)
    return groups


def detect_scope(groups):
    """Read schema metadata only (fast) to decide which datasets are chat-shaped."""
    in_scope = {}
    out_scope = {}
    for prefix, flist in groups.items():
        fpath = SRC / sorted(flist)[0]
        try:
            schema = pq.read_schema(str(fpath))
            cols = set(schema.names)
        except Exception as e:
            out_scope[prefix] = f"schema_read_error: {e}"
            continue
        hit = [c for c in MSG_COLUMNS if c in cols]
        if hit:
            in_scope[prefix] = sorted(flist)
        else:
            out_scope[prefix] = "no_messages_conversations_column"
    return in_scope, out_scope


def find_msg_column(columns):
    for c in MSG_COLUMNS:
        if c in columns:
            return c
    return None


def iter_column_values(fpath, col):
    """Stream raw values of a single column, one row at a time.

    Uses pure pyarrow batch streaming rather than pandas.read_parquet: it is
    just as fast on ordinary files (measured equal, ~0.28s either way, on a
    3,643-row file) but pandas' struct/list -> pandas conversion (a) crashes
    outright on some files with unrelated deeply nested struct columns
    ("Nested data conversions not implemented for chunked array outputs",
    seen on nvidia__Open-SWE-Traces' 'metadata' column) and (b) is not
    incremental -- it must finish converting the ENTIRE column before
    yielding a single row, which is unusable on a file like that one (a
    single 133GB-uncompressed row group). Streaming batches avoids both
    problems and lets rows start flowing immediately.
    """
    pf = pq.ParquetFile(fpath)
    for batch in pf.iter_batches(batch_size=1000, columns=[col]):
        for row in batch.to_pylist():
            yield row[col]


def process_dataset(item):
    prefix, flist = item
    out_path = OUT_DIR / f"{prefix}.jsonl"
    reasons = Counter()
    rows_in = 0
    rows_out = 0
    errors = []
    try:
        with open(out_path, "w") as out_f:
            for fname in flist:
                fpath = SRC / fname
                try:
                    schema_cols = set(pq.read_schema(str(fpath)).names)
                except Exception as e:
                    errors.append(f"{fname}: schema_read_error: {e}")
                    continue
                col = find_msg_column(schema_cols)
                if col is None:
                    errors.append(f"{fname}: no_msg_column_at_read_time")
                    continue
                try:
                    for val in iter_column_values(fpath, col):
                        rows_in += 1
                        try:
                            result, reason = convert_row(val)
                        except Exception:
                            reason = "skip_convert_exception"
                            result = None
                        reasons[reason] += 1
                        if result is not None:
                            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            rows_out += 1
                except Exception as e:
                    errors.append(f"{fname}: read_error: {e}")
                    continue
        if rows_out == 0:
            out_path.unlink(missing_ok=True)
    except Exception:
        errors.append(traceback.format_exc(limit=6))

    return {
        "prefix": prefix,
        "files": len(flist),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "reasons": dict(reasons),
        "errors": errors,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    groups = group_files()
    print(f"Total dataset groups found: {len(groups)}")

    in_scope, out_scope = detect_scope(groups)
    print(f"In scope (chat-shaped, will convert): {len(in_scope)}")
    print(f"Out of scope (not chat-shaped, skipped): {len(out_scope)}")

    with open(SKIPPED_REPORT_PATH, "w") as f:
        json.dump(
            {
                "note": (
                    "These datasets have no messages/conversations/conversation column, "
                    "so they were not converted by traces_to_openai_sft.py. Many are not "
                    "chat data at all (RL traces, hardware traces, raw prompt/answer pairs "
                    "needing bespoke per-dataset mapping). Review individually before adding."
                ),
                "count": len(out_scope),
                "datasets": out_scope,
            },
            f,
            indent=2,
        )

    total_rows_in = 0
    total_rows_out = 0
    combined_reasons = Counter()
    per_dataset_stats = {}
    dataset_errors = {}

    tasks = list(in_scope.items())
    done = 0
    with ProcessPoolExecutor(max_workers=32) as ex, open(LOG_PATH, "w") as log_f:
        futs = {ex.submit(process_dataset, item): item[0] for item in tasks}
        for fut in as_completed(futs):
            prefix = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"prefix": prefix, "files": 0, "rows_in": 0, "rows_out": 0, "reasons": {}, "errors": [str(e)]}
            done += 1
            total_rows_in += res["rows_in"]
            total_rows_out += res["rows_out"]
            combined_reasons.update(res["reasons"])
            per_dataset_stats[res["prefix"]] = {
                "rows_in": res["rows_in"],
                "rows_out": res["rows_out"],
                "files": res["files"],
            }
            if res["errors"]:
                dataset_errors[res["prefix"]] = res["errors"]
            line = f"[{done}/{len(tasks)}] {res['prefix']}: in={res['rows_in']} out={res['rows_out']} errors={len(res['errors'])}"
            print(line, flush=True)
            log_f.write(line + "\n")
            log_f.flush()

    elapsed = time.time() - t0
    report = {
        "elapsed_sec": round(elapsed, 1),
        "datasets_in_scope": len(in_scope),
        "datasets_out_of_scope": len(out_scope),
        "rows_in": total_rows_in,
        "rows_out": total_rows_out,
        "keep_rate": round(total_rows_out / total_rows_in, 4) if total_rows_in else 0,
        "reason_counts": dict(combined_reasons.most_common()),
        "per_dataset_stats": per_dataset_stats,
        "dataset_errors": dataset_errors,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n======== SUMMARY ========")
    print(f"Elapsed: {elapsed/60:.1f} min")
    print(f"Datasets converted: {len(in_scope)}  (skipped, not chat-shaped: {len(out_scope)})")
    print(f"Rows in:  {total_rows_in:,}")
    print(f"Rows out: {total_rows_out:,}  ({100*total_rows_out/total_rows_in:.1f}% kept)" if total_rows_in else "Rows out: 0")
    print("Reasons:")
    for k, v in combined_reasons.most_common():
        print(f"  {v:12,}  {k}")
    print(f"\nReport written to {REPORT_PATH}")
    print(f"Skipped-dataset list written to {SKIPPED_REPORT_PATH}")


if __name__ == "__main__":
    main()
