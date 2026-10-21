"""One-off fix for the "missing assistant turn" bug found by inspection.

Bug: for 8 datasets, the raw parquet's 'messages' column only ever holds
the system/user turns (this is a LangSmith/Langfuse-style API-call log,
where 'messages' = what was SENT to the model). The real reply lives in a
separate 'assistant_message' column (or 'response', for cfahlgren1__traces)
and, for tool-calling rows, in a separate 'tool_calls' column.
traces_to_openai_sft.py's convert_row() only ever reads the 'messages'
column, so these 8 datasets were converted with the assistant's real reply
silently dropped -- every output row ended in a lone 'user' turn.

Confirmed (see checks run before writing this): across all 218 raw rows in
these 8 datasets, 'messages' ALWAYS ends in a 'user' turn (never already
has a trailing assistant turn), so it is always safe to append one more
message rather than risk a duplicate.

One dataset that looked similar, thomasgauthier__pydantic-ai-traces, is
NOT included here: all 21 of its rows have messages=None (no captured user
input at all, only the tool-call output), so there is no real conversation
to reconstruct. The main pipeline already correctly produces 0 rows for it.

This script re-reads the raw parquet for the 8 affected datasets, builds
the assistant turn from 'assistant_message'/'response' (+ 'tool_calls' when
present), and rewrites just those 8 output files. The originals are backed
up first, never overwritten blind.
"""

from __future__ import annotations

import os
import glob
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

import sys as _sys; _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); from sft_pipeline.adapters.traces.base import convert_message, normalize_tool_calls, to_plain

SRC = Path(os.environ.get("SFTPIPE_PATH", "."))
OUT_DIR = Path(os.environ.get("SFTPIPE_PATH", "."))

# (dataset_prefix, assistant_text_column, has_tool_calls_column)
DATASETS = [
    ("davidberenstein1957__gpt-4o-traces", "assistant_message", True),
    ("kabelomalapane__gpt-4o-traces", "assistant_message", True),
    ("pprajan__gpt-4o-function-calling-traces", "assistant_message", True),
    ("pprajan__gpt-4o-traces", "assistant_message", True),
    ("cfahlgren1__gpt-4o-function-calling-traces", "assistant_message", True),
    ("cfahlgren1__gpt-4o-mini-vision-traces", "assistant_message", True),
    ("PierreColombo__gpt-4o-traces", "assistant_message", True),
    ("cfahlgren1__traces", "response", False),
]


def build_assistant_message(row, text_col, has_tool_calls):
    content = row.get(text_col)
    if isinstance(content, str):
        content = content.strip() or None
    tool_calls_raw = row.get("tool_calls") if has_tool_calls else None
    if isinstance(tool_calls_raw, str):
        # this column is stored as a JSON-serialized string, not a native
        # list<struct> -- confirmed via pyarrow schema (`tool_calls: string`)
        try:
            tool_calls_raw = json.loads(tool_calls_raw)
        except Exception:
            tool_calls_raw = None
    tool_calls = normalize_tool_calls(tool_calls_raw) if tool_calls_raw else None
    if content is None and not tool_calls:
        return None  # e.g. PierreColombo's 1 row with error="'Stream' object..." and no reply at all
    out = {"role": "assistant", "content": content if content is not None else ""}
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def fix_dataset(prefix, text_col, has_tool_calls):
    files = sorted(glob.glob(str(SRC / f"{prefix}*.parquet")))
    assert files, f"no source parquet found for {prefix}"

    out_path = OUT_DIR / f"{prefix}.jsonl"
    if out_path.exists():
        backup_path = OUT_DIR / f"{prefix}.jsonl.backup_before_assistant_fix"
        if not backup_path.exists():
            shutil.copy2(out_path, backup_path)

    rows_in = 0
    rows_out = 0
    skipped_no_reply = 0
    skipped_no_user_content = 0

    with open(out_path, "w") as out_f:
        for fp in files:
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(batch_size=1000):
                for row in batch.to_pylist():
                    rows_in += 1
                    msgs_raw = to_plain(row.get("messages"))
                    if not msgs_raw:
                        skipped_no_user_content += 1
                        continue
                    out_msgs = []
                    for m in msgs_raw:
                        cm = convert_message(m)
                        if cm is not None:
                            out_msgs.append(cm)
                    if not out_msgs:
                        skipped_no_user_content += 1
                        continue

                    assistant_msg = build_assistant_message(row, text_col, has_tool_calls)
                    if assistant_msg is None:
                        skipped_no_reply += 1
                        continue

                    out_msgs.append(assistant_msg)
                    out_f.write(json.dumps({"messages": out_msgs}, ensure_ascii=False) + "\n")
                    rows_out += 1

    return {
        "prefix": prefix,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "skipped_no_reply": skipped_no_reply,
        "skipped_no_user_content": skipped_no_user_content,
    }


def main():
    results = []
    for prefix, text_col, has_tool_calls in DATASETS:
        res = fix_dataset(prefix, text_col, has_tool_calls)
        results.append(res)
        print(
            f"{res['prefix']:55s} in={res['rows_in']:4d} out={res['rows_out']:4d} "
            f"skip_no_reply={res['skipped_no_reply']} skip_no_user={res['skipped_no_user_content']}"
        )

    total_in = sum(r["rows_in"] for r in results)
    total_out = sum(r["rows_out"] for r in results)
    print(f"\nTotal: {total_in} rows in, {total_out} rows out "
          f"({total_in - total_out} skipped: no valid reply or no valid user content)")


if __name__ == "__main__":
    main()
