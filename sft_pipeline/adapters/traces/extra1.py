"""Convert 3 more datasets that were originally bucketed as "not chat-shaped"
by column-name alone, but turned out (checked against real rows, not
guessed) to be real, convertible chat data:

- Arc-Intelligence__tau2-mms-teacher-traces: the 'prompt' column is actually
  a full messages list (system/user/assistant/tool roles, real structured
  tool_calls, same exact shape as the already-verified 285-dataset
  pipeline) sitting under a non-standard column name. 'response' is empty
  in all 243 rows and is ignored. Reuses convert_row() unchanged.
- shaafsalman__medical-reasoning-traces-11k: genuine Alpaca-style triples --
  instruction (2 unique values across 11,175 rows, a shared system prompt) +
  input (the real question) + output (the answer, with <think> reasoning
  inline in the text) -> system/user/assistant.
- jkazdan__qwen_reasoning_traces: a DPO preference-pair dataset (chosen vs
  rejected, both full ChatML text). Only 'chosen' (the correct answer) is
  used here -- 'rejected' is preference-training data, not something plain
  SFT needs, and is intentionally dropped, per direct sign-off with the
  user. Reuses parse_chatml() + convert_message() unchanged.
"""

import os
import json
from pathlib import Path

import pyarrow.parquet as pq

from .base import convert_row, convert_message, parse_chatml, iter_column_values

SRC = Path(os.environ.get("SFTPIPE_SRC", "."))   # raw source folder (set by sft_pipeline.convert --src)
OUT_DIR = Path(os.environ.get("SFTPIPE_OUT", "out"))   # output folder (set by sft_pipeline.convert --out)


def _shards(dataset_name):
    """Glob for ALL shard files of a dataset, not just the bare filename --
    some datasets (e.g. jkazdan__qwen_reasoning_traces) are split across a
    base file plus one or more '_<N>.parquet' shards, and assuming a single
    file silently drops the rest. Caught via a real cross-check after this
    function was added, not guessed in advance."""
    return sorted(SRC.glob(f"{dataset_name}*.parquet"))


def convert_arc_intelligence():
    name = "Arc-Intelligence__tau2-mms-teacher-traces"
    out_path = OUT_DIR / f"{name}.jsonl"
    rows_in = rows_out = 0
    with open(out_path, "w") as out_f:
        for fpath in _shards(name):
            for val in iter_column_values(fpath, "prompt"):
                rows_in += 1
                result, _reason = convert_row(val)
                if result is not None:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    rows_out += 1
    if rows_out == 0:
        out_path.unlink(missing_ok=True)
    return {"dataset": out_path.stem, "rows_in": rows_in, "rows_out": rows_out}


def convert_shaafsalman():
    name = "shaafsalman__medical-reasoning-traces-11k"
    out_path = OUT_DIR / f"{name}.jsonl"
    rows_in = rows_out = 0
    with open(out_path, "w") as out_f:
        for fpath in _shards(name):
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=1000, columns=["instruction", "input", "output"]):
                for row in batch.to_pylist():
                    rows_in += 1
                    instr, inp, out = row.get("instruction"), row.get("input"), row.get("output")
                    if not out:
                        continue
                    msgs = []
                    if instr:
                        msgs.append({"role": "system", "content": instr})
                    if inp:
                        msgs.append({"role": "user", "content": inp})
                    msgs.append({"role": "assistant", "content": out})
                    if len(msgs) < 2:
                        continue
                    out_f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                    rows_out += 1
    if rows_out == 0:
        out_path.unlink(missing_ok=True)
    return {"dataset": out_path.stem, "rows_in": rows_in, "rows_out": rows_out}


def convert_jkazdan():
    name = "jkazdan__qwen_reasoning_traces"
    out_path = OUT_DIR / f"{name}.jsonl"
    rows_in = rows_out = 0
    with open(out_path, "w") as out_f:
        for fpath in _shards(name):
            for val in iter_column_values(fpath, "chosen"):
                rows_in += 1
                if not val:
                    continue
                msgs = parse_chatml(val)
                if not msgs:
                    continue
                out_msgs = [cm for m in msgs if (cm := convert_message(m)) is not None]
                if not out_msgs or all(m["role"] == "system" for m in out_msgs):
                    continue
                out_f.write(json.dumps({"messages": out_msgs}, ensure_ascii=False) + "\n")
                rows_out += 1
    if rows_out == 0:
        out_path.unlink(missing_ok=True)
    return {"dataset": out_path.stem, "rows_in": rows_in, "rows_out": rows_out}


def main():
    for fn in (convert_arc_intelligence, convert_shaafsalman, convert_jkazdan):
        r = fn()
        print(f"{r['dataset']}: rows_in={r['rows_in']} rows_out={r['rows_out']}")


if __name__ == "__main__":
    main()
