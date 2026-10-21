"""qa_reasoning_style batch 1: convert zeju-0727__dyve_plus_120k_traces only.

cot-leaderboard__cot-eval-traces-2.0 (the other big dataset checked in this
batch, 3.75M rows) is deliberately NOT converted: manually reading real rows
showed its 'reasoning_trace' frequently reaches a DIFFERENT final answer than
the dataset's own labeled ground truth (e.g. ground truth C, trace concludes
"Answer E is the best choice"). It's an eval log of a specific weak model
(wenbopan/Faro-Yi-9B-DPO per its own config_data field) being tested on hard
LSAT-style reasoning, not curated correct reasoning -- converting it would
train wrong reasoning into the model at scale. Left out of this pipeline
entirely (not added to skipped_datasets_report.json's generic list, since
the reason is a correctness problem, not a schema-shape problem).

zeju-0727__dyve_plus_120k_traces: 'process' is a complete, self-contained
<think>-style answer that already includes the real final answer (confirmed
by manually reading several rows; a naive substring check flagged false
mismatches purely from notation differences like arctan2(y,x) vs
arctan(y/x), not missing content). 'answer' (a compact canonical form of the
same result) and 'label' (a per-step correctness signal for a separate
process-reward-model use case) are both redundant with what's already
inside 'process' and are dropped.
"""

import os
import json
from pathlib import Path

import pyarrow.parquet as pq

SRC = Path(os.environ.get("SFTPIPE_SRC", "."))   # raw source folder (set by sft_pipeline.convert --src)
OUT_DIR = Path(os.environ.get("SFTPIPE_OUT", "out"))   # output folder (set by sft_pipeline.convert --out)


def convert_zeju():
    fpaths = sorted(SRC.glob("zeju-0727__dyve_plus_120k_traces*.parquet"))
    out_path = OUT_DIR / "zeju-0727__dyve_plus_120k_traces.jsonl"
    rows_in = rows_out = 0
    with open(out_path, "w") as out_f:
        for fpath in fpaths:
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=2000, columns=["question", "process"]):
                for row in batch.to_pylist():
                    rows_in += 1
                    q, proc = row.get("question"), row.get("process")
                    if not q or not proc:
                        continue
                    msgs = [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": proc},
                    ]
                    out_f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                    rows_out += 1
    if rows_out == 0:
        out_path.unlink(missing_ok=True)
    return {"dataset": out_path.stem, "rows_in": rows_in, "rows_out": rows_out}


def main():
    r = convert_zeju()
    print(f"{r['dataset']}: rows_in={r['rows_in']} rows_out={r['rows_out']}")


if __name__ == "__main__":
    main()
