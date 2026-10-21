"""qa_reasoning_style batch 2: sam-12labs__DeepScaleR-Preview-Dataset_...

Unlike cot-leaderboard/II-Vietnam (excluded -- no reliable way to check
whether a generated response is actually correct), this dataset carries an
explicit, real per-attempt correctness signal:
'DeepSeek-R1-Distill-Qwen-32B_correct' is a list of booleans, one per
reasoning attempt, already computed by whoever built the dataset (confirmed
same list length as the reasoning/answer lists on 500 real rows, 0
mismatches). Only attempts marked True are used -- this is a real filter,
not a guess.

'DeepSeek-R1-Distill-Qwen-32B_reasoning[i]' already contains the model's
complete original chain-of-thought ending in the correct \\boxed{...} answer
(confirmed against the row's own ground-truth 'answer' field). The separate
'..._answer[i]' field is a cleaner, likely post-processed restatement of the
same solution and is NOT appended, to keep the authentic original generation
rather than a derived duplicate.

One source row can produce 0-8 output rows (one per correct attempt) -- this
is not data invention, just correctly unpacking an existing 1-to-many
structure (multiple independent correct solutions to the same problem).
"""

import os
import json
from pathlib import Path

import pyarrow.parquet as pq

SRC = Path(os.environ.get("SFTPIPE_SRC", "."))   # raw source folder (set by sft_pipeline.convert --src)
OUT_DIR = Path(os.environ.get("SFTPIPE_OUT", "out"))   # output folder (set by sft_pipeline.convert --out)

REASONING_COL = "DeepSeek-R1-Distill-Qwen-32B_reasoning"
CORRECT_COL = "DeepSeek-R1-Distill-Qwen-32B_correct"


def convert_sam12labs():
    name = "sam-12labs__DeepScaleR-Preview-Dataset_DeepSeek-R1-Distill-Qwen-32B_reasoning_traces"
    out_path = OUT_DIR / f"{name}.jsonl"
    fpaths = sorted(SRC.glob(f"{name}*.parquet"))
    rows_in = rows_out = 0
    with open(out_path, "w") as out_f:
        for fpath in fpaths:
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=1000, columns=["problem", REASONING_COL, CORRECT_COL]):
                for row in batch.to_pylist():
                    rows_in += 1
                    problem = row.get("problem")
                    reasonings = row.get(REASONING_COL) or []
                    corrects = row.get(CORRECT_COL) or []
                    if not problem or len(reasonings) != len(corrects):
                        continue
                    for reasoning, is_correct in zip(reasonings, corrects):
                        if not is_correct or not reasoning:
                            continue
                        msgs = [
                            {"role": "user", "content": problem},
                            {"role": "assistant", "content": reasoning},
                        ]
                        out_f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                        rows_out += 1
    if rows_out == 0:
        out_path.unlink(missing_ok=True)
    return {"dataset": out_path.stem, "source_rows_in": rows_in, "training_rows_out": rows_out}


def main():
    r = convert_sam12labs()
    print(f"{r['dataset']}: source_rows_in={r['source_rows_in']} training_rows_out={r['training_rows_out']}")


if __name__ == "__main__":
    main()
