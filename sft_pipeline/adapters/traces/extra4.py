"""qa_reasoning_style batch 3: the 'templated_response' family (12 datasets).

All 12 datasets carry a 'templated_response' column that is already a
complete ChatML-formatted conversation (<|im_start|>system...<|im_start|>user
...<|im_start|>assistant...<|im_end|>), ending in the model's own \\boxed{}
final answer -- reused via the same parse_chatml()/convert_message() path
already tested elsewhere in this project.

Each dataset also carries a real, explicit per-row correctness signal (not
guessed): 'Correct' (10 AIME datasets) or 'matches_solution' (2 s1k
datasets, genuinely mixed True/False -- 338 True / 162 False in a 500-row
check, so this is a real filter, not a no-op). Only rows where that field is
True are converted, same principle as the sam-12labs batch.
"""

import os
import json
from pathlib import Path

from .base import parse_chatml, convert_message

SRC = Path(os.environ.get("SFTPIPE_SRC", "."))   # raw source folder (set by sft_pipeline.convert --src)
OUT_DIR = Path(os.environ.get("SFTPIPE_OUT", "out"))   # output folder (set by sft_pipeline.convert --out)

DATASETS = [
    ("jonathanyin__aime_1983_2023_deepseek-r1-distill-qwen-1.5b_traces", "Correct"),
    ("jonathanyin__aime_1983_2023_deepseek-r1-distill-qwen-14b_traces", "Correct"),
    ("jonathanyin__aime_1983_2023_deepseek-r1-distill-qwen-7b_traces", "Correct"),
    ("jonathanyin__aime_1983_2023_deepseek-r1_traces", "Correct"),
    ("jonathanyin__aime_1983_2023_grok-3-mini-high_traces", "Correct"),
    ("jonathanyin__aime_1983_2023_grok-3-mini-high_traces_16392_r1_formatted", "Correct"),
    ("jonathanyin__aime_1983_2023_grok-3-mini-high_traces_32768_r1_formatted", "Correct"),
    ("jonathanyin__aime_1983_2023_grok-3-mini-high_traces_r1_formatted", "Correct"),
    ("jonathanyin__aime_1983_2023_qwq-32b_fcs_traces", "Correct"),
    ("jonathanyin__aime_1983_2023_qwq-32b_traces", "Correct"),
    ("jonathanyin__s1k_qwq-32b_correct_reasoning_traces", "matches_solution"),
    ("jonathanyin__s1k_qwq-32b_reasoning_traces", "matches_solution"),
]


def convert_one(name, correct_col):
    import pyarrow.parquet as pq

    fpaths = sorted(SRC.glob(f"{name}*.parquet"))
    out_path = OUT_DIR / f"{name}.jsonl"
    rows_in = rows_out = 0
    with open(out_path, "w") as out_f:
        for fpath in fpaths:
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=500, columns=["templated_response", correct_col]):
                for row in batch.to_pylist():
                    rows_in += 1
                    if not row.get(correct_col):
                        continue
                    tr = row.get("templated_response")
                    if not tr:
                        continue
                    msgs = parse_chatml(tr)
                    if not msgs:
                        continue
                    out_msgs = [cm for m in msgs if (cm := convert_message(m)) is not None]
                    if not out_msgs or all(m["role"] == "system" for m in out_msgs):
                        continue
                    out_f.write(json.dumps({"messages": out_msgs}, ensure_ascii=False) + "\n")
                    rows_out += 1
    if rows_out == 0:
        out_path.unlink(missing_ok=True)
    return {"dataset": name, "rows_in": rows_in, "rows_out": rows_out}


def main():
    for name, col in DATASETS:
        r = convert_one(name, col)
        print(f"{r['dataset']}: rows_in={r['rows_in']} rows_out={r['rows_out']}")


if __name__ == "__main__":
    main()
