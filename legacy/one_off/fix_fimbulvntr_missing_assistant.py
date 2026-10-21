"""One-off fix for fimbulvntr__deepseek-r1-traces-no-cjk -- same bug class as
fix_missing_assistant_turn.py's 8 datasets, different column names.

Bug: raw parquet's 'messages' column only ever holds system+user (confirmed:
100% of 271,759 rows end on 'user' in the current output, found via a
structural role-sequence audit). The real reply is split across two separate
columns: 'reasoning' (the model's chain-of-thought, confirmed it does NOT
already restate the final answer -- checked several real rows) and 'answer'
(the clean final response). Appended as one assistant message with
content=answer, reasoning_content=reasoning, matching this project's
established reasoning_content convention (the real GLM chat template
handles that field specially, verified earlier this session).
"""

import os
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

import sys as _sys; _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); from sft_pipeline.adapters.traces.base import convert_message, to_plain

SRC = Path(os.environ.get("SFTPIPE_PATH", "."))
OUT_DIR = Path(os.environ.get("SFTPIPE_PATH", "."))
NAME = "fimbulvntr__deepseek-r1-traces-no-cjk"


def main():
    files = sorted(SRC.glob(f"{NAME}*.parquet"))
    assert files, f"no source parquet found for {NAME}"

    out_path = OUT_DIR / f"{NAME}.jsonl"
    backup_path = OUT_DIR / f"{NAME}.jsonl.backup_before_assistant_fix"
    if out_path.exists() and not backup_path.exists():
        shutil.copy2(out_path, backup_path)

    rows_in = rows_out = skipped = 0
    with open(out_path, "w") as out_f:
        for fp in files:
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(batch_size=2000, columns=["messages", "reasoning", "answer"]):
                for row in batch.to_pylist():
                    rows_in += 1
                    msgs_raw = to_plain(row.get("messages"))
                    if not msgs_raw:
                        skipped += 1
                        continue
                    out_msgs = [cm for m in msgs_raw if (cm := convert_message(m)) is not None]
                    if not out_msgs:
                        skipped += 1
                        continue

                    answer = row.get("answer")
                    reasoning = row.get("reasoning")
                    if not answer:
                        skipped += 1
                        continue
                    assistant_msg = {"role": "assistant", "content": answer}
                    if reasoning:
                        assistant_msg["reasoning_content"] = reasoning

                    out_msgs.append(assistant_msg)
                    out_f.write(json.dumps({"messages": out_msgs}, ensure_ascii=False) + "\n")
                    rows_out += 1

    print(f"{NAME}: rows_in={rows_in} rows_out={rows_out} skipped={skipped}")


if __name__ == "__main__":
    main()
