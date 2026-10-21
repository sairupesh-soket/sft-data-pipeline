"""qa_reasoning_style batch 5 (tail): Wendong-Fan, TAUR-dev, update0909.

Wendong-Fan__math_problem_traces_test(2): plain problem/solution pairs,
manually verified correct and simple. No filter needed.

TAUR-dev__STEPS__s1__r1_traces: carries TWO independent model attempts per
question (gemini_attempt, deepseek_attempt), each with its own real,
genuinely-mixed grade field (gemini_grade: 50 Yes/44 No; deepseek_grade: 60
Yes/34 No out of 94 rows). Both attempts are used when graded Yes -- can
produce 0-2 output rows per source row.

update0909__cure-bench-reasoning-traces: tiny (11 rows), mixed
multiple-choice (letter) and free-text medical answers. Spot-checked
several real rows: letter answers match the generated_solution's
conclusion exactly; free-text answers were visibly aligned in substance
with the generated_solution's tail even where an exact substring match
failed (same class of formatting-only false positive already found and
rejected for zeju/capybara-math). Converted directly, skipping the 1 row
with a null generated_solution.

Excluded from this qa_reasoning_style pass (not silently dropped, real
reasons):
- Beanbagdzf__graph_problem_traces_test1 (190 rows): 'solution' is a raw
  Python tuple/set repr (e.g. "({0, 2, 4}, [...])"), not natural language --
  not a realistic assistant chat response.
- yosubshin__cosyn-with-traces: contains real image bytes (multimodal),
  same reason as the walton-*-cosyn-traces datasets excluded earlier.
- immindich__r1-qwen-7b-corrupted-traces-stats (32 rows): a rollout-
  corruption robustness study dataset, not intended as clean SFT data.
- achinta3__rank1_tempora_traces(_full) (14,457 rows): NOT a QA dataset --
  'reasoning' is a per-document retrieval-relevance judgment (ends in
  <relevance>0 or 1</relevance>), a fundamentally different task type
  (reranking) than question-answering. Left for a separate decision on
  whether/how to build that task type rather than force-fit it here.
"""

import os
import json
from pathlib import Path

import pyarrow.parquet as pq

SRC = Path(os.environ.get("SFTPIPE_SRC", "."))   # raw source folder (set by sft_pipeline.convert --src)
OUT_DIR = Path(os.environ.get("SFTPIPE_OUT", "out"))   # output folder (set by sft_pipeline.convert --out)


def _shards(name):
    return sorted(SRC.glob(f"{name}*.parquet"))


def convert_wendong():
    results = []
    for name in ["Wendong-Fan__math_problem_traces_test", "Wendong-Fan__math_problem_traces_test2"]:
        out_path = OUT_DIR / f"{name}.jsonl"
        rows_in = rows_out = 0
        with open(out_path, "w") as out_f:
            for fpath in _shards(name):
                pf = pq.ParquetFile(fpath)
                for batch in pf.iter_batches(batch_size=100, columns=["problem", "solution"]):
                    for row in batch.to_pylist():
                        rows_in += 1
                        p, s = row.get("problem"), row.get("solution")
                        if not p or not s:
                            continue
                        msgs = [{"role": "user", "content": p}, {"role": "assistant", "content": s}]
                        out_f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                        rows_out += 1
        if rows_out == 0:
            out_path.unlink(missing_ok=True)
        results.append({"dataset": name, "rows_in": rows_in, "rows_out": rows_out})
    return results


def convert_taur():
    name = "TAUR-dev__STEPS__s1__r1_traces"
    out_path = OUT_DIR / f"{name}.jsonl"
    rows_in = rows_out = 0
    with open(out_path, "w") as out_f:
        for fpath in _shards(name):
            pf = pq.ParquetFile(fpath)
            cols = ["question", "gemini_attempt", "gemini_grade", "deepseek_attempt", "deepseek_grade"]
            for batch in pf.iter_batches(batch_size=100, columns=cols):
                for row in batch.to_pylist():
                    rows_in += 1
                    q = row.get("question")
                    if not q:
                        continue
                    for attempt_col, grade_col in [("gemini_attempt", "gemini_grade"), ("deepseek_attempt", "deepseek_grade")]:
                        if row.get(grade_col) != "Yes":
                            continue
                        attempt = row.get(attempt_col)
                        if not attempt:
                            continue
                        msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": attempt}]
                        out_f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                        rows_out += 1
    if rows_out == 0:
        out_path.unlink(missing_ok=True)
    return {"dataset": name, "rows_in": rows_in, "rows_out": rows_out}


def convert_update0909():
    name = "update0909__cure-bench-reasoning-traces"
    out_path = OUT_DIR / f"{name}.jsonl"
    rows_in = rows_out = 0
    with open(out_path, "w") as out_f:
        for fpath in _shards(name):
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=50, columns=["problem", "generated_solution"]):
                for row in batch.to_pylist():
                    rows_in += 1
                    p, s = row.get("problem"), row.get("generated_solution")
                    if not p or not s:
                        continue
                    msgs = [{"role": "user", "content": p}, {"role": "assistant", "content": s}]
                    out_f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                    rows_out += 1
    if rows_out == 0:
        out_path.unlink(missing_ok=True)
    return {"dataset": name, "rows_in": rows_in, "rows_out": rows_out}


def main():
    for r in convert_wendong():
        print(f"{r['dataset']}: rows_in={r['rows_in']} rows_out={r['rows_out']}")
    r = convert_taur()
    print(f"{r['dataset']}: rows_in={r['rows_in']} rows_out={r['rows_out']}")
    r = convert_update0909()
    print(f"{r['dataset']}: rows_in={r['rows_in']} rows_out={r['rows_out']}")


if __name__ == "__main__":
    main()
