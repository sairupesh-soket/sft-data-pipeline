"""other_custom batch 3: mlfoundations pdf_science, valteu, jkazdan retail.

mlfoundations-dev__pdf_science_questions_verified_r1_traces__2_24: has a
real judge_correct boolean; only the '_verified_' variant is used (the
sibling '_verifiable_' variant has no judge fields at all -- same
judged-vs-non-judged duplication pattern already handled for reasoning-proj,
skip the unjudged one).

valteu__reasoning-traces: an English->Chinese financial-text translation
task with two independent model completions (deepseek-reasoner, Qwen/QwQ-32B),
no correctness field but no strong reason to distrust either (translation
quality, not a single-right-answer factual claim like the medical/math
cases that needed filtering) -- both used, giving 2 rows per source row.

jkazdan__retail-gpt-4o-traces: 'traj' is a full agent trajectory in the
identical role/content/tool_calls shape already verified elsewhere in this
project -- reused convert_row() unchanged. Filtered by reward == 1.0 (a
real, genuinely mixed task-success signal: 27 succeeded / 23 failed in a
50-row check).

Excluded from this round (real reasons):
- juniworld__prompt_inference_traces: only has 'prompt' + reference URL
  list, no response/answer field of any kind exists -- nothing to build an
  assistant turn from.
- dineshyv__llamastack-traces: every field's value is literally the string
  "test" -- placeholder/dummy data, not real content.
"""

import os
import json
from pathlib import Path

import pyarrow.parquet as pq

from .base import convert_row

SRC = Path(os.environ.get("SFTPIPE_SRC", "."))   # raw source folder (set by sft_pipeline.convert --src)
OUT_DIR = Path(os.environ.get("SFTPIPE_OUT", "out"))   # output folder (set by sft_pipeline.convert --out)


def _shards(name):
    return sorted(SRC.glob(f"{name}*.parquet"))


def _write(out_path, rows_iter):
    rows_in = rows_out = 0
    with open(out_path, "w") as out_f:
        for rows_in_inc, msgs in rows_iter:
            rows_in += rows_in_inc
            if msgs is None:
                continue
            out_f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            rows_out += 1
    if rows_out == 0:
        out_path.unlink(missing_ok=True)
    return rows_in, rows_out


def convert_pdf_science():
    name = "mlfoundations-dev__pdf_science_questions_verified_r1_traces__2_24"
    out_path = OUT_DIR / f"{name}.jsonl"

    def gen():
        for fpath in _shards(name):
            pf = pq.ParquetFile(fpath)
            cols = ["extracted_question", "extracted_answer_choices", "r1_solution", "judge_correct"]
            for batch in pf.iter_batches(batch_size=300, columns=cols):
                for row in batch.to_pylist():
                    if not row.get("judge_correct"):
                        yield 1, None
                        continue
                    q, choices, sol = row.get("extracted_question"), row.get("extracted_answer_choices"), row.get("r1_solution")
                    if not q or not sol:
                        yield 1, None
                        continue
                    user_content = q
                    if choices:
                        user_content += "\n" + "\n".join(choices)
                    yield 1, [{"role": "user", "content": user_content}, {"role": "assistant", "content": sol}]

    rin, rout = _write(out_path, gen())
    return {"dataset": name, "rows_in": rin, "rows_out": rout}


def convert_valteu():
    name = "valteu__reasoning-traces"
    out_path = OUT_DIR / f"{name}.jsonl"
    model_pairs = [("deepseek-reasoner_response", "deepseek-reasoner_reasoning"), ("Qwen/QwQ-32B_response", "Qwen/QwQ-32B_reasoning")]

    def gen():
        for fpath in _shards(name):
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=100, columns=["question"] + [c for pair in model_pairs for c in pair]):
                for row in batch.to_pylist():
                    q = row.get("question")
                    if not q:
                        yield 1, None
                        continue
                    for resp_col, reasoning_col in model_pairs:
                        resp, reasoning = row.get(resp_col), row.get(reasoning_col)
                        if not resp:
                            continue
                        content = f"{reasoning}\n\n{resp}" if reasoning else resp
                        yield 0, [{"role": "user", "content": q}, {"role": "assistant", "content": content}]
                    yield 1, None

    rin, rout = _write(out_path, gen())
    return {"dataset": name, "rows_in": rin, "rows_out": rout}


def convert_jkazdan_retail():
    name = "jkazdan__retail-gpt-4o-traces"
    out_path = OUT_DIR / f"{name}.jsonl"

    def gen():
        for fpath in _shards(name):
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=50, columns=["traj", "reward"]):
                for row in batch.to_pylist():
                    if row.get("reward") != 1.0:
                        yield 1, None
                        continue
                    result, _reason = convert_row(row.get("traj"))
                    yield 1, result["messages"] if result else None

    rin, rout = _write(out_path, gen())
    return {"dataset": name, "rows_in": rin, "rows_out": rout}


def main():
    for fn in (convert_pdf_science, convert_valteu, convert_jkazdan_retail):
        r = fn()
        print(f"{r['dataset']}: rows_in={r['rows_in']} rows_out={r['rows_out']}")


if __name__ == "__main__":
    main()
