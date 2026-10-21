"""other_custom batch 1: 6 direct datasets + the reasoning-proj family (25).

anonymousubmission__multilingual-medical-reasoning-traces: multiple-choice
medical QA (Italian). Filtered by reasoning_parsed_answer == correct_option
(a real, pre-computed alignment field -- not a heuristic). Note:
'formatted_similar_chunks_dense' is NOT used -- spot-checked and found to be
misaligned with the row's own question (same class of scrambled-auxiliary-
field issue found earlier in II-Vietnam), but it isn't needed for this
mapping anyway.

punwaiw__multi-subject-rlvr-final-reasoning-traces: 'question' is a
single-message list; reasoning_content already concludes with the same
answer as the separate 'text' field (verified on a real row). Uses
question[0].content + reasoning_content only.

shaafsalman__reasoning-traces-37k: problem/reasoning, verified the
reasoning ends in a real boxed conclusion. No correctness field available;
converted directly, same standard as most of the original 285-dataset
pipeline.

mlfoundations-dev__SCP_40k_verified_traces: THREE independent model
attempts per problem (o1_solution, qwq_solution, r1_solution), each with
its own real correctness signal
(is_o1_solution_same_with_matched_solution, is_qwq_solution_same_with_
matched_solution, judge_correct respectively). Only correct attempts used,
same multi-attempt-filter principle as sam-12labs/TAUR-dev.

yourbench__yourbench_full_inference_traces: NOT a QA dataset -- input is a
system+user request to GENERATE quiz questions from source material,
output is the generated questions. A legitimate, different task type
(instruction-following content generation), not force-fit into a QA shape.
input/output are JSON-encoded strings; parsed directly.

reasoning-proj family (71 datasets total, heavily duplicated across 8
models x multiple processing stages): only the 24 '___judged_<domain>_
traces_original_<model>' datasets (math/logic/science x 8 models) plus the
1 standalone 'verified_numerical_answers_traces_numinamath' dataset are
used -- all 25 carry a real verifier_score field (confirmed genuinely mixed
on a real sample: 2283 True / 76 False out of 2359 for judged_math
QwQ-32B). The plain non-judged duplicates (same questions, no verifier_
score), the smaller 'filtered_math' subset, and the 'baseline_analyzed/
baseline_doubt_data/baseline_prepared' research-annotation variants are
intentionally skipped -- same underlying question content, no added
training value, and including them would just duplicate rows already
covered by the judged variants.
"""

import os
import json
from pathlib import Path

import pyarrow.parquet as pq

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


def convert_anonymousubmission():
    name = "anonymousubmission__multilingual-medical-reasoning-traces"
    out_path = OUT_DIR / f"{name}.jsonl"

    def gen():
        for fpath in _shards(name):
            pf = pq.ParquetFile(fpath)
            cols = ["full_question", "list_of_options", "reasoning", "correct_option", "reasoning_parsed_answer"]
            for batch in pf.iter_batches(batch_size=300, columns=cols):
                for row in batch.to_pylist():
                    if row.get("reasoning_parsed_answer") != row.get("correct_option"):
                        yield 1, None
                        continue
                    q, opts, reasoning = row.get("full_question"), row.get("list_of_options"), row.get("reasoning")
                    if not q or not reasoning:
                        yield 1, None
                        continue
                    user_content = q
                    if opts:
                        user_content += "\n" + "\n".join(f"{i+1}. {o}" for i, o in enumerate(opts))
                    yield 1, [{"role": "user", "content": user_content}, {"role": "assistant", "content": reasoning}]

    rin, rout = _write(out_path, gen())
    return {"dataset": name, "rows_in": rin, "rows_out": rout}


def convert_punwaiw():
    name = "punwaiw__multi-subject-rlvr-final-reasoning-traces"
    out_path = OUT_DIR / f"{name}.jsonl"

    def gen():
        for fpath in _shards(name):
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=500, columns=["question", "reasoning_content"]):
                for row in batch.to_pylist():
                    q_list, reasoning = row.get("question"), row.get("reasoning_content")
                    if not q_list or not reasoning:
                        yield 1, None
                        continue
                    user_msg = next((m for m in q_list if m.get("role") == "user"), None)
                    if not user_msg or not user_msg.get("content"):
                        yield 1, None
                        continue
                    yield 1, [{"role": "user", "content": user_msg["content"]}, {"role": "assistant", "content": reasoning}]

    rin, rout = _write(out_path, gen())
    return {"dataset": name, "rows_in": rin, "rows_out": rout}


def convert_shaafsalman37k():
    name = "shaafsalman__reasoning-traces-37k"
    out_path = OUT_DIR / f"{name}.jsonl"

    def gen():
        for fpath in _shards(name):
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=500, columns=["problem", "reasoning"]):
                for row in batch.to_pylist():
                    p, r = row.get("problem"), row.get("reasoning")
                    if not p or not r:
                        yield 1, None
                        continue
                    yield 1, [{"role": "user", "content": p}, {"role": "assistant", "content": r}]

    rin, rout = _write(out_path, gen())
    return {"dataset": name, "rows_in": rin, "rows_out": rout}


def convert_scp40k():
    name = "mlfoundations-dev__SCP_40k_verified_traces"
    out_path = OUT_DIR / f"{name}.jsonl"
    attempts = [
        ("o1_solution", "is_o1_solution_same_with_matched_solution"),
        ("qwq_solution", "is_qwq_solution_same_with_matched_solution"),
        ("r1_solution", "judge_correct"),
    ]

    def gen():
        for fpath in _shards(name):
            pf = pq.ParquetFile(fpath)
            cols = ["problem"] + [c for pair in attempts for c in pair]
            for batch in pf.iter_batches(batch_size=200, columns=cols):
                for row in batch.to_pylist():
                    problem = row.get("problem")
                    if not problem:
                        yield 1, None
                        continue
                    for sol_col, flag_col in attempts:
                        if not row.get(flag_col):
                            continue
                        sol = row.get(sol_col)
                        if not sol:
                            continue
                        yield 0, [{"role": "user", "content": problem}, {"role": "assistant", "content": sol}]
                    yield 1, None

    rin, rout = _write(out_path, gen())
    return {"dataset": name, "rows_in": rin, "rows_out": rout}


def convert_yourbench():
    name = "yourbench__yourbench_full_inference_traces"
    out_path = OUT_DIR / f"{name}.jsonl"

    def gen():
        for fpath in _shards(name):
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=200, columns=["input", "output"]):
                for row in batch.to_pylist():
                    try:
                        inp = json.loads(row["input"])
                        out = json.loads(row["output"])
                    except Exception:
                        yield 1, None
                        continue
                    msgs = inp.get("messages")
                    content = out.get("content")
                    if not msgs or not content:
                        yield 1, None
                        continue
                    msgs = list(msgs) + [{"role": out.get("role") or "assistant", "content": content}]
                    yield 1, msgs

    rin, rout = _write(out_path, gen())
    return {"dataset": name, "rows_in": rin, "rows_out": rout}


REASONING_PROJ_MODELS = [
    "DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-14B",
    "DeepSeek-R1-Distill-Qwen-32B", "DeepSeek-R1-Distill-Qwen-7B", "EXAONE-Deep-32B",
    "Llama-3.1-Nemotron-Nano-8B-v1", "Phi-4-reasoning-plus", "QwQ-32B",
]
REASONING_PROJ_DOMAINS = ["math", "logic", "science"]


def convert_reasoning_proj_one(name):
    out_path = OUT_DIR / f"{name}.jsonl"

    def gen():
        for fpath in _shards(name):
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=500, columns=["question", "answer_content", "verifier_score"]):
                for row in batch.to_pylist():
                    vs = row.get("verifier_score")
                    if not (vs == 1 or vs == 1.0):
                        yield 1, None
                        continue
                    q, a = row.get("question"), row.get("answer_content")
                    if not q or not a:
                        yield 1, None
                        continue
                    yield 1, [{"role": "user", "content": q}, {"role": "assistant", "content": a}]

    rin, rout = _write(out_path, gen())
    return {"dataset": name, "rows_in": rin, "rows_out": rout}


def convert_reasoning_proj_family():
    results = []
    for domain in REASONING_PROJ_DOMAINS:
        for model in REASONING_PROJ_MODELS:
            name = f"reasoning-proj___judged_{domain}_traces_original_{model}"
            if not _shards(name):
                continue
            results.append(convert_reasoning_proj_one(name))
    numinamath = "reasoning-proj__verified_numerical_answers_traces_numinamath"
    if _shards(numinamath):
        results.append(convert_reasoning_proj_one(numinamath))
    return results


def main():
    for fn in (convert_anonymousubmission, convert_punwaiw, convert_shaafsalman37k, convert_scp40k, convert_yourbench):
        r = fn()
        print(f"{r['dataset']}: rows_in={r['rows_in']} rows_out={r['rows_out']}")
    for r in convert_reasoning_proj_family():
        print(f"{r['dataset']}: rows_in={r['rows_in']} rows_out={r['rows_out']}")


if __name__ == "__main__":
    main()
