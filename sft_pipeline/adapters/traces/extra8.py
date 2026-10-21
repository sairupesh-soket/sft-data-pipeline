"""other_custom batch 2: hoanganhpham medical family (partial), CogniSQL,
nolabs.

hoanganhpham__Medical-Train-{PubMedQA,MedMCQA,MedReason}-hard-with-traces:
same multi-attempt pattern as sam-12labs/TAUR-dev -- 'generations' is a list
of independent model attempts, 'generations_correctness' a matching list of
real booleans (confirmed genuinely mixed on real rows, e.g. [True, False]).
Only correct attempts used.

hoanganhpham__HealthBench_Knowledge_Question_250K_traces: 'trace' is
ALREADY a complete, real multi-turn conversation (verified real roles:
user/assistant/user/assistant, a 2-step summarize-then-generate-question
pipeline) -- used directly, no reconstruction needed.

CogniSQL__Reasoning_Traces: 'reasoning' already contains the complete
<reasoning>...</reasoning><answer>SQL</answer> response (verified the
<answer> tag's SQL matches the row's own ground-truth SQL field on a real
example) -- used as-is. User content combines the schema ('prompt') and the
question, since the schema is required to answer.

nolabs__deepfabric-devops-reasoning-traces: chain_of_thought + final_answer
combined as the assistant response (both read as genuine, complete content
on inspection).

NOT included in this batch (deferred, not silently dropped):
- hoanganhpham__Miriad-traces-and-rewards: uses a letter-grade judge
  ('4o-as-judge': B/B/B/C) rather than a boolean, and pass_rate is 0.0 for
  82% of a 500-row sample -- grading rubric not understood well enough yet
  to filter correctly.
- hoanganhpham__HealthBench_Knowledge_Question_250K_MCQ_with_traces: 'traces'
  is a list of raw candidate answer strings with no correctness field
  visible -- needs more digging before choosing one attempt.
- Jianyuan1__reasoing-traces: reasoning_steps is an MCTS search tree
  (text/mc_value/children, recursively nested), not a flat reasoning trace
  -- needs real tree-traversal logic to extract a clean single path.
- akseljoonas__smollm3-traces: uses non-standard 'tool-call'/'tool-response'
  role labels (hyphenated, not the structured tool_calls field the rest of
  this project's convert_message() expects) -- needs its own custom mapping,
  only 128 rows, not worth rushing.

Also excluded from other_custom this round (real reasons):
- the-acorn-ai__textarena-player-game-traces: RL game-playing data
  (SpellingBee word game against an opponent), not chat/QA.
- valeriol29__mooncake-traces: LLM-serving cache-hit/miss infrastructure
  tracing (hash_ids/input_length/output_length/timestamp), not chat data.
- DatOneDue__GRPO_Hourly_Think_Traces: no natural-language question field
  exists at all, only numeric stock technical indicators -- building a user
  turn would mean fabricating a prompt not present in the source.
- sbhambr1__sft_dataset_r1_traces_3k: broken download -- the file only
  contains a git-lfs pointer ("version https://git-lfs.github.com/spec/v1"),
  not real data.
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


def convert_medical_mcq_multiattempt(name):
    out_path = OUT_DIR / f"{name}.jsonl"

    def gen():
        for fpath in _shards(name):
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=300, columns=["problem", "generations", "generations_correctness"]):
                for row in batch.to_pylist():
                    problem = row.get("problem")
                    gens, corrects = row.get("generations") or [], row.get("generations_correctness") or []
                    if not problem or len(gens) != len(corrects):
                        yield 1, None
                        continue
                    for gen_text, is_correct in zip(gens, corrects):
                        if not is_correct or not gen_text:
                            continue
                        yield 0, [{"role": "user", "content": problem}, {"role": "assistant", "content": gen_text}]
                    yield 1, None

    rin, rout = _write(out_path, gen())
    return {"dataset": name, "rows_in": rin, "rows_out": rout}


def convert_healthbench_trace():
    name = "hoanganhpham__HealthBench_Knowledge_Question_250K_traces"
    out_path = OUT_DIR / f"{name}.jsonl"

    def gen():
        for fpath in _shards(name):
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=300, columns=["trace"]):
                for row in batch.to_pylist():
                    trace = row.get("trace")
                    if not trace:
                        yield 1, None
                        continue
                    yield 1, list(trace)

    rin, rout = _write(out_path, gen())
    return {"dataset": name, "rows_in": rin, "rows_out": rout}


def convert_cognisql():
    name = "CogniSQL__Reasoning_Traces"
    out_path = OUT_DIR / f"{name}.jsonl"

    def gen():
        for fpath in _shards(name):
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=500, columns=["prompt", "question", "reasoning"]):
                for row in batch.to_pylist():
                    schema, question, reasoning = row.get("prompt"), row.get("question"), row.get("reasoning")
                    if not question or not reasoning:
                        yield 1, None
                        continue
                    user_content = f"{schema}\n\n{question}" if schema else question
                    yield 1, [{"role": "user", "content": user_content}, {"role": "assistant", "content": reasoning}]

    rin, rout = _write(out_path, gen())
    return {"dataset": name, "rows_in": rin, "rows_out": rout}


def convert_nolabs():
    name = "nolabs__deepfabric-devops-reasoning-traces"
    out_path = OUT_DIR / f"{name}.jsonl"

    def gen():
        for fpath in _shards(name):
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=500, columns=["question", "chain_of_thought", "final_answer"]):
                for row in batch.to_pylist():
                    q, cot, ans = row.get("question"), row.get("chain_of_thought"), row.get("final_answer")
                    if not q or not ans:
                        yield 1, None
                        continue
                    content = f"{cot}\n\n{ans}" if cot else ans
                    yield 1, [{"role": "user", "content": q}, {"role": "assistant", "content": content}]

    rin, rout = _write(out_path, gen())
    return {"dataset": name, "rows_in": rin, "rows_out": rout}


def main():
    for name in [
        "hoanganhpham__Medical-Train-PubMedQA-hard-with-traces",
        "hoanganhpham__Medical-Train-MedMCQA-hard-with-traces",
        "hoanganhpham__Medical-Train-MedReason-with-traces",
    ]:
        r = convert_medical_mcq_multiattempt(name)
        print(f"{r['dataset']}: rows_in={r['rows_in']} rows_out={r['rows_out']}")
    for fn in (convert_healthbench_trace, convert_cognisql, convert_nolabs):
        r = fn()
        print(f"{r['dataset']}: rows_in={r['rows_in']} rows_out={r['rows_out']}")


if __name__ == "__main__":
    main()
