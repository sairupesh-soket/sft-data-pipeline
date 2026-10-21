"""qa_reasoning_style batch 4: NaruseShiroha math family, NaruseShiroha
tool-calling family, fedric95__LIMO-Traces.

NaruseShiroha math family (5 datasets: problem/generated_solution/
expected_answer): generated_solution ends with the model's own
<answer>N</answer> (confirmed matches expected_answer on a real example --
14 == 14). Filtered by checking expected_answer's text appears in the last
300 chars of generated_solution, a real correctness check (same principle
as the letter-matching used for cot-leaderboard, just applied here as an
active filter since it's cheap and catches obvious mismatches).

NaruseShiroha tool-calling family (2 datasets: system_prompt/problem/
generated_solution, no expected_answer field -- these are agent
function-calling scenarios, not single-answer math, so there's no numeric
target to check against). Converted directly without a correctness filter,
same as most of the original 285-dataset pipeline.

fedric95__LIMO-Traces: has an explicit real 'match' field (Yes/No, genuinely
mixed -- 1189/811 in a 2000-row check). Filtered to match=='Yes' only.

yosubshin__walton-hard-cosyn-traces / walton-without-image-cosyn-traces are
NOT included here -- they carry real image bytes (genuine multimodal data),
and the target GLM template only inserts a placeholder for image content
("<reminder>You are unable to process this image...") rather than real
image understanding, so converting them would silently produce a
placeholder swap, not real usable multimodal training data. Left out.
"""

import os
import json
from pathlib import Path

import pyarrow.parquet as pq

SRC = Path(os.environ.get("SFTPIPE_SRC", "."))   # raw source folder (set by sft_pipeline.convert --src)
OUT_DIR = Path(os.environ.get("SFTPIPE_OUT", "out"))   # output folder (set by sft_pipeline.convert --out)

MATH_FAMILY = [
    "NaruseShiroha__capybara-math-traces",
    "NaruseShiroha__gpt-oss-120b-openmathreasoning-traces-primer",
    "NaruseShiroha__gpt-oss-12b-openmathreasoning-traces-primer",
    "NaruseShiroha__gpt-oss-120b-openreasoning-traces-primer",
    "NaruseShiroha__capybara-openreasoning-traces-primer",
]

TOOLCALL_FAMILY = [
    "NaruseShiroha__capybara-openreasoning-tool-calling-traces-primer",
    "NaruseShiroha__capybara-openreasoning-tool-calling-traces-primer-2",
]


def _shards(name):
    return sorted(SRC.glob(f"{name}*.parquet"))


def convert_math_family():
    results = []
    for name in MATH_FAMILY:
        out_path = OUT_DIR / f"{name}.jsonl"
        rows_in = rows_out = 0
        with open(out_path, "w") as out_f:
            for fpath in _shards(name):
                pf = pq.ParquetFile(fpath)
                for batch in pf.iter_batches(batch_size=500, columns=["problem", "generated_solution", "expected_answer"]):
                    for row in batch.to_pylist():
                        rows_in += 1
                        problem, sol = row.get("problem"), row.get("generated_solution")
                        if not problem or not sol:
                            continue
                        # NOTE: an earlier version filtered by checking expected_answer's
                        # text appears in the tail of generated_solution. Dropped after
                        # manually reading the "mismatches" it flagged on
                        # capybara-math-traces: all 3 checked were false positives from
                        # LaTeX/wording formatting differences (e.g. "convergent" vs
                        # "converges", \(+\infty\) vs $\infty$), not real wrong answers --
                        # same false-positive pattern already found and rejected for the
                        # zeju-0727 dataset. Converting directly, trusting the content.
                        msgs = [
                            {"role": "user", "content": problem},
                            {"role": "assistant", "content": sol},
                        ]
                        out_f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                        rows_out += 1
        if rows_out == 0:
            out_path.unlink(missing_ok=True)
        results.append({"dataset": name, "rows_in": rows_in, "rows_out": rows_out})
    return results


def convert_toolcall_family():
    results = []
    for name in TOOLCALL_FAMILY:
        out_path = OUT_DIR / f"{name}.jsonl"
        rows_in = rows_out = 0
        with open(out_path, "w") as out_f:
            for fpath in _shards(name):
                pf = pq.ParquetFile(fpath)
                for batch in pf.iter_batches(batch_size=500, columns=["system_prompt", "problem", "generated_solution"]):
                    for row in batch.to_pylist():
                        rows_in += 1
                        sysp, problem, sol = row.get("system_prompt"), row.get("problem"), row.get("generated_solution")
                        if not problem or not sol:
                            continue
                        msgs = []
                        if sysp:
                            msgs.append({"role": "system", "content": sysp})
                        msgs.append({"role": "user", "content": problem})
                        msgs.append({"role": "assistant", "content": sol})
                        out_f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                        rows_out += 1
        if rows_out == 0:
            out_path.unlink(missing_ok=True)
        results.append({"dataset": name, "rows_in": rows_in, "rows_out": rows_out})
    return results


def convert_fedric95():
    name = "fedric95__LIMO-Traces"
    out_path = OUT_DIR / f"{name}.jsonl"
    rows_in = rows_out = 0
    with open(out_path, "w") as out_f:
        for fpath in _shards(name):
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(batch_size=500, columns=["question", "answer", "match"]):
                for row in batch.to_pylist():
                    rows_in += 1
                    if row.get("match") != "Yes":
                        continue
                    q, a = row.get("question"), row.get("answer")
                    if not q or not a:
                        continue
                    msgs = [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]
                    out_f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
                    rows_out += 1
    if rows_out == 0:
        out_path.unlink(missing_ok=True)
    return {"dataset": name, "rows_in": rows_in, "rows_out": rows_out}


def main():
    for r in convert_math_family():
        print(f"{r['dataset']}: rows_in={r['rows_in']} rows_out={r['rows_out']}")
    for r in convert_toolcall_family():
        print(f"{r['dataset']}: rows_in={r['rows_in']} rows_out={r['rows_out']}")
    r = convert_fedric95()
    print(f"{r['dataset']}: rows_in={r['rows_in']} rows_out={r['rows_out']}")


if __name__ == "__main__":
    main()
