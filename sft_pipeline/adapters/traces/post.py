"""Traces: post-convert fixes, in the production order (run after traces + traces_extra1..9, before clean).

    python3 -m sft_pipeline.convert traces_post --src RAW --out CONVERTED

1. missing assistant turn (8 API-call-log datasets): the raw "messages" column holds only system/user; the reply is in
   "assistant_message" (or "response") and "tool_calls" (JSON string). The output file is rebuilt with the reply.
   Ported from sft_traces_v1/scripts/fix_missing_assistant_turn.py (unchanged).
2. fimbulvntr__deepseek-r1-traces-no-cjk: reply in "reasoning" + "answer" columns -> one assistant turn
   (content = answer, reasoning_content = reasoning). Ported from fix_fimbulvntr_missing_assistant.py (unchanged).
3. parquet struct-union nulls in tool-call arguments (5 datasets whose raw arguments column is a STRUCT):
   null argument keys removed. Ported from fix_struct_nulls_and_empty_assistant.py (the empty-assistant part is
   done by the clean step role_rules).
4. excluded files: 21 byte-identical duplicate files (one copy kept) and one base-model trace file with degenerate
   output. They are MOVED to <log>/excluded/, not deleted.
Backups of rebuilt files go to <log>/backups/.
"""
import glob
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from .base import convert_message, normalize_tool_calls, to_plain

SRC = Path(os.environ.get("SFTPIPE_SRC", "."))
OUT_DIR = Path(os.environ.get("SFTPIPE_OUT", "out"))
LOG_DIR = Path(os.environ.get("SFTPIPE_LOG", str(OUT_DIR).rstrip("/") + "_logs")) / "traces_post"

DATASETS = [
    ("davidberenstein1957__gpt-4o-traces", "assistant_message", True),
    ("kabelomalapane__gpt-4o-traces", "assistant_message", True),
    ("pprajan__gpt-4o-function-calling-traces", "assistant_message", True),
    ("pprajan__gpt-4o-traces", "assistant_message", True),
    ("cfahlgren1__gpt-4o-function-calling-traces", "assistant_message", True),
    ("cfahlgren1__gpt-4o-mini-vision-traces", "assistant_message", True),
    ("PierreColombo__gpt-4o-traces", "assistant_message", True),
    ("cfahlgren1__traces", "response", False),
]


def build_assistant_message(row, text_col, has_tool_calls):
    content = row.get(text_col)
    if isinstance(content, str):
        content = content.strip() or None
    tool_calls_raw = row.get("tool_calls") if has_tool_calls else None
    if isinstance(tool_calls_raw, str):
        # this column is stored as a JSON-serialized string, not a native
        # list<struct> -- confirmed via pyarrow schema (`tool_calls: string`)
        try:
            tool_calls_raw = json.loads(tool_calls_raw)
        except Exception:
            tool_calls_raw = None
    tool_calls = normalize_tool_calls(tool_calls_raw) if tool_calls_raw else None
    if content is None and not tool_calls:
        return None  # e.g. PierreColombo's 1 row with error="'Stream' object..." and no reply at all
    out = {"role": "assistant", "content": content if content is not None else ""}
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def fix_dataset(prefix, text_col, has_tool_calls):
    files = sorted(glob.glob(str(SRC / f"{prefix}*.parquet")))
    assert files, f"no source parquet found for {prefix}"

    out_path = OUT_DIR / f"{prefix}.jsonl"
    if out_path.exists():
        backup_path = LOG_DIR / "backups" / f"{prefix}.jsonl.backup_before_assistant_fix"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if not backup_path.exists():
            shutil.copy2(out_path, backup_path)

    rows_in = 0
    rows_out = 0
    skipped_no_reply = 0
    skipped_no_user_content = 0

    with open(out_path, "w") as out_f:
        for fp in files:
            pf = pq.ParquetFile(fp)
            for batch in pf.iter_batches(batch_size=1000):
                for row in batch.to_pylist():
                    rows_in += 1
                    msgs_raw = to_plain(row.get("messages"))
                    if not msgs_raw:
                        skipped_no_user_content += 1
                        continue
                    out_msgs = []
                    for m in msgs_raw:
                        cm = convert_message(m)
                        if cm is not None:
                            out_msgs.append(cm)
                    if not out_msgs:
                        skipped_no_user_content += 1
                        continue

                    assistant_msg = build_assistant_message(row, text_col, has_tool_calls)
                    if assistant_msg is None:
                        skipped_no_reply += 1
                        continue

                    out_msgs.append(assistant_msg)
                    out_f.write(json.dumps({"messages": out_msgs}, ensure_ascii=False) + "\n")
                    rows_out += 1

    return {
        "prefix": prefix,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "skipped_no_reply": skipped_no_reply,
        "skipped_no_user_content": skipped_no_user_content,
    }



# ---------------- 2. fimbulvntr ----------------
NAME = "fimbulvntr__deepseek-r1-traces-no-cjk"


def fimbulvntr():
    files = sorted(SRC.glob(f"{NAME}*.parquet"))
    assert files, f"no source parquet found for {NAME}"

    out_path = OUT_DIR / f"{NAME}.jsonl"
    backup_path = LOG_DIR / f"{NAME}.jsonl.backup_before_assistant_fix"
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

    return dict(name=NAME, rows_in=rows_in, rows_out=rows_out, skipped=skipped)  # noqa
    print(f"{NAME}: rows_in={rows_in} rows_out={rows_out} skipped={skipped}")



# ---------------- 3. struct-union nulls ----------------
STRUCT_FILES = {
    "greghavens__gpt-5.6-sol-coding-and-debugging-traces.jsonl",
    "KermitCO__qwen3.5-9B-tau2bench-retail-traces.jsonl",
    "bobbyzhong__code-orchestrator-synth-traces.jsonl",
    "bobbyzhong__code-orchestrator-synth-traces-fixed.jsonl",
    "r0b0tlab__nemotron-nano-hermes-traces.jsonl",
}


def strip_nulls(o):
    if isinstance(o, dict):
        return {k: strip_nulls(v) for k, v in o.items() if v is not None}
    if isinstance(o, list):
        return [strip_nulls(x) for x in o]
    return o


def struct_nulls(fname):
    path = OUT_DIR / fname
    if not path.exists():
        return {"file": fname, "missing": 1}
    c = Counter()
    tmp = str(path) + ".tmp_sftpipe"
    with open(path) as fi, open(tmp, "w") as fo:
        for line in fi:
            d = json.loads(line)
            changed = False
            for m in d["messages"]:
                for tc in m.get("tool_calls") or []:
                    a = tc["function"].get("arguments")
                    if isinstance(a, str):
                        try:
                            a2 = json.loads(a)
                        except Exception:
                            continue
                        b = strip_nulls(a2)
                        if b != a2:
                            tc["function"]["arguments"] = json.dumps(b, ensure_ascii=False); c["calls_nulls_stripped"] += 1; changed = True
                    elif isinstance(a, dict):
                        b = strip_nulls(a)
                        if b != a:
                            tc["function"]["arguments"] = b; c["calls_nulls_stripped"] += 1; changed = True
            fo.write(json.dumps(d, ensure_ascii=False) + "\n" if changed else line)
    os.replace(tmp, path)
    return {"file": fname, **c}


# ---------------- 4. excluded files ----------------
EXCLUDED_DUPLICATES = [  # moved copy -> kept byte-identical twin
    ('DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_adam0670a22c.jsonl', 'DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_adam-0670a22c.jsonl'),
    ('DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_warmu81c76489.jsonl', 'DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_warm-81c76489.jsonl'),
    ('DCAgent__DCAgent_dev_set_71_tasks_DCAgent_all-puzzles-sandboxes-traces-terminus-2-with-gpf7edb1db.jsonl', 'DCAgent__DCAgent_dev_set_71_tasks_DCAgent_all-puzzles-sandboxes-traces-terminus-2-with-g-f7edb1db.jsonl'),
    ('DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_adam85ff894d.jsonl', 'DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_adam-85ff894d.jsonl'),
    ('DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_warmub18e9207.jsonl', 'DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_warm-b18e9207.jsonl'),
    ('DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_learn00f22550.jsonl', 'DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_lear-00f22550.jsonl'),
    ('jkazdan__sanitized-gpt-4o-traces.jsonl', 'jkazdan__cleaned-gpt-4o-traces.jsonl'),
    ('DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_cutof7823acbc.jsonl', 'DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_cuto-7823acbc.jsonl'),
    ('DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_learn72e6822b.jsonl', 'DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_lear-72e6822b.jsonl'),
    ('mlfoundations-dev__claude_3_7_tbench_traces_sharegptv1.jsonl', 'mlfoundations-dev__claude_3_7_tbench_traces_sharegpt.jsonl'),
    ('DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_cutof722c5660.jsonl', 'DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_cuto-722c5660.jsonl'),
    ('jonathanyin__aime_1983_2023_grok-3-mini-high_traces_r1_formatted.jsonl', 'jonathanyin__aime_1983_2023_grok-3-mini-high_traces_32768_r1_formatted.jsonl'),
    ('NaruseShiroha__gpt-oss-12b-openmathreasoning-traces-primer.jsonl', 'NaruseShiroha__gpt-oss-120b-openreasoning-traces-primer.jsonl'),
    ('smolagents__gaia-traces.jsonl', 'shamikbose89__gaia_traces.jsonl'),
    ('mlfoundations-dev__test-tbench-traces.jsonl', 'mlfoundations-dev__tbench_traces_sharegptv1.jsonl'),
    ('DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_adam592fbdab.jsonl', 'DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_adam-592fbdab.jsonl'),
    ('DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_adam9f796e10.jsonl', 'DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_adam-9f796e10.jsonl'),
    ('DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_adam9049c2c2.jsonl', 'DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_adam-9049c2c2.jsonl'),
    ('DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_learn50f16e4b.jsonl', 'DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_lear-50f16e4b.jsonl'),
    ('mlfoundations-dev__terminal-bench-traces-local.jsonl', 'mlfoundations-dev__tbench_traces_local_sharegptv1.jsonl'),
    ('DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_cutof8b42d66a.jsonl', 'DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_cuto-8b42d66a.jsonl'),
    ('DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_adam60b0b114.jsonl', 'DCAgent__DCAgent_dev_set_71_tasks_DCAgent_code-contests-sandboxes-traces-terminus-2_adam-60b0b114.jsonl'),
]
EXCLUDED_QUALITY = [  # base (non-instruct) model traces: degenerate output
    "DCAgent__DCAgent_dev_set_71_tasks_DCAgent_staqc-sandboxes-traces-terminus-2_Qwen3-8B-Base_20251102.jsonl",
]


def exclude():
    dst = LOG_DIR / "excluded"
    dst.mkdir(parents=True, exist_ok=True)
    moved = []
    for f in [a for a, _ in EXCLUDED_DUPLICATES] + EXCLUDED_QUALITY:
        p = OUT_DIR / f
        if p.exists():
            shutil.move(str(p), str(dst / f)); moved.append(f)
    return moved


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report = {"missing_assistant": [], "fimbulvntr": None, "struct_nulls": [], "excluded": []}
    for prefix, text_col, has_tc in DATASETS:
        if glob.glob(str(SRC / f"{prefix}*.parquet")):
            report["missing_assistant"].append(fix_dataset(prefix, text_col, has_tc))
    if list(SRC.glob(f"{NAME}*.parquet")):
        report["fimbulvntr"] = fimbulvntr()
    for f in sorted(STRUCT_FILES):
        report["struct_nulls"].append(struct_nulls(f))
    report["excluded"] = exclude()
    json.dump(report, open(LOG_DIR / "traces_post_report.json", "w"), indent=1)
    print(json.dumps(report, indent=1)[:3000])


if __name__ == "__main__":
    main()
