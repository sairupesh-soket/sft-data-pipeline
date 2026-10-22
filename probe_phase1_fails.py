#!/usr/bin/env python3
"""Classify Phase-1 unmappable chat rows by payload shape."""
from __future__ import annotations
import json, os, sys, ast
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq

sys.path.insert(0, "/projects/data/datasets/translation_data/SFT/DEDUP_PIPELINE")
import sft_normalize as N
sys.path.insert(0, "/code")
import run_phases as R

SRC = Path("/projects/data/datasets/translation_data/SFT/OUTPUT")

# focus on high-fail shard file set (shard 10 of 16)
files = R.index_phase(SRC, 1)
files = R.apply_shard(files, 16, 10)
print(f"probe shard10: {len(files)} files, {sum(f['n'] for f in files):,} rows", flush=True)

reasons = Counter()
examples = {}
per_file = Counter()
sample_n = 0
TARGET = 8000  # sample failing rows across files
MAX_PER_FILE = 400

def classify(raw, plan):
    # deeper than just None — what shape failed?
    if raw is None:
        return "raw_none"
    if isinstance(raw, float):  # NaN
        return "raw_nan"
    if isinstance(raw, str):
        t = raw.strip()
        if not t:
            return "raw_empty_str"
        if t[0] in "[{":
            parsed = None
            for fn in (json.loads, ast.literal_eval):
                try:
                    parsed = fn(t)
                    break
                except Exception:
                    pass
            if parsed is None:
                # try transcript parsers
                if N.parse_transcript(raw) or N.parse_bracketed(raw):
                    return "str_jsonish_but_parse_ok_unexpected"
                return "str_jsonish_unparseable"
            if isinstance(parsed, dict):
                return "str_json_dict_single"
            if isinstance(parsed, (list, tuple)):
                if not parsed:
                    return "str_json_empty_list"
                e0 = parsed[0]
                if isinstance(e0, str):
                    return "str_json_list_of_str"
                if isinstance(e0, dict):
                    keys = sorted(set().union(*[set(x.keys()) for x in parsed[:5] if isinstance(x, dict)]))
                    roles = []
                    for x in parsed[:20]:
                        if not isinstance(x, dict):
                            continue
                        roles.append(str(x.get("role") or x.get("from") or x.get("speaker") or "?"))
                    # check content emptiness
                    nonempty = 0
                    for x in parsed:
                        if not isinstance(x, dict):
                            continue
                        cont = x.get("content") or x.get("value") or x.get("text") or x.get("message") or ""
                        if str(cont).strip():
                            nonempty += 1
                    role_set = sorted(set(r.lower() for r in roles))
                    if nonempty == 0:
                        return f"str_json_list_dicts_empty_content keys={keys} roles={role_set}"
                    if not any(r.lower() in {"user","human","usr","prompter"} or "user" in r.lower() for r in roles):
                        # maybe only assistant / system
                        return f"str_json_list_dicts_no_user_role keys={keys} roles={role_set}"
                    if not any(r.lower() in {"assistant","gpt","bot","ai","model","answer","response"} or "assistant" in r.lower() for r in roles):
                        return f"str_json_list_dicts_no_asst_role keys={keys} roles={role_set}"
                    return f"str_json_list_dicts_other keys={keys} roles={role_set}"
                return f"str_json_list_of_{type(e0).__name__}"
            return f"str_json_{type(parsed).__name__}"
        # plain text
        if "User:" in raw or "user:" in raw[:200].lower():
            return "str_house_text_unparsed"
        if N.parse_transcript(raw):
            return "str_transcript_ok_unexpected"
        return "str_plain_no_role_markers"
    if isinstance(raw, (list, tuple)):
        if not raw:
            return "list_empty"
        e0 = raw[0]
        if isinstance(e0, str):
            return "list_of_str"
        if isinstance(e0, dict):
            keys = sorted(set().union(*[set(x.keys()) for x in raw[:5] if isinstance(x, dict)]))
            roles = [str(x.get("role") or x.get("from") or x.get("speaker") or "?") for x in raw[:30] if isinstance(x, dict)]
            nonempty = 0
            for x in raw:
                if not isinstance(x, dict):
                    continue
                cont = x.get("content") or x.get("value") or x.get("text") or x.get("message") or ""
                if str(cont).strip():
                    nonempty += 1
            role_set = sorted(set(r.lower() for r in roles))
            # also check after N.to_messages path pieces
            msgs = N.to_messages({plan["col"]: raw}, plan)
            if msgs is None:
                # check canonicalize/validate path? unmappable is before that
                has_user = any("user" in r or r in {"human","usr"} for r in role_set)
                has_asst = any(r in {"assistant","gpt","bot","ai","model"} or "assistant" in r for r in role_set)
                if nonempty == 0:
                    return f"list_dicts_empty_content keys={keys} roles={role_set}"
                if not has_user:
                    return f"list_dicts_no_user keys={keys} roles={role_set}"
                if not has_asst:
                    return f"list_dicts_no_asst keys={keys} roles={role_set}"
                # maybe weird role names that ROLE_ALIAS misses but positional should work
                return f"list_dicts_to_messages_none keys={keys} roles={role_set} nonempty={nonempty}/{len(raw)}"
            return "list_ok_unexpected"
        return f"list_of_{type(e0).__name__}"
    return f"raw_type_{type(raw).__name__}"


for meta in files:
    if sample_n >= TARGET:
        break
    col = meta["plan"]["col"]
    got = 0
    try:
        pf = pq.ParquetFile(meta["path"], memory_map=True)
        want = [c for c in R.needed_columns(meta["plan"], meta["cols"]) if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(columns=want, batch_size=256):
            if sample_n >= TARGET or got >= MAX_PER_FILE:
                break
            for row in batch.to_pylist():
                if sample_n >= TARGET or got >= MAX_PER_FILE:
                    break
                rec, reason = R.row_to_openai(row, meta["plan"])
                if rec is not None:
                    continue
                # only care about unmappable-ish / format
                raw = row.get(col)
                tag = classify(raw, meta["plan"])
                # also record pipeline reason
                reasons[f"pipe:{reason}|shape:{tag}"] += 1
                per_file[meta["rel"]] += 1
                if tag not in examples:
                    # compact example
                    if isinstance(raw, str):
                        ex = raw[:500]
                    elif isinstance(raw, list):
                        ex = raw[:3]
                    else:
                        ex = repr(raw)[:500]
                    examples[tag] = {"file": meta["rel"], "domain": meta["domain"], "example": ex, "pipe_reason": reason}
                sample_n += 1
                got += 1
    except Exception as e:
        print("file err", meta["rel"], type(e).__name__, e, flush=True)

print("\n=== TOP FAIL SHAPES (sampled", sample_n, ") ===", flush=True)
for k, v in reasons.most_common(40):
    print(f"{v:6d}  {k}", flush=True)

print("\n=== TOP FILES IN SAMPLE ===", flush=True)
for k, v in per_file.most_common(15):
    print(f"{v:6d}  {k}", flush=True)

print("\n=== EXAMPLES ===", flush=True)
for tag, info in list(examples.items())[:25]:
    print("\n---", tag, "---", flush=True)
    print("file:", info["file"], "domain:", info["domain"], "pipe:", info["pipe_reason"], flush=True)
    print(json.dumps(info["example"], ensure_ascii=False, default=str)[:800], flush=True)
