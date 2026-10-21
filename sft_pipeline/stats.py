"""SFT data statistics (ported from sft_audit/sft_stats.py). Read-only. One pass over all rows; every
distribution is reported in ROWS and GB (GB = uncompressed text bytes of those rows; disk GB is in Overview).

    python3 -m sft_pipeline.stats OUT_DIR NAME=FOLDER [NAME=FOLDER ...]

Output: OUT_DIR/sft_stats_<timestamp>.xlsx (+ sft_stats_latest.xlsx, + _per_file.csv)
Reasoning is counted from reasoning_content (current format) and from <think> in content (old format).
Tokens are REAL counts with the GPT o200k_base tokenizer (tiktoken), since
2026-09-25 (before that: characters / 4 estimate).
"""
import gzip, json, os, sys, time
from collections import Counter, defaultdict
from multiprocessing import Pool
import pandas as pd

DATASETS = {}   # name -> folder, filled from the command line
HERE = None      # output folder

GB = 1e9
BUCKETS = {
    "user_turns": [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 20), (21, 50), (51, None)],
    "tool_calls": [(0, 0), (1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, None)],
    "messages": [(1, 2), (3, 4), (5, 10), (11, 20), (21, 50), (51, 100), (101, 200), (201, None)],
    "tokens_o200k": [(0, 2048), (2049, 4096), (4097, 8192), (8193, 16384), (16385, 32768),
                   (32769, 65536), (65537, 131072), (131073, None)],
    "final_answer_tokens_o200k": [(0, 256), (257, 1024), (1025, 4096), (4097, 16384), (16385, 32768), (32769, None)],
}


LEN_LABELS = [f"{lo}+" if hi is None else f"{lo}-{hi}" for lo, hi in BUCKETS["tokens_o200k"]]


def bucket(kind, v):
    for lo, hi in BUCKETS[kind]:
        if v >= lo and (hi is None or v <= hi):
            return f"{lo}+" if hi is None else (str(lo) if lo == hi else f"{lo}-{hi}")
    return "other"


_ENC = None


def n_tokens(texts):
    """o200k_base token count of a list of strings (special-token text counted as plain text)."""
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("o200k_base")
    texts = [t for t in texts if t]
    return sum(len(x) for x in _ENC.encode_ordinary_batch(texts, num_threads=1)) if texts else 0


def part_tokens(texts):
    """o200k_base token count of each string (0 for empty), in order."""
    global _ENC
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("o200k_base")
    idx = [i for i, t in enumerate(texts) if t]
    out = [0] * len(texts)
    for i, toks in zip(idx, _ENC.encode_ordinary_batch([texts[i] for i in idx], num_threads=1)):
        out[i] = len(toks)
    return out


def row_stats(d, nbytes, C):
    msgs = d["messages"]
    roles = Counter(m["role"] for m in msgs)
    calls = sum(len(m.get("tool_calls") or []) for m in msgs)
    parts = [m.get("content") or "" for m in msgs] + [m.get("reasoning_content") or "" for m in msgs if m.get("reasoning_content")]
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            parts += [fn.get("name") or "", fn.get("arguments") or ""]
    if d.get("tools"):
        parts.append(json.dumps(d["tools"], ensure_ascii=False))
    lens = part_tokens(parts)
    toks = sum(lens)
    last = msgs[-1].get("content") or ""
    # final answer = last message after </think>; without </think> it is the whole last message, whose count we already have
    ans_toks = n_tokens([last.split("</think>")[-1]]) if "</think>" in last else lens[len(msgs) - 1]
    feats = {
        "user_turns": bucket("user_turns", roles["user"]),
        "tool_calls": bucket("tool_calls", calls),
        "messages": bucket("messages", len(msgs)),
        "tokens_o200k": bucket("tokens_o200k", toks),
        "final_answer_tokens_o200k": bucket("final_answer_tokens_o200k", ans_toks),
    }
    flags = {
        "single_turn": roles["user"] == 1, "multi_turn": roles["user"] > 1,
        "uses_tools (has tool calls)": calls > 0, "has_tools_list": bool(d.get("tools")),
        "has_system_prompt": roles["system"] > 0,
        "has_reasoning": any((m.get("reasoning_content") or "").strip() or "<think>" in (m.get("content") or "") for m in msgs if m["role"] == "assistant"),
    }
    C["rows"] += 1; C["bytes"] += nbytes
    C["user_turns_sum"] += roles["user"]; C["assistant_turns_sum"] += roles["assistant"]
    C["tool_calls_sum"] += calls; C["tool_msgs_sum"] += roles["tool"]; C["messages_sum"] += len(msgs)
    C["tokens_sum"] += toks
    if toks > C["tokens_max"]:
        C["tokens_max"] = toks          # per-file max context length (per-dataset sum of this is meaningless)
    for k, v in feats.items():
        C[(k, v, "rows")] += 1; C[(k, v, "bytes")] += nbytes
    for k, v in flags.items():
        if v:
            C[("flag", k, "rows")] += 1; C[("flag", k, "bytes")] += nbytes


CHUNK = 512 * 1024 ** 2   # plain .jsonl files bigger than this are split into byte ranges (a 123 GB file
                          # in one process took ~6.5 h and held up the whole run, 2026-09-25)


def one(args):
    """stats of one file, or of the byte range [start, end) of a plain .jsonl (lines that START in the range)."""
    ds, root, rel, start, end = args
    p = os.path.join(root, rel)
    C = Counter()
    if p.endswith(".gz"):
        with gzip.open(p, "rb") as f:
            for line in f:
                if line.strip():
                    row_stats(json.loads(line), len(line), C)
    else:
        with open(p, "rb") as f:
            if start:
                f.seek(start - 1)
                f.readline()              # finish the line that started before this range
            pos = f.tell()
            while end is None or pos < end:
                line = f.readline()
                if not line:
                    break
                pos += len(line)
                if line.strip():
                    row_stats(json.loads(line), len(line), C)
    if not start:
        C["disk_bytes"] = os.path.getsize(p)
    return ds, rel, C


def merge(a, b):
    """add chunk counters of the same file; tokens_max is a max, not a sum."""
    mx = max(a.get("tokens_max", 0), b.get("tokens_max", 0))
    a.update(b)
    a["tokens_max"] = mx
    return a


def main():
    global HERE
    if len(sys.argv) < 3:
        raise SystemExit("usage: python3 -m sft_pipeline.stats OUT_DIR NAME=FOLDER [NAME=FOLDER ...]")
    HERE = os.path.abspath(sys.argv[1])
    for a in sys.argv[2:]:
        n, p = a.split("=", 1)
        DATASETS[n] = os.path.abspath(p)
    names = list(DATASETS)
    jobs = []
    for ds in names:
        root = DATASETS[ds]
        for r, dirs, fs in os.walk(root):
            dirs[:] = [x for x in dirs if not x.startswith("_")]
            for f in fs:
                if not f.endswith((".jsonl", ".jsonl.gz")):
                    continue
                rel = os.path.relpath(os.path.join(r, f), root)
                size = os.path.getsize(os.path.join(r, f))
                if f.endswith(".jsonl") and size > CHUNK:
                    jobs += [(ds, root, rel, a, min(a + CHUNK, size)) for a in range(0, size, CHUNK)]
                else:
                    jobs.append((ds, root, rel, 0, None))
    jobs.sort(key=lambda j: -((j[4] - j[3]) if j[4] is not None else os.path.getsize(os.path.join(j[1], j[2]))))
    print(f"{len(jobs)} jobs (big files split into {CHUNK >> 20} MB ranges)", flush=True)
    per_ds = defaultdict(Counter); files = {}
    # multiprocessing.Pool, not ProcessPoolExecutor: submitting ~30K futures at once deadlocked on Python 3.10 (2026-09-25)
    with Pool(int(os.environ.get("WORKERS", 48))) as pool:
        for i, res in enumerate(pool.imap_unordered(one, jobs), 1):
            ds, rel, C = res
            per_ds[ds].update(C)
            files[(ds, rel)] = merge(files[(ds, rel)], C) if (ds, rel) in files else C
            if i % 200 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] jobs", flush=True)
    per_file = [(ds, rel, C) for (ds, rel), C in files.items()]

    ov = []
    for ds in names:
        C = per_ds[ds]; n = max(C["rows"], 1)
        ov.append({"dataset": ds, "files": sum(1 for x in per_file if x[0] == ds), "rows": C["rows"],
                   "GB_on_disk": round(C["disk_bytes"] / GB, 2), "GB_text": round(C["bytes"] / GB, 2),
                   "tokens_o200k_billion": round(C["tokens_sum"] / 1e9, 2),
                   "avg_user_turns": round(C["user_turns_sum"] / n, 2), "avg_assistant_turns": round(C["assistant_turns_sum"] / n, 2),
                   "avg_messages": round(C["messages_sum"] / n, 2), "avg_tool_calls": round(C["tool_calls_sum"] / n, 3),
                   "total_tool_calls": C["tool_calls_sum"], "avg_tokens_o200k_per_row": round(C["tokens_sum"] / n)})
    tot = {k: sum(r[k] for r in ov) for k in ("files", "rows", "GB_on_disk", "GB_text", "tokens_o200k_billion", "total_tool_calls")}
    ov.append({"dataset": "ALL", **{k: round(v, 2) if isinstance(v, float) else v for k, v in tot.items()}})

    def dist(kind):
        rows = []
        for ds in names:
            C = per_ds[ds]; n = max(C["rows"], 1); b = max(C["bytes"], 1)
            for lo, hi in BUCKETS[kind]:
                lab = f"{lo}+" if hi is None else (str(lo) if lo == hi else f"{lo}-{hi}")
                r, by = C[(kind, lab, "rows")], C[(kind, lab, "bytes")]
                rows.append({"dataset": ds, kind: lab, "rows": r, "pct_rows": round(100 * r / n, 2),
                             "GB_text": round(by / GB, 3), "pct_GB": round(100 * by / b, 2)})
        return pd.DataFrame(rows)

    flags = []
    for ds in names:
        C = per_ds[ds]; n = max(C["rows"], 1); b = max(C["bytes"], 1)
        for k in ("single_turn", "multi_turn", "uses_tools (has tool calls)", "has_tools_list",
                  "has_system_prompt", "has_reasoning"):
            r, by = C[("flag", k, "rows")], C[("flag", k, "bytes")]
            flags.append({"dataset": ds, "type": k, "rows": r, "pct_rows": round(100 * r / n, 2),
                          "GB_text": round(by / GB, 3), "pct_GB": round(100 * by / b, 2)})
    pf = pd.DataFrame([{"dataset": ds, "file": rel, "full_path": os.path.join(DATASETS[ds], rel), "rows": C["rows"], "GB_on_disk": round(C["disk_bytes"] / GB, 3),
                        "GB_text": round(C["bytes"] / GB, 3),
                        "avg_user_turns": round(C["user_turns_sum"] / max(C["rows"], 1), 2),
                        "avg_tool_calls": round(C["tool_calls_sum"] / max(C["rows"], 1), 3),
                        "multi_turn_rows": C[("flag", "multi_turn", "rows")],
                        "tool_using_rows": C[("flag", "uses_tools (has tool calls)", "rows")],
                        "reasoning_rows": C[("flag", "has_reasoning", "rows")],
                        "tokens_o200k": C["tokens_sum"],
                        "avg_tokens_o200k": round(C["tokens_sum"] / max(C["rows"], 1)),
                        "max_tokens_o200k": C["tokens_max"],
                        **{f"rows_len_{lab}": C[("tokens_o200k", lab, "rows")] for lab in LEN_LABELS}}
                       for ds, rel, C in sorted(per_file, key=lambda x: (x[0], x[1]))])
    notes = pd.DataFrame({"note": [
        "GB_text = size of the rows as plain JSON text (uncompressed). GB_on_disk = file size on disk (smaller for .jsonl.gz datasets).",
        "tokens_o200k = REAL token count with the GPT o200k_base tokenizer (tiktoken) over all message content + tool-call names/arguments + the tools[] JSON. Chat-template special tokens (role markers etc.) are NOT included -- they depend on the model's template.",
        "user_turns = number of user messages in the row. messages = all messages (system + user + assistant + tool).",
        "tool_calls = number of structured tool calls made by the assistant in the row.",
        "final_answer_tokens_o200k = o200k tokens of the last assistant message after any </think> block.",
        "Types can overlap (a row can be multi-turn AND use tools AND have reasoning).",
    ]})
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(HERE, exist_ok=True)
    out = os.path.join(HERE, f"sft_stats_{stamp}.xlsx")
    def with_path(df):
        """every sheet gets the dataset folder path next to the dataset name (user ask, 2026-09-25)"""
        df = pd.DataFrame(df)
        if "dataset" in df.columns and "dataset_path" not in df.columns:
            df.insert(1, "dataset_path", df["dataset"].map(lambda d: DATASETS.get(d, "")))
        return df
    pf.to_csv(out.replace(".xlsx", "_per_file.csv"), index=False)   # full per-file table, for domain stats
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        with_path(ov).to_excel(xw, sheet_name="Overview", index=False)
        with_path(flags).to_excel(xw, sheet_name="Types", index=False)
        with_path(dist("user_turns")).to_excel(xw, sheet_name="Turns", index=False)
        with_path(dist("tool_calls")).to_excel(xw, sheet_name="Tool calls", index=False)
        with_path(dist("messages")).to_excel(xw, sheet_name="Messages", index=False)
        with_path(dist("tokens_o200k")).to_excel(xw, sheet_name="Length (tokens)", index=False)
        with_path(dist("final_answer_tokens_o200k")).to_excel(xw, sheet_name="Answer length", index=False)
        with_path(pf).to_excel(xw, sheet_name="Per file", index=False)
        notes.to_excel(xw, sheet_name="Notes", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                w = max((len(str(c.value)) for c in col[:300] if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, w + 2), 80)
            ws.freeze_panes = "A2"
    import shutil
    shutil.copy2(out, os.path.join(HERE, "sft_stats_latest.xlsx"))
    shutil.copy2(out.replace(".xlsx", "_per_file.csv"), os.path.join(HERE, "sft_stats_latest_per_file.csv"))
    print(pd.DataFrame(ov).to_string(index=False))
    print("written", out)


if __name__ == "__main__":
    main()
