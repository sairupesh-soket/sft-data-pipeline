#!/usr/bin/env python3
"""Attach the canonical `python` tool schema to existing output rows (same rule as sft_clean_filter.needs_python_tool).
Row count never changes (only the `tools` field is added), so chunk records stay valid; the per-chunk record and the
report get a repair counter `python_tool_schema_attached`. Atomic per file (temp file + rename). Dry-run unless --apply."""
import argparse, glob, json, mmap, os, sys, time, collections
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from . import strict_filter as F

def work(args):
    path, root, apply = args
    with open(path, "rb") as f:
        try: mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError: return path, 0, 0
        has = mm.find(b'"tool_calls"') != -1; mm.close()
    if not has: return path, 0, 0
    n = rows = 0; tmp = path + ".py_tmp"; out = open(tmp, "wb") if apply else None
    with open(path, "rb") as f:
        for line in f:
            rows += 1
            if b'"tool_calls"' in line:
                r = json.loads(line)
                if F.needs_python_tool(r["messages"], r.get("tools")):
                    r["tools"] = [F.PYTHON_TOOL]; line = (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"); n += 1
            if out: out.write(line)
    if out:
        out.close()
        if n:
            os.replace(tmp, path)
            rel = os.path.relpath(path, root); mp = os.path.join(root, "_meta", rel + ".json")
            if os.path.exists(mp):
                d = json.load(open(mp)); d["repairs"]["python_tool_schema_attached"] = d["repairs"].get("python_tool_schema_attached", 0) + n
                json.dump(d, open(mp + ".tmp", "w")); os.replace(mp + ".tmp", mp)
        else: os.remove(tmp)
    return path, n, rows

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--apply", action="store_true"); ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(a.root, "*", "*.jsonl")) if not f.split("/")[-2].startswith("_")); t0 = time.time(); total = 0; nfiles = 0
    with Pool(a.workers) as p:
        for i, (path, n, rows) in enumerate(p.imap_unordered(work, [(f, a.root, a.apply) for f in files], chunksize=1), 1):
            total += n; nfiles += bool(n)
            if i % 1000 == 0: print(f"[{i}/{len(files)}] {time.time()-t0:.0f}s schemas attached so far {total:,}", flush=True)
    print(f"{'APPLIED' if a.apply else 'DRY-RUN'}: attached the python tool schema to {total:,} rows in {nfiles:,} files ({time.time()-t0:.0f}s)")
    rp = os.path.join(a.root, "_filter_report.json")
    if a.apply and os.path.exists(rp):
        d = json.load(open(rp)); d["repaired"]["python_tool_schema_attached"] = total
        d.setdefault("post_steps", []).append("python_tool_schema_attached: %d rows (sft_python_tool_postpass.py); row counts unchanged" % total)
        json.dump(d, open(rp + ".tmp", "w"), indent=1, ensure_ascii=False); os.replace(rp + ".tmp", rp); print("report updated")
