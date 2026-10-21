"""Generic row-by-row pass over a folder of OpenAI-format JSONL.

A STEP is a function  step(row: dict, c: Counter) -> (status, row)
  status "keep"        : row not changed (the original line is written back byte for byte)
  status "changed"     : row changed (the new row is written)
  status "drop:<why>"  : row removed; it is written with its reason to LOG/dropped/<rel>
Steps run in the given order; the first "drop" stops the chain for that row.

run_pass(src, out, steps, ...) mirrors the folder layout of src into out. The source is never changed
unless inplace=True (then out must equal src). A file is listed in LOG/_pass_log.jsonl when it is done,
so a killed run resumes where it stopped.
"""
import json
import os
from collections import Counter
from multiprocessing import Pool

import orjson

from . import io as sio

_CFG = {}


def apply_steps(row, steps, c):
    changed = False
    for step in steps:
        status, row = step(row, c)
        if status.startswith("drop"):
            return status, None
        if status == "changed":
            changed = True
    return ("changed" if changed else "keep"), row


def _one(rel):
    src, out, steps, log, prefilter = _CFG["src"], _CFG["out"], _CFG["steps"], _CFG["log"], _CFG["prefilter"]
    sp, dp = os.path.join(src, rel), os.path.join(out, rel)
    tmp = sio.atomic_target(dp)
    drop_path = os.path.join(log, "dropped", rel) if log else None
    c = Counter()
    fd = None
    with sio.open_bytes(sp) as fi, sio.open_out_bytes(tmp, dp) as fo:
        for ln, line in enumerate(fi, 1):
            if not line.strip():
                continue
            c["rows_in"] += 1
            if prefilter is not None and not prefilter(line):
                fo.write(line if line.endswith(b"\n") else line + b"\n")
                c["rows_out"] += 1
                continue
            try:
                row = orjson.loads(line)
            except Exception:
                status, row = "drop:bad_json", None
            else:
                status, row = apply_steps(row, steps, c)
            if status.startswith("drop"):
                c[status] += 1
                if drop_path:
                    if fd is None:
                        os.makedirs(os.path.dirname(drop_path), exist_ok=True)
                        fd = open(drop_path + ".tmp_sftpipe", "w", encoding="utf-8")
                    fd.write(json.dumps({"line": ln, "reason": status[5:], "row": line.decode("utf-8", "replace").rstrip("\n")},
                                        ensure_ascii=False) + "\n")
                continue
            if status == "changed":
                c["rows_changed"] += 1
                fo.write(orjson.dumps(row) + b"\n")
            else:
                fo.write(line if line.endswith(b"\n") else line + b"\n")
            c["rows_out"] += 1
    sio.commit(tmp, dp)
    if fd is not None:
        fd.close()
        os.replace(drop_path + ".tmp_sftpipe", drop_path)
    return rel, dict(c)


def run_pass(src, out, steps, log=None, nproc=8, inplace=False, prefilter=None, files=None):
    """Run steps over every data file of src. Returns the total Counter."""
    src, out = os.path.abspath(src), os.path.abspath(out)
    if src == out and not inplace:
        raise SystemExit("output == source: pass inplace=True (--inplace) to rewrite the source")
    if not inplace and out.startswith(src + os.sep):
        raise SystemExit("output must not be inside the source folder")
    log = os.path.abspath(log) if log else os.path.join(os.path.dirname(out), os.path.basename(out) + "_logs")
    if log.startswith(out + os.sep) or log == out:
        raise SystemExit("log folder must be outside the output data folder")
    os.makedirs(log, exist_ok=True)
    rels = [os.path.relpath(p, src) for p in (files or sio.list_files(src))]
    done = set()
    plog = os.path.join(log, "_pass_log.jsonl")
    if os.path.exists(plog):
        done = {json.loads(l)["file"] for l in open(plog)}
    todo = [r for r in rels if r not in done]
    _CFG.update(src=src, out=out, steps=steps, log=log, prefilter=prefilter)
    tot = Counter()
    with Pool(nproc) as pool, open(plog, "a") as lf:
        for rel, c in pool.imap_unordered(_one, todo, chunksize=1):
            tot.update(c)
            lf.write(json.dumps({"file": rel, **c}) + "\n")
            lf.flush()
    for l in open(plog):
        d = json.loads(l)
        if d["file"] in done:
            d.pop("file")
            tot.update(d)
    json.dump(dict(sorted(tot.items())), open(os.path.join(log, "pass_counts.json"), "w"), indent=1)
    return tot
