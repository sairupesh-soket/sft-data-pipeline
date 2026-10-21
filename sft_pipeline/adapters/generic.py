"""Generic adapter: any table / JSONL schema -> {"messages": [...]} with schema_infer (ported sft_normalize_v4).

For each input file (parquet, .jsonl, .jsonl.gz under --src):
  1. plan(column names): find the conversation column (messages / conversations / ShareGPT ...) or the field pair
     (instruction+input / question / prompt ... -> answer / response / output ..., optional system and reasoning
     columns). One plan per file, not per row.
  2. to_messages(row, plan): one row -> list of {"role", "content"}; bad rows give None and are counted.
     A reasoning column is written as <think>...</think> in the answer; run sft_pipeline.clean after this
     (step move_think puts it into reasoning_content).
Files with no usable plan are listed in <log>/no_adapter.jsonl with their columns.
Output: <out>/<same relative path>.jsonl.  Resumable: <log>/convert_log.jsonl.
"""
import json
import os
from collections import Counter
from multiprocessing import Pool

from . import schema_infer as N
from .. import io as sio

SRC = OUT = LOG = None
EXT = (".parquet", ".jsonl", ".jsonl.gz")


def _rows_and_cols(path):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        cols = list(pq.ParquetFile(path).schema_arrow.names)
        return cols, sio.iter_parquet_rows(path)
    first = None
    with sio.open_text(path) as f:
        for line in f:
            if line.strip():
                first = json.loads(line)
                break
    cols = list(first.keys()) if isinstance(first, dict) else []

    def it():
        with sio.open_text(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    return cols, it()


def convert_file(rel):
    src = os.path.join(SRC, rel)
    base = rel[: -len(".jsonl.gz")] if rel.endswith(".jsonl.gz") else os.path.splitext(rel)[0]
    dst = os.path.join(OUT, base + ".jsonl")
    c = Counter()
    cols, rows = _rows_and_cols(src)
    p = N.plan(cols)
    if p["mode"] is None:
        return rel, {"no_adapter": 1}, cols
    tmp = sio.atomic_target(dst)
    with open(tmp, "w", encoding="utf-8") as fo:
        for r in rows:
            c["rows_in"] += 1
            msgs = N.to_messages(r, p)
            if not msgs:
                c["unmappable"] += 1
                continue
            fo.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            c["rows_out"] += 1
    if c["rows_out"]:
        sio.commit(tmp, dst)
    else:
        os.remove(tmp)          # no usable row: no empty output file
    c["mode:" + p["mode"]] += 1
    return rel, dict(c), None


def main(argv=None):
    global SRC, OUT, LOG
    import sys
    nproc = int((argv or sys.argv[1:] or [8])[0])
    SRC, OUT = os.environ["SFTPIPE_SRC"], os.environ["SFTPIPE_OUT"]
    LOG = os.environ.get("SFTPIPE_LOG", OUT.rstrip("/") + "_logs")
    os.makedirs(LOG, exist_ok=True)
    rels = [os.path.relpath(p, SRC) for p in sio.list_files(SRC, EXT)]
    log = os.path.join(LOG, "convert_log.jsonl")
    done = {json.loads(l)["file"] for l in open(log)} if os.path.exists(log) else set()
    tot = Counter()
    with Pool(nproc) as pool, open(log, "a") as lf, open(os.path.join(LOG, "no_adapter.jsonl"), "a") as na:
        for rel, c, cols in pool.imap_unordered(convert_file, [r for r in rels if r not in done]):
            tot.update(c)
            lf.write(json.dumps({"file": rel, **c}) + "\n"); lf.flush()
            if cols is not None:
                na.write(json.dumps({"file": rel, "columns": cols}) + "\n")
    print(json.dumps(dict(sorted(tot.items())), indent=1))


if __name__ == "__main__":
    main()
