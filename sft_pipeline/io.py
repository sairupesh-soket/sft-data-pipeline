"""File helpers shared by all stages: list input files, open .jsonl / .jsonl.gz, read parquet rows,
write atomically (tmp file + rename)."""
import gzip
import os

DATA_EXT = (".jsonl", ".jsonl.gz")


def list_files(root, exts=DATA_EXT):
    """All data files under root, sorted. Folders and files whose name starts with "_" are skipped
    (logs, backups, rejects)."""
    out = []
    for d, dirs, fs in os.walk(root):
        dirs[:] = sorted(x for x in dirs if not x.startswith("_"))
        for f in sorted(fs):
            if f.endswith(exts) and not f.startswith("_"):
                out.append(os.path.join(d, f))
    return out


def open_text(path, mode="rt"):
    """Open a .jsonl or .jsonl.gz file in text mode (utf-8)."""
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", compresslevel=3) if "w" in mode else gzip.open(path, mode, encoding="utf-8")
    return open(path, mode.replace("t", ""), encoding="utf-8")


def open_bytes(path, mode="rb"):
    if path.endswith(".gz"):
        return gzip.open(path, mode, compresslevel=3) if "w" in mode else gzip.open(path, mode)
    return open(path, mode)


def atomic_target(path):
    """Return a tmp path next to path. Write there, then call commit(tmp, path)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path + ".tmp_sftpipe"


def commit(tmp, path):
    os.replace(tmp, path)


def iter_parquet_rows(path, columns=None, batch_size=2048):
    """Yield parquet rows as plain dicts (numpy / pyarrow values converted to Python)."""
    import pyarrow.parquet as pq
    from .openai_format import to_plain
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            yield to_plain(row)
