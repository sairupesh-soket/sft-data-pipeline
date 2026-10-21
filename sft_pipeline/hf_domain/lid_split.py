"""
Splits every shard in the source folder (env SFTPIPE_SRC; output SFTPIPE_OUT, progress SFTPIPE_LOG,
fastText model LID_MODEL = lid.176.bin)
by predicted language (fasttext lid.176.bin) into
sft_language_wise/<lang_code>/<same_shard_filename>.jsonl -- one jsonl per
source shard per language, so language folders end up with multiple jsonls
(one per shard that contributed rows), not one giant merged file.

Read-only against the source data; only ever writes under
sft_language_wise/ and lang_id_progress/ inside this project folder.

Resumable: each shard writes a small marker to lang_id_progress/ when fully
done, so a killed/restarted run skips shards already completed rather than
redoing 991GB of work from scratch.

LID input per row: all message "content" fields concatenated (covers
multi-turn rows and mixed-role text), newlines flattened to spaces (fasttext
predict() only reads up to the first newline), capped at 2000 chars -- language
identification doesn't need the whole document, just enough real text.
"""
import os
import sys
import json
import glob
import time
from multiprocessing import Pool

import fasttext

SRC_DIR = os.environ.get("SFTPIPE_SRC", "")
OUT_DIR = os.environ.get("SFTPIPE_OUT", "")
PROGRESS_DIR = os.environ.get("SFTPIPE_LOG", "")
LID_MODEL_PATH = os.environ.get("LID_MODEL", "")
LID_TEXT_CAP = 2000
N_WORKERS = 16
CONFIDENCE_THRESHOLD = 0.5  # below this, route to "uncertain" instead of guessing
MIN_ALPHA_RATIO = 0.3       # below this fraction of alphabetic chars, treat as no real language

_model = None  # loaded once per worker process, not per row


def get_model():
    global _model
    if _model is None:
        _model = fasttext.load_model(LID_MODEL_PATH)
    return _model


def extract_lid_text(row):
    msgs = row.get("messages")
    if not msgs:
        return ""
    parts = [m.get("content", "") for m in msgs if isinstance(m, dict) and m.get("content")]
    text = " ".join(parts)
    text = text.replace("\n", " ").replace("\r", " ")
    return text[:LID_TEXT_CAP].strip()


def looks_like_natural_language(text):
    """Cheap pre-check: pure arithmetic/symbol/number content (e.g. a math
    problem's raw numbers) has no real language to detect at all -- confirmed
    real case: '34591+321542452 34 591 + ...' got assigned a random low-
    confidence language guess (Occitan, 12%) instead of being recognized as
    having no natural language content in the first place."""
    if not text:
        return False
    alpha = sum(1 for c in text if c.isalpha())
    return (alpha / len(text)) >= MIN_ALPHA_RATIO


def classify_language(model, lid_text):
    """Returns the bucket name a row should go into: a real lang code, or
    'no_natural_language' (mostly digits/symbols, nothing to detect), or
    'uncertain' (top prediction confidence too low to trust -- confirmed real
    cases: English math word problems misclassified as Cebuano at 55%
    confidence, and English translation-task instructions wrapping embedded
    foreign example text landing on near-random low-confidence guesses like
    Scots/Mazanderani at 12-30%)."""
    if not lid_text:
        return "unknown"
    if not looks_like_natural_language(lid_text):
        return "no_natural_language"
    try:
        labels, probs = model.predict(lid_text, k=1)
    except Exception:
        return "unknown"
    if not labels:
        return "unknown"
    confidence = float(probs[0])
    if confidence < CONFIDENCE_THRESHOLD:
        return "uncertain"
    return labels[0].replace("__label__", "")


def process_shard(shard_path):
    shard_name = os.path.basename(shard_path)
    marker = os.path.join(PROGRESS_DIR, shard_name + ".done")
    if os.path.exists(marker):
        return (shard_name, "skipped-already-done", 0, 0, {})

    model = get_model()
    lang_counts = {}
    n_total = 0
    n_bad = 0
    open_files = {}

    t0 = time.time()
    try:
        with open(shard_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                n_total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    n_bad += 1
                    continue

                lid_text = extract_lid_text(row)
                lang = classify_language(model, lid_text)

                lang_counts[lang] = lang_counts.get(lang, 0) + 1

                if lang not in open_files:
                    lang_dir = os.path.join(OUT_DIR, lang)
                    os.makedirs(lang_dir, exist_ok=True)
                    open_files[lang] = open(os.path.join(lang_dir, shard_name), "w", encoding="utf-8")
                open_files[lang].write(line + "\n")
    finally:
        for fh in open_files.values():
            fh.close()

    os.makedirs(PROGRESS_DIR, exist_ok=True)
    with open(marker, "w") as f:
        json.dump({"shard": shard_name, "n_total": n_total, "n_bad_json": n_bad,
                   "lang_counts": lang_counts, "seconds": time.time() - t0}, f, indent=2)

    dt = time.time() - t0
    return (shard_name, "done", n_total, n_bad, {"seconds": round(dt, 1), "n_langs": len(lang_counts)})


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PROGRESS_DIR, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(SRC_DIR, "*.jsonl")))
    print(f"found {len(shards)} shard files to process, {N_WORKERS} parallel workers", flush=True)

    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(process_shard, shards), 1):
            shard_name, status, n_total, n_bad, extra = result
            el = time.time() - t0
            print(f"[{i}/{len(shards)}] {shard_name}: {status} rows={n_total} bad_json={n_bad} {extra}  "
                  f"(elapsed {el/60:.1f} min)", flush=True)

    print(f"\nDONE: all {len(shards)} shards processed in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
