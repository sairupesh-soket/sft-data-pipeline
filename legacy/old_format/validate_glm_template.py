"""Render a sample of every converted dataset through the real GLM-5.2 chat
template to confirm the OpenAI-format JSONL actually loads cleanly for SFT --
not just that it's valid JSON. Samples first + last N lines of every
dataset file (schema is consistent within a dataset, so this catches
cross-dataset diversity without reading everything).
"""

import os
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

TOK_PATH = os.environ.get("SFTPIPE_PATH", ".")
OUT_DIR = Path(os.environ.get("SFTPIPE_PATH", "."))
SAMPLE_PER_FILE = 5
SKIP_FILES = set()  # empty once the full pipeline run has finished


def sample_lines(path, n):
    lines = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i < n:
                lines.append(line)
            else:
                break
    if len(lines) < n:
        return lines
    # also grab a tail sample for datasets with >2n lines. Some datasets have
    # very long lines (~70KB+), so read a generous window and always drop the
    # first split fragment -- it may be a partially-read (truncated) line.
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        chunk = min(size, 4_000_000)
        f.seek(max(0, size - chunk))
        data = f.read().decode("utf-8", errors="ignore")
    parts = data.split("\n")
    if size > chunk:
        parts = parts[1:]  # first fragment may be truncated mid-line
    tail_lines = [p for p in parts if p.strip()][-n:]
    return lines + tail_lines


def main():
    tok = AutoTokenizer.from_pretrained(TOK_PATH, trust_remote_code=True)

    files = sorted(p for p in OUT_DIR.glob("*.jsonl") if p.name not in SKIP_FILES)
    print(f"Validating against {len(files)} dataset files, {SAMPLE_PER_FILE*2} samples each (max)")

    total = 0
    ok = 0
    error_examples = defaultdict(list)
    error_counts = Counter()
    failing_datasets = set()

    for fp in files:
        try:
            lines = sample_lines(fp, SAMPLE_PER_FILE)
        except Exception as e:
            print(f"COULD NOT SAMPLE {fp.name}: {e}")
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                error_counts["json_parse_error"] += 1
                if len(error_examples["json_parse_error"]) < 3:
                    error_examples["json_parse_error"].append((fp.name, str(e)))
                failing_datasets.add(fp.name)
                total += 1
                continue
            msgs = obj.get("messages")
            total += 1
            try:
                tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
                ok += 1
            except Exception as e:
                key = f"{type(e).__name__}: {str(e)[:120]}"
                error_counts[key] += 1
                if len(error_examples[key]) < 3:
                    error_examples[key].append((fp.name, json.dumps(msgs, ensure_ascii=False)[:400]))
                failing_datasets.add(fp.name)

    print(f"\n======== RESULT ========")
    print(f"Total sampled rows: {total}")
    print(f"Rendered+tokenized OK: {ok} ({100*ok/total:.2f}%)" if total else "no rows")
    print(f"Failing datasets: {len(failing_datasets)} / {len(files)}")
    print("\nError breakdown:")
    for k, v in error_counts.most_common():
        print(f"  {v:6d}  {k}")
        for fname, ex in error_examples[k]:
            print(f"           e.g. [{fname}] {ex}")

    with open(OUT_DIR / "glm_template_validation_report.json", "w") as f:
        json.dump(
            {
                "total_sampled": total,
                "ok": ok,
                "failing_datasets": sorted(failing_datasets),
                "error_counts": dict(error_counts),
                "error_examples": {k: v for k, v in error_examples.items()},
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
