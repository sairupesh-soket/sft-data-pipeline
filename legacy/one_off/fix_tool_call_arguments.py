"""Post-process fix: tool_calls[].function.arguments must be a dict/object,
not a JSON string, for the actual GLM chat template to render (confirmed via
AutoTokenizer.apply_chat_template against the real zai-org/GLM-5.2-FP8
tokenizer -- a string there raises
"UndefinedError: 'str object' has no attribute 'items'" because the template
does tc.arguments.items() directly).

This was already fixed at the source in traces_to_openai_sft.py's
normalize_tool_calls(), but files converted before that fix still have the
old (OpenAI-spec-correct, GLM-template-incompatible) string form. Rather than
re-reading 72GB of source parquet again, this rewrites just the affected
lines in the already-produced JSONL files in place.

Only rewrites a file if at least one line actually changes; otherwise leaves
it untouched. Uses a temp file + atomic rename per dataset file so a kill
mid-run can't leave a half-written file.
"""

import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

OUT_DIR = Path(os.environ.get("SFTPIPE_PATH", "."))


def fix_arguments(args):
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"value": args}
    elif args is None:
        args = {}
    if not isinstance(args, dict):
        args = {"value": args}
    return args


def fix_line(obj):
    changed = False
    for m in obj.get("messages", []):
        tcs = m.get("tool_calls")
        if not tcs:
            continue
        for tc in tcs:
            fn = tc.get("function")
            if not fn:
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                fn["arguments"] = fix_arguments(args)
                changed = True
    return obj, changed


def process_file(path_str):
    path = Path(path_str)
    tmp_path = path.with_suffix(".jsonl.tmp")
    any_changed = False
    rows = 0
    try:
        # Stream line-by-line, writing each line straight to the temp file as
        # we go -- never hold the whole file in memory. The first version of
        # this script accumulated every line into a Python list before
        # writing it all out at once, which for the 121GB nvidia file grew
        # past 220GB of RAM (JSON parsing overhead) and had to be killed.
        with open(path, "r") as fin, open(tmp_path, "w") as fout:
            for line in fin:
                line = line.rstrip("\n")
                if not line:
                    continue
                rows += 1
                if '"tool_calls"' not in line:
                    fout.write(line + "\n")
                    continue
                obj = json.loads(line)
                obj, changed = fix_line(obj)
                if changed:
                    any_changed = True
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                else:
                    fout.write(line + "\n")
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return {"file": path.name, "rows": rows, "changed": False, "error": str(e)}

    if any_changed:
        os.replace(tmp_path, path)
    else:
        tmp_path.unlink(missing_ok=True)

    return {"file": path.name, "rows": rows, "changed": any_changed, "error": None}


def main():
    files = sorted(str(p) for p in OUT_DIR.glob("*.jsonl"))
    print(f"Scanning {len(files)} files for string tool_call arguments...")

    changed_files = 0
    errors = []
    with ProcessPoolExecutor(max_workers=32) as ex:
        futs = {ex.submit(process_file, f): f for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res["error"]:
                errors.append(res)
                print(f"[{i}/{len(files)}] ERROR {res['file']}: {res['error']}")
            elif res["changed"]:
                changed_files += 1
                print(f"[{i}/{len(files)}] FIXED {res['file']} ({res['rows']} rows)")

    print(f"\nDone. Files changed: {changed_files}/{len(files)}. Errors: {len(errors)}")
    if errors:
        for e in errors:
            print(" ", e)


if __name__ == "__main__":
    main()
