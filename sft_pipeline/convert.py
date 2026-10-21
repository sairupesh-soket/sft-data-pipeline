"""Convert raw SFT sources into OpenAI-format JSONL.

    python3 -m sft_pipeline.convert ADAPTER --src RAW_FOLDER --out OUT_FOLDER [--log LOG_FOLDER] [adapter args ...]

Adapters (sft_pipeline/adapters/):
  generic          any parquet / JSONL schema, found from the column names (schema_infer = sft_normalize_v4)
                   args: [NPROC]
  traces           HF "traces" datasets with a messages / conversations / conversation column; ShareGPT from/value
  traces_extra1..9 per-dataset converters for traces datasets with other columns (see each module docstring)
  traces_post      traces post-convert fixes (missing assistant turn, fimbulvntr, struct nulls, excluded files)
  agentic          HF agentic datasets: one explicit converter per dataset   args: [NAME ...] [--limit N]
  posttraining     codeLLM posttraining: A (native OpenAI rows) + B1 (reliable split markers)   args: [NPROC]
  posttraining_b2  codeLLM posttraining B2: MCQ / code-fence heuristic split   args: [NPROC]
The source folder is only read. After convert, run:  python3 -m sft_pipeline.clean OUT CLEAN_OUT
then:                                                python3 -m sft_pipeline.validate CLEAN_OUT REPORT_DIR
"""
import argparse
import importlib
import os
import sys

ADAPTERS = {
    "generic": "sft_pipeline.adapters.generic",
    "traces": "sft_pipeline.adapters.traces.base",
    "traces_post": "sft_pipeline.adapters.traces.post",
    "agentic": "sft_pipeline.adapters.agentic",
    "posttraining": "sft_pipeline.adapters.posttraining",
    "posttraining_b2": "sft_pipeline.adapters.posttraining_b2",
}
ADAPTERS.update({f"traces_extra{i}": f"sft_pipeline.adapters.traces.extra{i}" for i in range(1, 10)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter", choices=sorted(ADAPTERS))
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log")
    a, rest = ap.parse_known_args()
    src, out = os.path.abspath(a.src), os.path.abspath(a.out)
    if out == src or out.startswith(src + os.sep):
        raise SystemExit("--out must not be the source folder or inside it")
    os.environ["SFTPIPE_SRC"], os.environ["SFTPIPE_OUT"] = src, out
    os.environ["SFTPIPE_LOG"] = os.path.abspath(a.log) if a.log else out.rstrip("/") + "_logs"
    os.makedirs(out, exist_ok=True)
    mod = importlib.import_module(ADAPTERS[a.adapter])
    sys.argv = [mod.__name__] + rest
    mod.main()


if __name__ == "__main__":
    main()
