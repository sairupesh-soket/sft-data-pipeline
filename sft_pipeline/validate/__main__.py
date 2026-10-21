"""python3 -m sft_pipeline.validate ROOT OUT NPROC [RENDER_EVERY]  - flag rows (read only). See dataset_check.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_check  # noqa: E402

if len(sys.argv) < 4:
    raise SystemExit(__doc__)
dataset_check.main()
