# Convert HF trace parquets → OpenAI supervised fine-tuning JSONL

## What it does

1. Reads all `*.parquet` under the INPUT dir (default: Traces `OUTPUT/`)
2. Maps heterogeneous schemas → OpenAI chat SFT:
   ```json
   {"messages":[{"role":"system|user|assistant|tool","content":"..."}], "tools":[...]}
   ```
3. Filters:
   - **Drops** samples with Claude / Gemini / other AI-vendor-centric **system** prompts
   - **Scrubs** model-name mentions from message **content** (keeps tool ids / tool call names)
   - **Strips** all non-SFT metadata (`model`, `timestamp`, rewards, etc.)
4. Writes JSONL under `openai_sft_out/by_dataset/<dataset>/...jsonl`
5. Writes `conversion_report.json`, `per_dataset_stats.json`, `errors.jsonl`, `convert.log`

## Run

```bash
cd /home/bharath/sft-hf-traces/openai_sft_pipeline

# Smoke test (5 files, no write)
python3 convert_to_openai_sft.py --limit-files 5 --dry-run

# Small real write
python3 convert_to_openai_sft.py --limit-files 20 --workers 4

# Full conversion
python3 convert_to_openai_sft.py --workers 8

# Custom paths (e.g. shared disk when writable)
python3 convert_to_openai_sft.py \
  --input-dir /projects/data/datasets/translation_data/Traces_datasets/OUTPUT \
  --output-dir /projects/data/datasets/translation_data/Traces_datasets/OPENAI_SFT \
  --workers 8 --resume
```

`parallel_tool_calls` is only added when `tools` are present (set `true` if any assistant turn has >1 tool call).
