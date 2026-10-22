# Trace SFT + mid-training utils

Convert Cursor / Langfuse traces into OpenAI SFT JSONL, then optionally linearize and tokenize for mid-training.

## 1. Traces → OpenAI SFT

```bash
python env_traj_utils/sft_pipeline.py path/to/structured_full_90d.json \
    -o traces_openai_sft.jsonl \
    --agent-only \
    --dropped dropped.jsonl \
    --report sft_report.json
```

Drops empty input/output traces on the fly and validates `system` / `user` / `assistant` / `tool` format.

## 2. Mid-training linearize

After SFT JSONL is ready, convert to linearized **Markdown** for mid-training:

```bash
python env_traj_utils/linearize_trajectories.py traces_openai_sft.jsonl \
    -o traces_linearized.md
```

Each sample is a Markdown section (`# Sample N`) with `USER` / `THINK` / `TOOL` / `ASSISTANT` text. System prompt is omitted by default (`--include-system` to keep it).

Also supported: `.jsonl` (`{"text": "..."}`) for tokenization, `.txt` for a plain dump.

## 3. Prompt mutations (Baseten-style, TensorStudio / Qwen)

Base seed is the general agent system prompt in `cursor_system_prompt.md` (also used by `sft_pipeline.py` as `CURSOR_SFT_SYSTEM`).

```bash
# A) Generate bank (OpenAI-compatible)
$env:OPENAI_BASE_URL="https://api.tensorstudio.ai/v1"
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="qwen3-30b"   # qwen3-coder-next is not on this gateway; use qwen3-30b / qwen25-7b

python prompt_mutations.py generate `
  --seed-file cursor_system_prompt.md `
  --count 12 `
  -o prompt_bank.json

# B) Apply onto SFT JSONL (trajectory turns unchanged)
python env_traj_utils/prompt_mutations.py apply `
  structured_5000_openai_sft.jsonl `
  --bank prompt_bank.json `
  -o structured_5000_openai_sft_mutated.jsonl
```

## 4. Tokenize → parquet

```bash
python env_traj_utils/tokenize_trajectories.py traces_linearized.jsonl \
    -m /path/to/tokenizer \
    --max-tokens 131072 \
    --parquet-output traces_midtrain.parquet
```

Linearized rows (`text`) are encoded as plain text. SFT rows (`messages`) still use the chat template.

## Files

| Script | Role |
|--------|------|
| `sft_pipeline.py` | traces JSON → validated OpenAI SFT JSONL |
| `adapt_langfuse_traces.py` | span mapping used by the SFT pipeline |
| `cursor_system_prompt.md` | Base agent system prompt (mutation seed; product-agnostic) |
| `prompt_mutations.py` | LLM / handwritten system-prompt bank + apply to JSONL |
| `linearize_trajectories.py` | SFT JSONL → mid-train Markdown (`.md`) linearized text |
| `tokenize_trajectories.py` | JSONL → token length stats + mid-train parquet |
