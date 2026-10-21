# sft-data-pipeline

Reusable code to prepare SFT training data in the **OpenAI chat format**:

```
raw sources (parquet / jsonl, many schemas)
   │  1. convert   sft_pipeline/adapters   schema -> {"messages": [...], "tools": [...]}
   ▼
OpenAI-format JSONL
   │  2. clean     sft_pipeline/repairs    known format problems repaired (output folder; source is not changed)
   ▼
clean JSONL
   │  3. validate  sft_pipeline/validate   62 rules; FLAGS rows (file + line + reasons), changes nothing
   ▼
issue lists, samples, stats  (4. stats: rows, o200k tokens, turns, tool calls, reasoning)
```

The validation only **flags** rows. It does not decide to fix or remove them.

## Target format

```json
{"messages": [
   {"role": "system", "content": "..."},
   {"role": "user", "content": "..."},
   {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function",
        "function": {"name": "get_time", "arguments": "{\"city\": \"Rome\"}"}}]},
   {"role": "tool", "tool_call_id": "call_0", "content": "{\"time\": \"10:00\"}"},
   {"role": "assistant", "content": "It is 10:00 in Rome.", "reasoning_content": "The tool says 10:00."}],
 "tools": [{"type": "function", "function": {"name": "get_time", "description": "...", "parameters": {...}}}]}
```

- Reasoning is always in `reasoning_content`, never in `content`.
- Tool calls are in `tool_calls`; `arguments` is a JSON string; tool results are `role: "tool"` messages.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. convert (pick the adapter for the source)
python3 -m sft_pipeline.convert generic --src RAW --out CONVERTED [NPROC]
python3 -m sft_pipeline.convert traces  --src RAW --out CONVERTED
python3 -m sft_pipeline.convert agentic --src RAW --out CONVERTED [DATASET ...] [--limit N]
python3 -m sft_pipeline.convert posttraining    --src RAW --out CONVERTED [NPROC]
python3 -m sft_pipeline.convert posttraining_b2 --src RAW --out CONVERTED [NPROC]

# 2. clean (repair steps in the order of the source family; --gate also removes rows that fail a FLAG rule)
python3 -m sft_pipeline.clean CONVERTED CLEAN --preset agentic --nproc 32

# 3. validate (read only)
export EKA_SPM=/path/eka_v0_1_1.model EKA_TEMPLATE=/path/chat_template.jinja
python3 -m sft_pipeline.validate CLEAN REPORT_DIR 32 100

# 4. stats
python3 -m sft_pipeline.stats STATS_DIR mydata=CLEAN
```

### Production recipes (verified end to end: same rows as the production data)

| Family | Commands |
|---|---|
| HF agentic | `convert agentic` -> `clean --preset agentic` |
| HF traces | `convert traces` -> `convert traces_extra1..9` -> `convert traces_post` -> `clean --preset traces` |
| posttraining | `convert posttraining` (+ `posttraining_b2`) -> `clean --preset posttraining` |
| hf_domain | (external normalize) -> `hf_domain/lid_split.py` -> `hf_domain/strict_filter.py` (production flags in docs) -> `hf_domain/python_tool_postpass.py --apply` -> `clean --preset hf_domain` |
| new data, current format | `clean --preset current` |

Logs never go into a data folder: by default `<out>_logs/` next to the output.
Folders and files that start with `_` are skipped as input. Every stage resumes after a kill.

## Layout

| Path | Content |
|---|---|
| `sft_pipeline/openai_format.py` | message normalization shared by the adapters (roles, tool calls, tool definitions, ChatML, tool ids) |
| `sft_pipeline/adapters/` | `generic` + `schema_infer` (any schema from column names), `traces/` (base + extra1..9), `agentic`, `posttraining`, `posttraining_b2` |
| `sft_pipeline/repairs/` | `format`, `shapes`, `tools` (ids + Hermes), `think_content`, `reasoning`, `roles`, `final`, `text_drops` |
| `sft_pipeline/clean.py` | runs the repair steps in order |
| `sft_pipeline/validate/` | validation rules (flag only), eka render check, issue lists, samples |
| `sft_pipeline/hf_domain/` | language split (fastText), strict OpenAI filter, quality filter, post-passes |
| `sft_pipeline/stats.py` | statistics workbook |
| `legacy/` | one-off fixes for single datasets; scripts of the old storage format |
| `tests/` | synthetic before/after cases for every repair step and adapter helper |
| `docs/` | full documentation (PDF) |

## Add a new source

1. Try `generic` first. Read `<out>_logs/no_adapter.jsonl` for files it cannot map.
2. If the schema is special, add a converter (see `adapters/traces/extra*.py` or the per-dataset table in
   `adapters/agentic.py`): read real rows first, map them to messages with `openai_format`, write one JSONL per source file.
3. Run `clean`, then `validate`. Read the flagged samples before you trust a count.

## Tests

```bash
python3 tests/test_pipeline.py
```
