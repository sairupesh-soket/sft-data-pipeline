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
   │  3. validate  sft_pipeline/validate   68 rules; FLAGS rows (file + line + reasons), changes nothing
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

## Validation rules (68)

The validation only FLAGS rows (file, line, rule ids). It does not change or remove data.
A row that passes all FLAG rules is in the proper OpenAI format: reasoning only in `reasoning_content`, tool calls only in `tool_calls`, tool results only in `role: "tool"` messages, and no template tokens or hidden turns in the text.
Each rule has a flagged and a correct example row (OpenAI format) in `docs/validation_rules.py`; `tests/test_rule_examples.py` checks them against the real code.

| Group | Rules | Label |
|---|---|---|
| Conversation structure | `AUDIT_does_not_end_with_assistant`, `AUDIT_system_then_assistant`, `AUDIT_first_turn_not_user`, `AUDIT_system_not_first`, `AUDIT_user_user`, `AUDIT_assistant_assistant` | FLAG |
| OpenAI format | `FORMAT_bad_shape`, `AUDIT_empty_messages`, `AUDIT_non_openai_top_key`, `AUDIT_unknown_role`, `AUDIT_unknown_msg_key`, `AUDIT_content_not_string`, `AUDIT_reasoning_on_non_assistant`, `AUDIT_args_not_json_string`, `AUDIT_args_invalid_json`, `TOOL_malformed`, `AUDIT_tool_call_no_id`, `AUDIT_tool_call_no_name`, `FORMAT_tool_call_type`, `AUDIT_tool_call_not_in_tools`, `AUDIT_tool_calls_but_no_tools`, `AUDIT_tool_def_bad_shape`, `AUDIT_tool_without_call`, `AUDIT_tool_without_id`, `AUDIT_tool_id_mismatch`, `TOOL_call_without_reply`, `TOOL_duplicate_call_id`, `FORMAT_tool_message_name` | FLAG |
| Empty content | `AUDIT_empty_user`, `AUDIT_empty_assistant`, `AUDIT_final_answer_empty` | FLAG |
| Reasoning in the answer text | `THINK_in_content`, `AUDIT_think_tag_left`, `AUDIT_other_reasoning_tags`, `EMBED_reasoning_tag`, `REASON_in_answer`, `GEN_artifact_in_answer`, `RC_equals_answer`, `AUDIT_reasoning_is_problem_creation` | FLAG |
| Tool calls, tool results and turns hidden in the text | `TOOLCALL_as_text`, `EMBED_tool_call_text`, `EMBED_tool_result_text`, `EMBED_role_turns`, `EMBED_agent_text_protocol`, `SYS_tools_promised_but_no_tools`, `USER_embedded_transcript`, `USER_python_repr_messages` | FLAG |
| Special tokens and template render | `AUDIT_special_token_leak`, `EKA_T1`, `RENDER` | FLAG |
| Text quality | `AUDIT_replacement_char`, `AUDIT_mojibake`, `AUDIT_control_chars`, `AUDIT_repetition_loop`, `AUDIT_harness_error_text`, `AUDIT_output_limit_loop`, `AUDIT_degenerate_answer`, `AUDIT_answer_equals_prompt`, `LITERAL_backslash_n`, `CODE_newlines_collapsed`, `PDF_hard_line_breaks` | FLAG |
| Safety, identity and sources | `BACKDOOR`, `IDENTITY`, `SRC_incorrect_solutions` | FLAG |
| Information only | `END_single_call`, `END_traj_stops`, `AUDIT_unfilled_template`, `INFO_as_an_AI_language_model` | INFO |

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
| `docs/` | `SFT_PIPELINE.pdf` (conversion + cleaning + validation), `SFT_VALIDATION_RULES.pdf`, their builders |

## Add a new source

1. Try `generic` first. Read `<out>_logs/no_adapter.jsonl` for files it cannot map.
2. If the schema is special, add a converter (see `adapters/traces/extra*.py` or the per-dataset table in
   `adapters/agentic.py`): read real rows first, map them to messages with `openai_format`, write one JSONL per source file.
3. Run `clean`, then `validate`. Read the flagged samples before you trust a count.

## Tests

```bash
python3 tests/test_pipeline.py          # repair steps and adapter helpers (before / after)
python3 tests/test_validate_format.py   # 44 OpenAI-format breaks are flagged, good rows pass
python3 tests/test_rule_examples.py     # every documented rule example behaves as documented
python3 tests/test_embedded.py          # 49 hidden-content formats are flagged, plain answers pass
```
