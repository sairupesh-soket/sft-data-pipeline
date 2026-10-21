# sft-validation

Read-only validation of SFT training data in OpenAI chat format (JSONL).
The scripts find rows that can corrupt or confuse the model during SFT, count them,
write one list with every bad row (file + line + reasons), and write sample rows for each issue.
The scripts never change the input data.

## Data format that the rules expect

- One JSON object per line: `{"messages": [...], "tools": [...]}` (`tools` only if tools are available).
- Roles: `system`, `user`, `assistant`, `tool`.
- Reasoning is always in `reasoning_content` of the assistant message, never in `content`.
- Tool calls are always in `tool_calls` (`arguments` is a JSON string); results are `role: "tool"` messages with `tool_call_id`.

## Usage

```bash
pip install -r requirements.txt
python3 sft_validation/dataset_check.py ROOT OUT NPROC [RENDER_EVERY]
```

- `ROOT`: folder with `.jsonl` files (all sub-folders are read; folders that start with `_` are skipped).
- `OUT`: output folder (must not be inside `ROOT`).
- `NPROC`: number of worker processes. Use a compute node for big data, not a login node.
- `RENDER_EVERY`: render 1 row in N with the eka chat template (default 100; use 1 for small data).

Tokenizer and template paths (for the render check):

```bash
export EKA_SPM=/path/to/eka_v0_1_1.model
export EKA_TEMPLATE=/path/to/eka_v0_2_0/chat_template.jinja
```

The base audit can also run alone: `python3 sft_validation/final_audit.py ROOT OUTDIR [NPROC]`.

## Output

| File | Content |
|---|---|
| `ISSUES_REPORT.txt` | rows per issue, top sources per issue, stats (reasoning / tool calls / turns / length) |
| `REMOVE_SUMMARY.txt` | rows with at least one real issue, rows per reason |
| `ROWS_TO_REMOVE.txt` | every row with a real issue: file, line (1-based), byte offset, reasons |
| `ROWS_INFO_ONLY.txt` | rows that have info flags only (keep) |
| `samples/<issue>.txt` | up to 10 full rows per issue, from different sources |
| `counts.json` | all counters (machine readable) |

## Rules

| Group | Rules | Action |
|---|---|---|
| Structure (A) | does_not_end_with_assistant, system_then_assistant, first_turn_not_user, system_not_first, user_user, assistant_assistant | drop |
| OpenAI format (B) | empty_messages, non_openai_top_key, unknown_msg_key, unknown_role, content_not_string, reasoning_not_string, reasoning_on_non_assistant, args_not_json_string, args_invalid_json, tool_call_no_id, tool_call_no_name, tool_without_call, tool_without_id, tool_id_mismatch, tool_call_not_in_tools, tool_calls_but_no_tools, tool_def_bad_shape, TOOL_malformed | drop |
| Empty (C) | empty_user, empty_assistant, final_answer_empty | drop |
| Reasoning | THINK_in_content / think_tag_left, other_reasoning_tags, REASON_in_answer, GEN_artifact_in_answer, RC_equals_answer, reasoning_is_problem_creation | drop or fix |
| Tokens / template | special_token_leak, EKA_T1_<field> (eka user-defined template pieces as text), RENDER_mismatch (eka render check) | drop |
| Text quality (D) | replacement_char, mojibake, control_chars, repetition_loop, harness_error_text, output_limit_loop, degenerate_answer, answer_equals_prompt, LITERAL_backslash_n, CODE_newlines_collapsed, PDF_hard_line_breaks | drop or fix |
| Tool use content | SYS_tools_promised_but_no_tools, TOOLCALL_as_text | drop |
| Conversation content | USER_embedded_transcript, USER_python_repr_messages | drop |
| Safety / identity / source | BACKDOOR, IDENTITY_<name> (strict rule), SRC_incorrect_solutions | drop |
| Info only (row is kept) | END_single_call, END_traj_stops (row ends on a tool call), unfilled_template, IDENTITY_userrule (old broad rule), INFO_as_an_AI_language_model, selftalk_in_answer, fake_turn_in_answer, answer_in_prompt, null_in_args, very_long_row | keep |

## Important notes

- **Check precision before you trust a count.** Read random flagged rows by hand for each rule.
  Known false positives: literal `\n` inside quoted strings or LaTeX (`\neq`), PDF rule on code without a fence,
  the old broad identity rule (`I'm using OpenAI's API`).
- **A repair can create damage.** Never replace literal `\n` inside a quoted string or a JSON value:
  there it is a correct escape (`csv = "a,b\n1,2"`, `{"keystrokes": "ls -la\n"}`).
- The rules check format and text. They do not check if an answer is correct (facts, math, code results).

## Files

| File | Content |
|---|---|
| `sft_validation/dataset_check.py` | main script: all rules, stats, row lists, samples |
| `sft_validation/final_audit.py` | base audit, 44 rules (groups A to E); also usable alone |
| `sft_validation/render_check.py` | eka tokenizer + chat template render check |
| `sft_validation/identity_source_rules.py` | known-wrong source datasets, old broad identity rule |
| `sft_validation/common.py` | file splitting, sample formatting, tool-argument check, reasoning markers |
