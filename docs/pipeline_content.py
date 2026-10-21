# -*- coding: utf-8 -*-
"""Text of the SFT data pipeline document (ASD-STE100 Simplified Technical English).
Block types: ("p", text) paragraph, ("b", [items]) bullets, ("code", text), ("h2", text), ("table", header, rows),
("note", text). Examples come from tests/cases.py (real output of the code)."""

OVERVIEW = [
    ("p", "This document tells how to prepare SFT (supervised fine-tuning) data with the sft-data-pipeline code. "
          "The pipeline reads raw datasets in many schemas. It writes one standard format: the OpenAI chat format. "
          "Then it repairs known format problems and flags the rows that still have problems."),
    ("p", "The code is in one repository. All stages use the same file reader, the same writer and the same command style. "
          "No stage has a fixed data path. You give the paths on the command line."),
    ("h2", "Stages"),
    ("table", ["Stage", "Command", "What it does"], [
        ["1. Convert", "python3 -m sft_pipeline.convert ADAPTER", "Reads the raw files and writes OpenAI-format rows. One adapter for each source family."],
        ["2. Clean", "python3 -m sft_pipeline.clean", "Runs the repair steps in a fixed order. Writes a new folder. The source does not change."],
        ["3. Validate", "python3 -m sft_pipeline.validate", "Checks every row with 68 rules. Flags the bad rows (file, line, rule). Changes nothing."],
        ["4. Stats", "python3 -m sft_pipeline.stats", "Counts rows, tokens, turns, tool calls and reasoning rows. Writes a workbook."],
    ]),
    ("note", "The validation only flags rows. It does not decide to fix or remove data. The data owners decide."),
    ("h2", "General rules for all stages"),
    ("b", ["The source folder is never changed. Each stage writes to a new output folder (in-place mode needs an explicit flag).",
           "Logs and reports go to a separate folder (default: <out>_logs). They never go into a data folder.",
           "Input folders and files whose name starts with \"_\" are skipped (logs, backups, rejects).",
           "Each stage writes a progress log. If a job stops, run the same command again: it continues from the last finished file.",
           "Each file is written to a temporary file first, then renamed. A stopped job never leaves a half-written file.",
           "Run big jobs on a compute node, not on a login node. Use one node per job."]),
]

FORMAT = [
    ("p", "All stages produce this format. One row is one JSON object on one line (JSONL)."),
    ("code", '{"messages": [\n'
             '   {"role": "system", "content": "You are a helpful assistant."},\n'
             '   {"role": "user", "content": "Time in Rome?"},\n'
             '   {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function",\n'
             '        "function": {"name": "get_time", "arguments": "{\\"city\\": \\"Rome\\"}"}}]},\n'
             '   {"role": "tool", "tool_call_id": "call_0", "content": "{\\"time\\": \\"10:00\\"}"},\n'
             '   {"role": "assistant", "content": "It is 10:00 in Rome.",\n'
             '    "reasoning_content": "The tool says 10:00."}],\n'
             ' "tools": [{"type": "function", "function": {"name": "get_time",\n'
             '    "description": "Current time in a city", "parameters": {"type": "object", ...}}}]}'),
    ("table", ["Field", "Rule"], [
        ["messages", "List of messages. Roles: system (only first), user, assistant, tool. The row ends with an assistant message."],
        ["content", "Always a string. The visible text only. No reasoning, no tool-call text, no template tokens."],
        ["reasoning_content", "Reasoning of the assistant, in its own field. Only on assistant messages. Never empty."],
        ["tool_calls", "List of {id, type: \"function\", function: {name, arguments}}. arguments is a JSON string."],
        ["tool messages", "role \"tool\", tool_call_id of the call, content = the tool result."],
        ["tools", "Top-level list of tool definitions {type: \"function\", function: {name, description, parameters}}."],
    ]),
    ("p", "The training chat template (eka) changes this format into tokens: it writes reasoning_content as <think>...</think> "
          "and tool_calls as <tool_call> blocks. Thus, the data itself must never contain these tags as text."),
]

ADAPTERS_INTRO = [
    ("p", "An adapter reads one family of raw sources and writes OpenAI-format rows. All adapters use the same message "
          "normalization (sft_pipeline/openai_format.py):"),
    ("b", ["Role names are mapped: human -> user, gpt / chatgpt / bot -> assistant, observation / function / ipython -> tool, developer -> system.",
           "Tool calls become {id, type, function{name, arguments}}; arguments become a JSON string.",
           "The legacy OpenAI field function_call is read as a tool call.",
           "Tool definitions become a top-level \"tools\" list in the OpenAI shape.",
           "Reasoning fields (reasoning_content, reasoning, think) go to reasoning_content.",
           "Parquet struct-union nulls are removed (only for values that came from a parquet struct column).",
           "An assistant text turn and the next tool-call-only turn are joined; tool replies get the ids of their calls."]),
    ("p", "After convert, always run clean. Some adapters write reasoning as <think>...</think> in the answer; "
          "the clean step move_think puts it into reasoning_content."),
]

ADAPTERS = [
    dict(name="generic", module="adapters/generic.py + adapters/schema_infer.py",
         input="Any parquet / JSONL file. The schema is found from the column names.",
         what=["For each file, plan() looks at the column names once. It finds a conversation column "
               "(messages, conversations, ShareGPT turns, ChatML or Llama-3 text, \"User:/Assistant:\" transcripts, "
               "\"### Prompt / ### Response\" text) or a field pair (instruction + input, question, prompt, query ... -> "
               "answer, response, output, solution ...; optional system and reasoning columns).",
               "to_messages() applies the plan to each row. A bad row gives no output and is counted as unmappable.",
               "Files with no usable plan are listed in <log>/no_adapter.jsonl with their column names."],
         cmd="python3 -m sft_pipeline.convert generic --src RAW --out CONVERTED 32"),
    dict(name="traces", module="adapters/traces/base.py",
         input="HF \"traces\" datasets (parquet) with a messages / conversations / conversation column.",
         what=["Converts the chat column of each dataset. ShareGPT from/value keys are mapped to role/content.",
               "Datasets without a chat column are listed in skipped_datasets_report.json for a special converter.",
               "reasoning_content is kept as its own field."],
         cmd="python3 -m sft_pipeline.convert traces --src RAW --out CONVERTED"),
    dict(name="traces_extra1 ... traces_extra9", module="adapters/traces/extra1.py ... extra9.py",
         input="Traces datasets with other column names. Each module converts a named list of datasets.",
         what=["Each module docstring tells, for each dataset, which columns hold the question, the reasoning and the answer, "
               "and which rows are used (for example only rows with a correctness field that says the answer is right)."],
         cmd="python3 -m sft_pipeline.convert traces_extra3 --src RAW --out CONVERTED"),
    dict(name="agentic", module="adapters/agentic.py",
         input="HF agentic datasets (parquet): agent trajectories, tool-calling data, chat and Q/A data.",
         what=["One explicit converter for each dataset (table below). There is no generic pass, because these sources mix "
               "many shapes.",
               "Datasets that are benchmarks, image data or catalogs are skipped with a reason (SKIP table in the module)."],
         cmd="python3 -m sft_pipeline.convert agentic --src RAW --out CONVERTED [DATASET ...] [--limit 200]"),
    dict(name="posttraining", module="adapters/posttraining.py",
         input="codeLLM posttraining datasets (jsonl, parquet, arrow).",
         what=["A (native): rows that already have an OpenAI-style messages list; Hermes <tools>/<tool_call>/<tool_response> "
               "text is changed to real tools / tool_calls / tool messages; for RM data only the \"chosen\" side is used.",
               "B1 (marker mapped): text blobs split only on reliable markers: <extra_id_1>User/Assistant, input: / output:, "
               "a leading Question: + Answer:, prompt + <think>...</think>.",
               "Role rules: the row ends on an assistant turn; no assistant turn directly after the system prompt."],
         cmd="python3 -m sft_pipeline.convert posttraining --src RAW --out CONVERTED 64"),
    dict(name="posttraining_b2", module="adapters/posttraining_b2.py",
         input="Text blobs that B1 cannot split on a reliable marker.",
         what=["mcq: split after the first option block (A/B/C...; the block ends at the first blank line); a leading "
               "\"Answer:\" label is removed.",
               "code_fence: a prompt followed by a final code block; the last code block is the reply.",
               "Rows whose reply does not end cleanly are not written."],
         cmd="python3 -m sft_pipeline.convert posttraining_b2 --src RAW --out CONVERTED 64"),
]

REPAIRS_INTRO = [
    ("p", "The clean stage runs repair steps over every row, in the order below. Each step reads one row and gives one of "
          "three results: keep (no change; the original line is written byte for byte), changed (the new row is written), "
          "or drop (the row is not written; it goes with its reason to <log>/dropped/<file>)."),
    ("code", "python3 -m sft_pipeline.clean CONVERTED CLEAN --preset agentic --nproc 32\n"
             "python3 -m sft_pipeline.clean CONVERTED CLEAN --preset current --skip hermes\n"
             "python3 -m sft_pipeline.clean CONVERTED CLEAN --steps format,move_think,role_rules"),
    ("p", "The step order is important: the same steps in a different order give different rows. Use the preset of the "
          "source family (chapter 5). The preset \"current\" is for new data that is already in the current format. "
          "With --gate, a last step also removes rows that fail a FLAG rule of the validation."),
]

REPAIRS = {
    "format": ("Plain OpenAI keys", "repairs/format.py",
               ["tool_calls get the shape {id, type: \"function\", function: {name, arguments}}; arguments become a JSON string.",
                "content null becomes \"\". Empty reasoning_content is removed.",
                "Message keys that are not allowed for the role are removed; top-level keys other than messages / tools are removed."]),
    "shapes": ("Shape repairs", "repairs/shapes.py",
               ["A flat tool definition {name, description, parameters} is wrapped as {type: \"function\", function: {...}}.",
                "A Kimi tool-declare system message (<|im_system|>tool_declare...) is removed; the tools are already in \"tools\".",
                "A visible answer \"null\" with the real reply inside <think> gets the <think> text as the answer.",
                "An <answer>...</answer> trapped inside <think> is moved after </think>.",
                "Drop: content that is not a string; a final assistant turn that shows nothing."]),
    "hermes": ("Hermes text tool calls", "repairs/tools.py",
               ["Tools are read from the system prompt (<tools> JSON list, one JSON per line, markdown tool list, or "
                "\"following tools\" JSON).",
                "Assistant \"text <tool_call>{json}</tool_call>\" becomes content = text + tool_calls; the next user message with "
                "<tool_response> becomes tool messages. The system message is removed.",
                "All or nothing per row: drop when a called tool is not in the tools list, when the number of results is not the "
                "number of calls, or when there is text after the last call."]),
    "tool_ids": ("Tool reply ids", "repairs/tools.py",
                 ["A tool reply without tool_call_id gets the id of the matching call, in order.",
                  "A tool reply with no structured call before it (the call was written as text) becomes a user message."]),
    "think_content": ("Reasoning inside the content", "repairs/think_content.py",
                      ["A think-only assistant message followed by another assistant message is merged with it.",
                       "Problem-creation reasoning (\"The user wants me to create ... problem\") on a solve request is removed.",
                       "When the only answer is inside <think> and contains a final answer, the tags are removed.",
                       "A last assistant turn that is only <think>...</think> is removed (row trimmed)."]),
    "move_think": ("<think> to reasoning_content", "repairs/reasoning.py",
                   ["\"<think>R</think>A\" (one block, at the start) -> reasoning_content = R, content = A.",
                    "\"<think><think>R</think>A\" -> one <think> removed, then as above. Empty <think> -> no reasoning_content.",
                    "Drop: <think> not closed, </think> without <think>, two or more blocks, a block not at the start, "
                    "<think> in a user message, an empty final answer after the move."]),
    "move_other_tags": ("Other reasoning tags", "repairs/reasoning.py",
                        ["<thinking>, <reasoning>, <|begin_of_thought|>, <scratchpad>, <thought>, <analysis> and similar "
                         "at the start of the answer -> reasoning_content.",
                         "Drop: the tag is in another place, or the message already has reasoning_content."]),
    "harness_fallback": ("Agent-harness fallback text", "repairs/roles.py",
                         ["The last assistant text is the harness fallback (\"Technical difficulties. Please continue with the task.\"): "
                          "the row is trimmed back to the last real assistant turn. Drop when no real assistant turn is left."]),
    "final": ("Small final repairs", "repairs/final.py",
              ["A final assistant turn with a real answer AND tool calls (nothing after it): the dangling tool calls are removed.",
               "Literal backslash-n characters at the END of user / assistant text are removed.",
               "reasoning_content identical to the answer is removed."]),
    "role_rules": ("Role rules", "repairs/roles.py",
                   ["Empty assistant turns (no text, no tool call) are removed.",
                    "The row is trimmed so that it ends on its last assistant turn. Drop when there is no assistant turn.",
                    "Drop when the system prompt is directly followed by an assistant turn.",
                    "Note: when an empty assistant turn is removed, two user turns can come together. The validation flags this (user_user)."]),
    "text_drops": ("Final-answer text drops (optional)", "repairs/text_drops.py",
                   ["Drop: the final answer is empty, starts like raw self-talk (\"Okay, let me\", \"Hmm\", \"I need to solve\"), "
                    "or is cut off (ends on a function word or a comma, 300+ characters)."]),
}

HF_DOMAIN = [
    ("p", "These tools were used for the large hf_domain corpus. They are in sft_pipeline/hf_domain/."),
    ("table", ["Tool", "What it does"], [
        ["lid_split.py", "Splits each shard by language with fastText lid.176 (all message text, first 2,000 characters). "
                         "One output file per shard per language. Env: SFTPIPE_SRC, SFTPIPE_OUT, SFTPIPE_LOG, LID_MODEL."],
        ["strict_filter.py", "Makes strict OpenAI rows. Repairs: content stored as a Python list of parts, a whole conversation "
                             "dumped into one turn, double think tags, <answer> wrappers, Hermes tool calls, trailing special tokens. "
                             "Rejects: parse errors, bad roles, empty content, images, broken think blocks, missing context. "
                             "Options: --pii redact, --think keep|strip, --max-chars. Self test: --selftest."],
        ["quality_filter.py", "Drop-only filter: format, think blocks, garbage text, loops. Kept rows are byte for byte the input. "
                              "Rejected rows go to a rejects folder with their reasons. Self test: --self-test."],
        ["context_postpass.py", "Flags rows whose question misses content the answer needs (<image>, <DNA> placeholders; cut-off "
                                "question confirmed by the answer). Dry run unless --apply."],
        ["python_tool_postpass.py", "Adds the python tool definition to rows that call the python tool. Dry run unless --apply."],
    ]),
]

VALIDATE = [
    ("p", "The validation checks every row with 68 rules (64 FLAG, 4 INFO), including the eka template render check. "
          "It writes a list of every flagged row and sample rows for each rule. It changes nothing."),
    ("code", "export EKA_SPM=/path/eka_v0_1_1.model\nexport EKA_TEMPLATE=/path/eka_v0_2_0/chat_template.jinja\n"
             "python3 -m sft_pipeline.validate CLEAN REPORT_DIR 32 100   # 32 processes, render 1 row in 100"),
    ("table", ["Output file", "Content"], [
        ["ISSUES_REPORT.txt", "Rows for each rule, top sources, stats (reasoning, tool calls, turns, length)"],
        ["FLAG_SUMMARY.txt", "Rows with at least one FLAG rule, rows for each rule"],
        ["ROWS_FLAGGED.txt", "Every flagged row: file, line (1-based), byte offset, rule ids"],
        ["ROWS_INFO_ONLY.txt", "Rows with INFO rules only"],
        ["samples/<rule>.txt", "Up to 10 full rows for each rule, from different sources"],
    ]),
    ("note", "Before you trust a count, read random flagged rows of each rule. A rule can also flag good rows."),
]

STATS = [
    ("code", "python3 -m sft_pipeline.stats STATS_DIR dataset_a=FOLDER_A dataset_b=FOLDER_B"),
    ("p", "One read of all rows. Big files are split into byte ranges. The workbook has these sheets: Overview, Types "
          "(single / multi turn, tool use, tools list, system prompt, reasoning), Turns, Tool calls, Messages, Length (tokens), "
          "Answer length, Per file, Notes. Tokens are real counts with the o200k_base tokenizer (tiktoken) over all content, "
          "reasoning_content, tool calls and the tools list. Template tokens are not included."),
]

LEGACY = [
    ("p", "The legacy folder holds code that is not part of the main flow."),
    ("table", ["Folder", "Content"], [
        ["legacy/one_off", "Fixes for one dataset each: missing assistant turn (reply in a separate column), fimbulvntr reasoning + "
                           "answer columns, countdown rollouts (only correct answers kept), struct nulls in tool arguments, "
                           "tool-call arguments that the template could not load."],
        ["legacy/old_format", "Steps for the old storage format, where reasoning was kept as <think> inside the content: "
                              "to_openai_format.py, the first combined cleaner, the GLM template render check. "
                              "Do not use them for new data: the current format keeps reasoning in reasoning_content."],
    ]),
]

EXTERNAL = [
    ("p", "Other teams own these steps. They are not in this repository. They are listed so that the full data flow is clear."),
    ("table", ["Step", "Input", "Output"], [
        ["Schema normalization (DEDUP_PIPELINE normalize_job)", "raw HF datasets", "canonical parquet (uses the same schema inference as the generic adapter)"],
        ["Exact and fuzzy deduplication (MinHash, Jaccard >= 0.8)", "final OpenAI datasets", "deduplicated datasets"],
        ["Quality classifier split (high / low)", "deduplicated data", "high and low folders"],
    ]),
]

NEW_SOURCE = [
    ("b", ["Read 20 to 50 real rows of the new source first. Find the question, the reasoning, the answer and the tool fields.",
           "Try the generic adapter. Read <log>/no_adapter.jsonl and the counts in <log>/convert_log.jsonl.",
           "If the schema is special, write a converter: copy the pattern of adapters/traces/extra*.py (one function for each "
           "dataset) or add an entry to the CONVERT table of adapters/agentic.py. Use openai_format.convert_message and "
           "finalize. Write the reason for each mapping in the docstring.",
           "Add a small invented before/after case to tests/cases.py and run python3 tests/test_pipeline.py.",
           "Run clean, then validate. Read the samples of each flagged rule before you report a count.",
           "If you add a repair step, add it to clean.py STEPS, add a test case, and add a matching rule to the validation."]),
]

LESSONS = [
    "Read real rows before you write a converter. Column names do not tell the full truth.",
    "Check the miss side, not only the flagged side. A rule can look precise and still miss many rows of the same pattern.",
    "Read flagged rows by hand before you trust a count.",
    "A repair can create new damage. After a repair, run the same checks again and compare with the original rows. "
    "Example: replace a literal \\n only outside quoted strings and JSON values.",
    "Validate every derived copy. A copy made before a repair step does not have that repair.",
    "Do not run two jobs or sessions on the same output folder.",
    "Keep logs out of data folders. A log file with a .jsonl name in a data folder becomes training data by mistake.",
]

NOT_CHECKED = [
    ("p", "The pipeline checks the format, the structure and the text of each row. It does not check that an answer is correct "
          "(facts, math results, code results). For that, use an LLM judge or run the code. It does not remove duplicates; "
          "the deduplication step of the other team does that."),
]


RECIPES_INTRO = [
    ("p", "A recipe is the full list of commands, in order, that made the final data of one source family. "
          "Each recipe was read from the production run scripts and logs. Then it was tested end to end: raw sample -> "
          "this repository -> compared row by row with the production data before deduplication (with the same last "
          "reasoning steps on both sides)."),
    ("note", "Deduplication and the quality split are done by other teams after these recipes."),
]

RECIPES = [
    ("HF agentic", [
        "python3 -m sft_pipeline.convert agentic --src RAW --out CONV",
        "python3 -m sft_pipeline.clean CONV CLEAN --preset agentic [--gate]",
     ], "cleanup -> fold -> tool_ids -> think_content -> shapes -> move_think -> move_other_tags -> text_drops"),
    ("HF traces", [
        "python3 -m sft_pipeline.convert traces --src RAW --out CONV",
        "python3 -m sft_pipeline.convert traces_extra1 --src RAW --out CONV     (... to traces_extra9)",
        "python3 -m sft_pipeline.convert traces_post --src RAW --out CONV",
        "python3 -m sft_pipeline.clean CONV CLEAN --preset traces [--gate]",
     ], "cleanup -> harness_fallback -> fold -> tool_ids -> think_content -> shapes -> move_think -> move_other_tags -> text_drops"),
    ("codeLLM posttraining (A, B1, B2)", [
        "python3 -m sft_pipeline.convert posttraining --src RAW --out CONV",
        "python3 -m sft_pipeline.convert posttraining_b2 --src RAW --out CONV",
        "python3 -m sft_pipeline.clean CONV CLEAN --preset posttraining [--gate]",
     ], "fold -> shapes -> tool_ids -> think_content -> role_rules -> move_think -> move_other_tags -> text_drops"),
    ("hf_domain", [
        "(external) schema normalization -> canonical data",
        "python3 sft_pipeline/hf_domain/lid_split.py              (env SFTPIPE_SRC / SFTPIPE_OUT / SFTPIPE_LOG / LID_MODEL)",
        "python3 sft_pipeline/hf_domain/strict_filter.py --in LANG_SPLIT --out STRICT --think keep --empty-think drop \\",
        "        --max-chars 200000 --special-tokens strip --pii datatrove --context-check on --code-tool python",
        "python3 sft_pipeline/hf_domain/python_tool_postpass.py --root STRICT --apply",
        "python3 -m sft_pipeline.clean STRICT CLEAN --preset hf_domain [--gate]",
     ], "fold -> think_content -> shapes -> move_think -> move_other_tags -> text_drops"),
]

E2E_TABLE = (["Family", "Sample", "Rows (repo = production)", "Result"], [
    ["HF agentic", "no_robots, glaive-function-calling-v2, tau2-sft-v4", "19,934 + 112,738 + 200", "100% equal"],
    ["HF traces", "gpt-4o-traces (missing reply), hermes-agent-traces (ShareGPT), tau2bench-retail (struct nulls), medical-reasoning-11k (extra1)",
     "2 + 3,646 + 1,092 + 11,106", "100% equal; 18 more rows in the repo output are the rows that the production audit gate removed (special tokens 16, empty answer 2); --gate removes them"],
    ["posttraining", "B1 KodCode RL, B1 DeepMind code_contests, B2 Nemotron text", "10,000 + 336 + 25,688", "100% equal"],
    ["hf_domain", "-", "-", "Not tested end to end (the first step is external). strict_filter and quality_filter pass their self tests."],
])
