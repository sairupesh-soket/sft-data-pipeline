# legacy

Code that is NOT part of the main flow (`convert -> clean -> validate`). Kept for reuse and for reference.
Paths were removed: set `SFTPIPE_PATH` (or edit the script) before you run one.

## one_off/  - fixes for one dataset (or a few) each

| Script | Problem it fixes |
|---|---|
| `fix_countdown_and_tools.py` | countdown RL rollouts: raw ChatML prompt in the user text; rebuilds system/user/assistant and keeps only rows whose answer evaluates to the target. (The tool part is in `sft_pipeline/repairs/tools.py`.) |
| `fix_tool_call_arguments.py` | tool-call arguments stored as strings in files converted before the traces converter was fixed at the source (the current converter does not need it). |

The traces fixes for the missing assistant turn, fimbulvntr and struct nulls are now in the main flow:
`python3 -m sft_pipeline.convert traces_post` (`sft_pipeline/adapters/traces/post.py`).

## old_format/  - steps for the OLD storage format (reasoning as `<think>` inside content)

The current format keeps reasoning in `reasoning_content`. These scripts fold reasoning back into the content
or check the GLM template, so do not use them for new data. They are here so old outputs can be explained.

| Script | What it did |
|---|---|
| `to_openai_format.py` | reasoning_content -> `<think>...</think>` in content; arguments to JSON string; non-OpenAI keys removed |
| `sft_clean_v1.py` | the first combined cleaner (to_openai_format + shapes + tools + think fixes + role rules + audit gate) |
| `validate_glm_template.py` | render check with the GLM-5.2 template (the current check uses the eka template) |
