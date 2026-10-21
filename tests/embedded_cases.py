"""Hidden-content formats: reasoning, tool calls, tool results, roles and agent protocols written inside the text.
Used by tests/test_embedded.py: every case must be flagged."""
# content that hides reasoning / tool calls / tool results / roles inside text
R = {  # reasoning in the answer
 "think": "<think>2+2=4</think>4",
 "think_close_only": "2+2=4</think>4",
 "thinking": "<thinking>2+2=4</thinking>4",
 "reasoning": "<reasoning>2+2=4</reasoning>4",
 "begin_of_thought": "<|begin_of_thought|>2+2=4<|end_of_thought|>4",
 "scratchpad": "<scratchpad>2+2=4</scratchpad>4",
 "THOUGHT": "<THOUGHT>2+2=4</THOUGHT>4",
 "startthought": "<|startthought|>2+2=4<|endthought|>4",
 "pipe_thinking": "<|thinking|>2+2=4</|thinking|>4",
 "reasoning_start": "<reasoning_start>2+2=4<reasoning_end>4",
 "start_of_reasoning": "<|start_of_reasoning|>2+2=4<|end_of_reasoning|>4",
 "analysis_tag": "<analysis>2+2=4</analysis>4",
 "thought_tag": "<thought>2+2=4</thought>4",
 "kimi_think": "◁think▷2+2=4◁/think▷4",
 "seed_think": "<seed:think>2+2=4</seed:think>4",
 "harmony_channel": "<|channel|>analysis<|message|>2+2=4<|end|><|start|>assistant<|channel|>final<|message|>4",
 "react_thought": "Thought: I need to add.\nAction: calculator\nAction Input: 2+2\nObservation: 4\nThought: done\nFinal Answer: 4",
}
TC = {  # tool call written as text in assistant content
 "hermes_tool_call": "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"city\": \"Rome\"}}\n</tool_call>",
 "function_call_tag": "<function_call>{\"name\": \"get_time\", \"arguments\": {\"city\": \"Rome\"}}</function_call>",
 "functioncall_glaive": "<functioncall> {\"name\": \"get_time\", \"arguments\": '{\"city\": \"Rome\"}'}",
 "function_calls_xml": "<function_calls>\n<invoke name=\"get_time\">\n<parameter name=\"city\">Rome</parameter>\n</invoke>\n</function_calls>",
 "use_mcp_tool": "<use_mcp_tool>\n<server_name>time</server_name>\n<tool_name>get_time</tool_name>\n<arguments>{\"city\": \"Rome\"}</arguments>\n</use_mcp_tool>",
 "mistral_TOOL_CALLS": "[TOOL_CALLS] [{\"name\": \"get_time\", \"arguments\": {\"city\": \"Rome\"}}]",
 "llama_python_tag": "<|python_tag|>get_time.call(city=\"Rome\")",
 "deepseek_tool_calls": "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n```json\n{\"city\": \"Rome\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
 "kimi_tool_call": "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_time:0<|tool_call_argument_begin|>{\"city\": \"Rome\"}<|tool_call_end|><|tool_calls_section_end|>",
 "json_call_bare": "{\"name\": \"get_time\", \"arguments\": {\"city\": \"Rome\"}}",
 "json_call_list": "[{\"name\": \"get_time\", \"arguments\": {\"city\": \"Rome\"}}]",
 "json_call_in_fence": "```json\n{\"name\": \"get_time\", \"arguments\": {\"city\": \"Rome\"}}\n```",
 "tool_use_tag": "<tool_use>\n{\"name\": \"get_time\", \"input\": {\"city\": \"Rome\"}}\n</tool_use>",
 "action_input": "Action: get_time\nAction Input: {\"city\": \"Rome\"}",
}
TR = {  # tool result written as text (usually in a user message)
 "tool_response_tag": "<tool_response>\n{\"time\": \"10:00\"}\n</tool_response>",
 "function_response_text": "FUNCTION RESPONSE: {\"time\": \"10:00\"}",
 "observation": "Observation: {\"time\": \"10:00\"}",
 "ipython_role_marker": "<|start_header_id|>ipython<|end_header_id|>\n{\"time\": \"10:00\"}<|eot_id|>",
 "tool_result_tag": "<tool_result>{\"time\": \"10:00\"}</tool_result>",
}
RM = {  # chat-role markers / template tokens inside text
 "chatml": "Sure.<|im_end|>\n<|im_start|>user\nAnd Spain?",
 "llama3": "Sure.<|eot_id|><|start_header_id|>user<|end_header_id|>",
 "inst": "[INST] And Spain? [/INST]",
 "gemma": "<start_of_turn>user\nAnd Spain?<end_of_turn>",
 "human_assistant": "Sure.\n\nHuman: And Spain?\n\nAssistant: Madrid.",
 "endoftext": "Sure.<|endoftext|>",
 "eka_assistant": "Sure.<|assistant|>",
}
AG = {  # agent text protocols (assistant side)
 "swe_thought_bash": "THOUGHT: I need to search.\n\n```bash\ngrep -rl engine .\n```",
 "codeact_execute": "<execute> print(2+2) </execute>",
 "terminus_json": '{\n  "analysis": "fresh container",\n  "commands": [{"keystrokes": "ls -la\\n", "duration": 0.1}]\n}',
}
AGU = {  # agent text protocols (user side = result)
 "returncode": "<returncode>0</returncode>\n<output>\n./a.py\n</output>",
 "new_terminal_output": "New Terminal Output:\nroot@x:/app# ls\na.py",
 "current_terminal_state": "Current terminal state:\nroot@x:/app#",
}
