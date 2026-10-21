"""Small synthetic before/after cases for every repair step and adapter helper.
Used by tests/test_pipeline.py (the expected results) and by docs/build_pdf.py (the examples in the PDF).
All rows are invented; no real data."""

T = "tool"

REPAIR_CASES = [
    # (step name, title, input row, expected: row dict or "drop:<reason>")
    ("format", "tool arguments as a JSON string, null content, unknown keys",
     {"messages": [
         {"role": "user", "content": "Weather in Paris?", "weight": 1},
         {"role": "assistant", "content": None,
          "tool_calls": [{"id": "call_0", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
         {"role": "tool", "tool_call_id": "call_0", "name": "get_weather", "content": "21 C"},
         {"role": "assistant", "content": "It is 21 C in Paris.", "reasoning_content": ""}],
      "id": "row-17"},
     {"messages": [
         {"role": "user", "content": "Weather in Paris?"},
         {"role": "assistant", "content": "",
          "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
         {"role": "tool", "tool_call_id": "call_0", "content": "21 C"},
         {"role": "assistant", "content": "It is 21 C in Paris."}]}),

    ("shapes", "flat tool definition is wrapped",
     {"messages": [{"role": "user", "content": "Time?"}, {"role": "assistant", "content": "It is noon."}],
      "tools": [{"name": "get_time", "description": "Current time", "parameters": {"type": "object", "properties": {}}}]},
     {"messages": [{"role": "user", "content": "Time?"}, {"role": "assistant", "content": "It is noon."}],
      "tools": [{"type": "function", "function": {"name": "get_time", "description": "Current time",
                                                  "parameters": {"type": "object", "properties": {}}}}]}),

    ("shapes", "\"null\" answer while the real reply is inside <think>",
     {"messages": [{"role": "user", "content": "List the files."},
                   {"role": "assistant", "content": "<think>{\"commands\": [\"ls -la\"]}</think>null"}]},
     {"messages": [{"role": "user", "content": "List the files."},
                   {"role": "assistant", "content": "{\"commands\": [\"ls -la\"]}"}]}),

    ("hermes", "Hermes text tool call -> OpenAI tool_calls",
     {"messages": [
         {"role": "system", "content": "You can call tools.\n<tools>\n[{\"name\": \"get_time\", \"description\": \"Time in a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}]\n</tools>"},
         {"role": "user", "content": "Time in Tokyo?"},
         {"role": "assistant", "content": "Let me check.\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"city\": \"Tokyo\"}}\n</tool_call>"},
         {"role": "user", "content": "<tool_response>\n{\"time\": \"09:00\"}\n</tool_response>"},
         {"role": "assistant", "content": "It is 09:00 in Tokyo."}]},
     {"messages": [
         {"role": "user", "content": "Time in Tokyo?"},
         {"role": "assistant", "content": "Let me check.",
          "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "get_time", "arguments": "{\"city\": \"Tokyo\"}"}}]},
         {"role": "tool", "tool_call_id": "call_0", "content": "{\"time\": \"09:00\"}"},
         {"role": "assistant", "content": "It is 09:00 in Tokyo."}],
      "tools": [{"type": "function", "function": {"name": "get_time", "description": "Time in a city",
                                                  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}),

    ("tool_ids", "tool reply without tool_call_id gets the id of its call",
     {"messages": [{"role": "user", "content": "Time?"},
                   {"role": "assistant", "content": "", "tool_calls": [{"id": "call_7", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}]},
                   {"role": "tool", "content": "12:00"},
                   {"role": "assistant", "content": "It is 12:00."}]},
     {"messages": [{"role": "user", "content": "Time?"},
                   {"role": "assistant", "content": "", "tool_calls": [{"id": "call_7", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}]},
                   {"role": "tool", "content": "12:00", "tool_call_id": "call_7"},
                   {"role": "assistant", "content": "It is 12:00."}]}),

    ("tool_ids", "tool reply after a call written as text becomes a user message",
     {"messages": [{"role": "user", "content": "Run ls."},
                   {"role": "assistant", "content": "Action: shell(\"ls\")"},
                   {"role": "tool", "tool_call_id": "x", "content": "a.txt b.txt"},
                   {"role": "assistant", "content": "There are two files."}]},
     {"messages": [{"role": "user", "content": "Run ls."},
                   {"role": "assistant", "content": "Action: shell(\"ls\")"},
                   {"role": "user", "content": "a.txt b.txt"},
                   {"role": "assistant", "content": "There are two files."}]}),

    ("think_content", "split turn: think-only message + answer message are merged",
     {"messages": [{"role": "user", "content": "2+2?"},
                   {"role": "assistant", "content": "<think>2 plus 2 is 4.</think>"},
                   {"role": "assistant", "content": "4"}]},
     {"messages": [{"role": "user", "content": "2+2?"},
                   {"role": "assistant", "content": "<think>2 plus 2 is 4.</think>\n\n4"}]}),

    ("move_think", "<think>R</think>A -> reasoning_content R, content A",
     {"messages": [{"role": "user", "content": "15% of 80?"},
                   {"role": "assistant", "content": "<think>0.15 * 80 = 12.</think>15% of 80 is 12."}]},
     {"messages": [{"role": "user", "content": "15% of 80?"},
                   {"role": "assistant", "content": "15% of 80 is 12.", "reasoning_content": "0.15 * 80 = 12."}]}),

    ("move_think", "<think> not closed -> row dropped",
     {"messages": [{"role": "user", "content": "15% of 80?"},
                   {"role": "assistant", "content": "<think>0.15 * 80 = ..."}]},
     "drop:think_not_closed"),

    ("move_other_tags", "<thinking>R</thinking>A -> reasoning_content",
     {"messages": [{"role": "user", "content": "Capital of Peru?"},
                   {"role": "assistant", "content": "<thinking>Peru is in South America; its capital is Lima.</thinking>Lima."}]},
     {"messages": [{"role": "user", "content": "Capital of Peru?"},
                   {"role": "assistant", "content": "Lima.", "reasoning_content": "Peru is in South America; its capital is Lima."}]}),

    ("harness_fallback", "agent-harness fallback text at the end -> trimmed",
     {"messages": [{"role": "user", "content": "Fix the bug."},
                   {"role": "assistant", "content": "I changed line 4."},
                   {"role": "user", "content": "Tests fail."},
                   {"role": "assistant", "content": "Technical difficulties. Please continue with the task."}]},
     {"messages": [{"role": "user", "content": "Fix the bug."},
                   {"role": "assistant", "content": "I changed line 4."}]}),

    ("final", "dangling tool call after a full answer is removed",
     {"messages": [{"role": "user", "content": "When is the meeting?"},
                   {"role": "assistant", "content": "The meeting is at 3 pm.",
                    "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "get_calendar", "arguments": "{}"}}]}]},
     {"messages": [{"role": "user", "content": "When is the meeting?"},
                   {"role": "assistant", "content": "The meeting is at 3 pm."}]}),

    ("final", "trailing literal \\n removed; duplicate reasoning removed",
     {"messages": [{"role": "user", "content": "Angle?\\n"},
                   {"role": "assistant", "content": "70 degrees", "reasoning_content": "70 degrees"}]},
     {"messages": [{"role": "user", "content": "Angle?"},
                   {"role": "assistant", "content": "70 degrees"}]}),

    ("role_rules", "row trimmed to end on the assistant; empty assistant turn removed",
     {"messages": [{"role": "user", "content": "Hi"},
                   {"role": "assistant", "content": ""},
                   {"role": "user", "content": "Hello?"},
                   {"role": "assistant", "content": "Hi! How can I help?"},
                   {"role": "user", "content": "Thanks"}]},
     {"messages": [{"role": "user", "content": "Hi"},
                   {"role": "user", "content": "Hello?"},
                   {"role": "assistant", "content": "Hi! How can I help?"}]}),

    ("role_rules", "assistant right after the system prompt -> row dropped",
     {"messages": [{"role": "system", "content": "Be brief."},
                   {"role": "assistant", "content": "Hello."}]},
     "drop:role_rules"),

    ("text_drops", "final answer starts like raw self-talk -> row dropped",
     {"messages": [{"role": "user", "content": "Solve x+1=3."},
                   {"role": "assistant", "content": "Okay, let me think. x is 2."}]},
     "drop:untagged_selftalk"),
]

ADAPTER_CASES = [
    # (title, function, input, expected)
    ("ShareGPT from/value -> roles", "convert_message",
     {"from": "gpt", "value": "Hello!"}, {"role": "assistant", "content": "Hello!"}),
    ("legacy function_call -> tool_calls", "convert_message",
     {"role": "assistant", "content": None, "function_call": {"name": "get_time", "arguments": "{\"city\": \"Rome\"}"}},
     {"role": "assistant", "content": "",
      "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "get_time", "arguments": "{\"city\": \"Rome\"}"}}]}),
    ("reasoning column kept as reasoning_content", "convert_message",
     {"role": "assistant", "content": "42", "reasoning": "6 x 7 = 42"},
     {"role": "assistant", "content": "42", "reasoning_content": "6 x 7 = 42"}),
    ("ChatML text -> messages", "parse_chatml",
     "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>",
     [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]),
    ("Alpaca columns instruction / input / output -> messages", "generic",
     {"instruction": "Translate to French.", "input": "Good morning", "output": "Bonjour"},
     [{"role": "user", "content": "Translate to French.\n\nGood morning"}, {"role": "assistant", "content": "Bonjour"}]),
    ("question / answer columns -> messages", "generic",
     {"question": "Capital of Japan?", "answer": "Tokyo"},
     [{"role": "user", "content": "Capital of Japan?"}, {"role": "assistant", "content": "Tokyo"}]),
    ("conversations column (ShareGPT) -> messages", "generic",
     {"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]},
     [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]),
]
