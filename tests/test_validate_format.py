"""Every kind of OpenAI-format break must be FLAGGED by the validation (and must not crash it).
Run: python3 tests/test_validate_format.py"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sft_pipeline", "validate"))
import dataset_check as D  # noqa: E402

U = {"role": "user", "content": "Hi"}
A = {"role": "assistant", "content": "Hello"}
FN = {"name": "f", "arguments": "{}"}
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]


def call(cid="call_0", **kw):
    return {"id": cid, "type": "function", "function": dict(FN), **kw}


def tc(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def tool(cid="call_0", content="1", **kw):
    return {"role": "tool", "tool_call_id": cid, "content": content, **kw}


GOOD = {
    "plain": {"messages": [U, A]},
    "system_user_assistant": {"messages": [{"role": "system", "content": "s"}, U, A]},
    "tool_round_trip": {"messages": [U, tc(call()), tool(), A], "tools": TOOLS},
    "parallel_calls": {"messages": [U, tc(call("a"), call("b")), tool("a"), tool("b"), A], "tools": TOOLS},
    "ends_on_tool_call": {"messages": [U, tc(call())], "tools": TOOLS},
    "reasoning_content": {"messages": [U, dict(A, reasoning_content="because")]},
}
BAD = {
    "no_messages_key": {"conversations": [U, A]},
    "messages_not_list": {"messages": "hi"},
    "message_not_object": {"messages": ["hi", A]},
    "role_missing": {"messages": [{"content": "Hi"}, A]},
    "role_unknown": {"messages": [{"role": "human", "content": "Hi"}, A]},
    "content_missing": {"messages": [{"role": "user"}, A]},
    "content_parts_list": {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}, A]},
    "content_number": {"messages": [{"role": "user", "content": 5}, A]},
    "assistant_null_no_calls": {"messages": [U, {"role": "assistant", "content": None}]},
    "tool_calls_not_list": {"messages": [U, {"role": "assistant", "content": "", "tool_calls": {"id": "x"}}, A], "tools": TOOLS},
    "tool_call_function_not_object": {"messages": [U, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": "f()"}]}, A], "tools": TOOLS},
    "tool_call_type_missing": {"messages": [U, tc({"id": "call_0", "function": dict(FN)}), tool(), A], "tools": TOOLS},
    "tool_call_type_wrong": {"messages": [U, tc(call(type="code")), tool(), A], "tools": TOOLS},
    "args_object_not_string": {"messages": [U, tc({"id": "call_0", "type": "function", "function": {"name": "f", "arguments": {}}}), tool(), A], "tools": TOOLS},
    "args_invalid_json": {"messages": [U, tc({"id": "call_0", "type": "function", "function": {"name": "f", "arguments": "{x"}}), tool(), A], "tools": TOOLS},
    "args_json_not_object": {"messages": [U, tc({"id": "call_0", "type": "function", "function": {"name": "f", "arguments": "[1]"}}), tool(), A], "tools": TOOLS},
    "tool_call_without_id": {"messages": [U, tc({"type": "function", "function": dict(FN)}), tool(), A], "tools": TOOLS},
    "tool_call_without_name": {"messages": [U, tc({"id": "call_0", "type": "function", "function": {"arguments": "{}"}}), tool(), A], "tools": TOOLS},
    "tool_calls_on_user": {"messages": [dict(U, tool_calls=[call()]), A], "tools": TOOLS},
    "tool_calls_but_no_tools": {"messages": [U, tc(call()), tool(), A]},
    "called_tool_not_in_tools": {"messages": [U, tc({"id": "call_0", "type": "function", "function": {"name": "g", "arguments": "{}"}}), tool(), A], "tools": TOOLS},
    "tool_content_not_string": {"messages": [U, tc(call()), tool(content={"a": 1}), A], "tools": TOOLS},
    "tool_reply_without_id": {"messages": [U, tc(call()), {"role": "tool", "content": "1"}, A], "tools": TOOLS},
    "tool_reply_wrong_id": {"messages": [U, tc(call()), tool("zzz"), A], "tools": TOOLS},
    "tool_reply_without_call": {"messages": [U, tool(), A]},
    "call_without_reply_then_user": {"messages": [U, tc(call()), U, A], "tools": TOOLS},
    "two_calls_one_reply": {"messages": [U, tc(call("a"), call("b")), tool("a"), A], "tools": TOOLS},
    "duplicate_call_ids": {"messages": [U, tc(call("a"), call("a")), tool("a"), tool("a"), A], "tools": TOOLS},
    "name_on_tool_message": {"messages": [U, tc(call()), tool(name="f"), A], "tools": TOOLS},
    "tools_not_list": {"messages": [U, A], "tools": {"type": "function"}},
    "tool_def_flat": {"messages": [U, A], "tools": [{"name": "f", "parameters": {}}]},
    "tool_def_no_parameters": {"messages": [U, A], "tools": [{"type": "function", "function": {"name": "f"}}]},
    "reasoning_on_user": {"messages": [dict(U, reasoning_content="r"), A]},
    "reasoning_not_string": {"messages": [U, dict(A, reasoning_content=["r"])]},
    "unknown_message_key": {"messages": [U, dict(A, weight=1)]},
    "extra_top_level_key": {"messages": [U, A], "id": "7"},
    "system_not_first": {"messages": [U, {"role": "system", "content": "s"}, A]},
    "first_turn_assistant": {"messages": [A, U, A]},
    "system_then_assistant": {"messages": [{"role": "system", "content": "s"}, A]},
    "user_user": {"messages": [U, U, A]},
    "assistant_assistant": {"messages": [U, A, A]},
    "ends_on_user": {"messages": [U, A, U]},
    "empty_messages": {"messages": []},
    "think_in_content": {"messages": [U, {"role": "assistant", "content": "<think>r</think>a"}]},
}


def flags(row):
    line = json.dumps(row).encode()
    return D.row_issues(row, line, "x.jsonl", False) - D.INFO_ONLY


def test_good_rows_pass():
    for k, r in GOOD.items():
        assert not flags(r), (k, flags(r))


def test_bad_rows_flagged():
    for k, r in BAD.items():
        assert flags(r), k


if __name__ == "__main__":
    bad = 0
    for k, r in GOOD.items():
        f = flags(r); ok = not f; bad += not ok
        print(("ok  " if ok else "FAIL"), "good:", k, sorted(f))
    for k, r in BAD.items():
        f = flags(r); ok = bool(f); bad += not ok
        print(("ok  " if ok else "FAIL"), "bad: ", k, sorted(f)[:3])
    print(f"{len(GOOD)} good rows, {len(BAD)} bad rows, failures: {bad}")
    sys.exit(1 if bad else 0)
