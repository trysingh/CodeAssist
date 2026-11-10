"""Native (Gemma-4 style) tool calls passed through as text by a runner are treated as tool calls, not as the answer."""
from services.code_agent import parse_action, strip_tool_calls


def test_native_call_with_quoted_strings():
    r = '<|tool_call>call:read{path: "services/process_manager.py", start: 1, end: 185}<tool_call|>'
    assert parse_action(r) == ("read", {"path": "services/process_manager.py", "start": 1, "end": 185}, "")


def test_native_call_with_gemma_string_tokens():
    r = '<|tool_call>call:grep{pattern:<|"|>port<|"|>,path:<|"|>services/*.py<|"|>}<tool_call|>'
    assert parse_action(r) == ("grep", {"pattern": "port", "path": "services/*.py"}, "")


def test_native_call_after_text_and_without_closing_tag():
    assert parse_action('Let me check.\n<|tool_call>call:ls{path: "views"}') == ("ls", {"path": "views"}, "")


def test_unreadable_native_call_is_an_error_not_a_final_answer():
    name, args, err = parse_action("<|tool_call>call:read{garbage")
    assert name == "" and err


def test_action_lines_and_plain_answers_unchanged():
    assert parse_action('ACTION: {"tool": "read", "args": {"path": "a.py"}}') == ("read", {"path": "a.py"}, "")
    assert parse_action("The port is wrong at `views/admin.py:12`.") is None


def test_strip_tool_calls_keeps_only_the_answer():
    assert strip_tool_calls('Answer text\n<|tool_call>call:read{path: "x"}<tool_call|>') == "Answer text"
    assert strip_tool_calls('Answer\nACTION: {"tool": "ls"}') == "Answer"
