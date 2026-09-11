"""
tests/test_agent.py  ·  Lab 1.3 test suite
───────────────────────────────────────────
Run with:   pytest tests/

Tests are organised into three groups matching the three functions
students implement:

    TestBuildSystemPrompt  – build_system_prompt()
    TestParseLlmResponse   – parse_llm_response()
    TestRunAgent           – run_agent()  (uses a mocked LLM)

The LLM is mocked at the HTTP layer so tests are fast, free, and
deterministic.  No real API calls are made unless you run with
    pytest tests/ --run-llm
which enables the end-to-end smoke test.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent import (
    MAX_ITERATIONS,
    AgentResult,
    build_system_prompt,
    parse_llm_response,
    run_agent,
)
from memory import SemanticMemory


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_completion(content: str) -> MagicMock:
    """Build a fake openai ChatCompletion with a single assistant message."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _tool_call_text(thought: str, action: str, args: dict) -> str:
    return (
        f"Thought: {thought}\n"
        f"Action: {action}\n"
        f"Action Input: {json.dumps(args)}"
    )


def _final_answer_text(thought: str, answer: str) -> str:
    return f"Thought: {thought}\nFINAL ANSWER: {answer}"


# ─────────────────────────────────────────────────────────────────────────────
# Group 1 · build_system_prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSystemPrompt:
    def test_returns_string(self):
        mem = SemanticMemory(collection_name="test_bsp_1")
        result = build_system_prompt(mem, "some task")
        assert isinstance(result, str)

    def test_empty_memory_returns_base_prompt(self):
        """When memory is empty the base SYSTEM_PROMPT should be returned unchanged."""
        from agent import SYSTEM_PROMPT
        mem = SemanticMemory(collection_name="test_bsp_2")
        result = build_system_prompt(mem, "write a sort function")
        assert result == SYSTEM_PROMPT

    def test_memories_appended_when_present(self):
        """Stored memories must appear in the returned prompt."""
        mem = SemanticMemory(collection_name="test_bsp_3")
        mem.store("User prefers type-annotated Python with docstrings.")
        mem.store("exec_python output is truncated at 2000 chars.")

        result = build_system_prompt(mem, "write a sort function")

        # The prompt must be longer than the base prompt.
        from agent import SYSTEM_PROMPT
        assert len(result) > len(SYSTEM_PROMPT)

        # At least one stored memory must appear verbatim in the prompt.
        assert (
            "type-annotated" in result or "truncated at 2000" in result
        ), "Expected at least one retrieved memory to appear in the system prompt."

    def test_memories_section_has_heading(self):
        """The injected memory block must have a heading so the LLM can identify it."""
        mem = SemanticMemory(collection_name="test_bsp_4")
        mem.store("Prior fact: the workspace dir is ./workspace")
        result = build_system_prompt(mem, "read a file")
        # Some kind of heading (case-insensitive check).
        assert "memor" in result.lower(), (
            "Expected a 'memories' heading in the system prompt when memories exist."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Group 2 · parse_llm_response
# ─────────────────────────────────────────────────────────────────────────────

class TestParseLlmResponse:
    # ── FORMAT A: tool call ───────────────────────────────────────────────────

    def test_parse_tool_call_keys(self):
        text = _tool_call_text(
            "I will write the file first.",
            "write_file",
            {"path": "out.py", "content": "print('hi')"},
        )
        result = parse_llm_response(text)
        assert "thought" in result
        assert "action" in result
        assert "action_input" in result

    def test_parse_tool_call_values(self):
        text = _tool_call_text(
            "Reading the file to check its contents.",
            "read_file",
            {"path": "main.py"},
        )
        result = parse_llm_response(text)
        assert result["action"] == "read_file"
        assert result["action_input"] == {"path": "main.py"}
        assert "Reading" in result["thought"]

    def test_parse_tool_call_multiline_json(self):
        """Action Input JSON may span multiple lines."""
        text = (
            "Thought: Writing the function now.\n"
            "Action: write_file\n"
            "Action Input: {\n"
            '  "path": "utils.py",\n'
            '  "content": "def foo():\\n    pass\\n"\n'
            "}"
        )
        result = parse_llm_response(text)
        assert result.get("action") == "write_file"
        assert isinstance(result.get("action_input"), dict)
        assert result["action_input"]["path"] == "utils.py"

    def test_parse_exec_python_tool(self):
        text = _tool_call_text(
            "Executing the script to verify it runs.",
            "exec_python",
            {"code": "print(1 + 1)"},
        )
        result = parse_llm_response(text)
        assert result["action"] == "exec_python"
        assert result["action_input"]["code"] == "print(1 + 1)"

    def test_parse_error_on_invalid_json(self):
        """A malformed Action Input should return a parse_error key, not raise."""
        text = (
            "Thought: Trying something.\n"
            "Action: write_file\n"
            "Action Input: {bad json here"
        )
        result = parse_llm_response(text)
        assert "parse_error" in result, (
            "Expected a 'parse_error' key when Action Input is not valid JSON."
        )
        assert "final_answer" not in result

    # ── FORMAT B: final answer ────────────────────────────────────────────────

    def test_parse_final_answer_keys(self):
        text = _final_answer_text("Done.", "The function is in workspace/utils.py")
        result = parse_llm_response(text)
        assert "final_answer" in result
        assert "thought" in result

    def test_parse_final_answer_no_action(self):
        text = _final_answer_text("All tests pass.", "workspace/solution.py")
        result = parse_llm_response(text)
        assert "action" not in result
        assert "action_input" not in result

    def test_parse_final_answer_content(self):
        answer_text = "The word_frequency function is complete and tested."
        text = _final_answer_text("Task complete.", answer_text)
        result = parse_llm_response(text)
        assert answer_text in result["final_answer"]

    def test_thought_is_non_empty_string(self):
        for text in [
            _tool_call_text("Some thought.", "read_file", {"path": "x.py"}),
            _final_answer_text("Another thought.", "done"),
        ]:
            result = parse_llm_response(text)
            assert isinstance(result.get("thought"), str)
            assert len(result["thought"].strip()) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Group 3 · run_agent  (mocked LLM)
# ─────────────────────────────────────────────────────────────────────────────

WRITE_CALL = _tool_call_text(
    "I will write the function to a file.",
    "write_file",
    {"path": "word_freq.py", "content": "def word_frequency(fp): return {}"},
)

EXEC_CALL = _tool_call_text(
    "Now I will execute the file to verify it imports.",
    "exec_python",
    {"code": "import importlib.util, sys; print('ok')"},
)

FINAL = _final_answer_text(
    "The function is written and executes without error.",
    "workspace/word_freq.py contains the word_frequency function.",
)


class TestRunAgent:
    # Mock sequence: write → exec → final answer
    _mock_responses = [
        _make_completion(WRITE_CALL),
        _make_completion(EXEC_CALL),
        _make_completion(FINAL),
    ]

    def _patched_run(self, mock_responses: list) -> AgentResult:
        """Run run_agent with the LLM patched to return *mock_responses* in order."""
        call_count = [0]

        def fake_create(**kwargs):
            idx = call_count[0]
            call_count[0] += 1
            if idx < len(mock_responses):
                return mock_responses[idx]
            # Fallback: always return FINAL so the loop terminates.
            return _make_completion(FINAL)

        with patch("agent.client") as mock_client:
            mock_client.chat.completions.create.side_effect = fake_create
            return run_agent("Write a word frequency function.")

    def test_returns_agent_result(self):
        result = self._patched_run(self._mock_responses)
        assert isinstance(result, AgentResult)

    def test_final_answer_populated(self):
        result = self._patched_run(self._mock_responses)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_stopped_early_false_on_clean_finish(self):
        result = self._patched_run(self._mock_responses)
        assert result.stopped_early is False

    def test_iteration_count(self):
        result = self._patched_run(self._mock_responses)
        # 3 LLM calls: write → exec → final
        assert result.iterations == 3

    def test_tool_calls_recorded(self):
        result = self._patched_run(self._mock_responses)
        # Two tool calls (write_file and exec_python) should be logged.
        assert len(result.tool_calls) == 2

    def test_tool_calls_have_required_keys(self):
        result = self._patched_run(self._mock_responses)
        for call in result.tool_calls:
            assert "tool" in call
            assert "args" in call
            assert "result" in call

    def test_tool_names_in_log(self):
        result = self._patched_run(self._mock_responses)
        names = [c["tool"] for c in result.tool_calls]
        assert "write_file" in names
        assert "exec_python" in names

    def test_max_iterations_guard(self):
        """Loop must stop and set stopped_early=True if no FINAL ANSWER arrives."""
        # Return only tool-call responses — never a final answer.
        infinite_responses = [
            _make_completion(
                _tool_call_text(f"Step {i}", "read_file", {"path": "x.py"})
            )
            for i in range(MAX_ITERATIONS + 5)
        ]
        result = self._patched_run(infinite_responses)
        assert result.stopped_early is True
        assert result.iterations <= MAX_ITERATIONS

    def test_parse_error_does_not_crash(self):
        """A malformed LLM response must not raise; the loop must continue."""
        bad_response = _make_completion(
            "Thought: Hmm.\nAction: write_file\nAction Input: {bad json"
        )
        responses = [bad_response, _make_completion(FINAL)]
        result = self._patched_run(responses)
        assert isinstance(result, AgentResult)

    def test_single_iteration_final_answer(self):
        """Agent should finish in one iteration if LLM returns FINAL ANSWER immediately."""
        result = self._patched_run([_make_completion(FINAL)])
        assert result.iterations == 1
        assert result.stopped_early is False


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end smoke test (only runs with --run-llm flag)
# ─────────────────────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption(
        "--run-llm",
        action="store_true",
        default=False,
        help="Run end-to-end tests that make real LLM API calls.",
    )


@pytest.fixture
def run_llm(request):
    return request.config.getoption("--run-llm")


@pytest.mark.llm
def test_e2e_word_frequency(run_llm):
    """
    End-to-end: agent must produce a working Python word_frequency function.
    Only runs when pytest is invoked with  --run-llm.
    """
    if not run_llm:
        pytest.skip("Skipping real-LLM test. Use --run-llm to enable.")

    from pathlib import Path
    from tools import WORKSPACE

    task = (
        "Write a Python function called word_frequency(filepath: str) -> dict "
        "that reads a text file and returns a word-count dict (case-insensitive, "
        "no punctuation).  Save it to workspace/word_frequency.py. "
        "Then execute a quick smoke test that calls the function on a string "
        "written to a temp file and prints the result."
    )
    result = run_agent(task)

    assert not result.stopped_early, "Agent hit MAX_ITERATIONS without finishing."

    output_file = WORKSPACE / "word_frequency.py"
    assert output_file.exists(), "Expected workspace/word_frequency.py to be created."

    source = output_file.read_text()
    assert "def word_frequency" in source, "Expected function definition in output file."
