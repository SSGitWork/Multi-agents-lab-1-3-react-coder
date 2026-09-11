"""
solution/agent_solution.py  ·  Lab 1.3 – ReACT Agent  (INSTRUCTOR SOLUTION)
──────────────────────────────────────────────────────────────────────────────
This file lives in the private instructor repo and is NEVER linked from
student-facing materials.

It is a drop-in replacement for agent.py.  To test it:
    cp solution/agent_solution.py agent.py
    pytest tests/
"""

from __future__ import annotations

import json
from typing import Any

from llm_client import client, MODEL
from memory import SemanticMemory, SlidingWindowBuffer
from tools import TOOL_SCHEMAS, dispatch_tool

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ITERATIONS = 10

SYSTEM_PROMPT = """You are a Coder Agent.  Your job is to complete coding tasks
by reasoning step-by-step and using the available tools.

## Response format

Every response MUST use EXACTLY one of the two formats below.

FORMAT A – when you need to call a tool:
    Thought: <your reasoning about what to do next>
    Action: <tool_name>
    Action Input: <valid JSON object with the tool's arguments>

FORMAT B – when the task is complete:
    Thought: <brief explanation of what you produced>
    FINAL ANSWER: <the deliverable: code, explanation, result, or a file path>

## Rules

- Always emit a Thought before every Action or FINAL ANSWER.
- Action Input must be a single-line valid JSON object.
- Never invent tool results; always wait for the Observation.
- If a tool returns an ERROR, try to fix the issue rather than repeating
  the same call.
- Stop and emit FINAL ANSWER as soon as the task is complete.
- You have a maximum of {max_iter} iterations.  Use them wisely.
""".format(max_iter=MAX_ITERATIONS)


# ── Output schema ─────────────────────────────────────────────────────────────

class AgentResult:
    def __init__(
        self,
        answer: str,
        iterations: int,
        tool_calls: list[dict[str, Any]],
        stopped_early: bool,
    ) -> None:
        self.answer = answer
        self.iterations = iterations
        self.tool_calls = tool_calls
        self.stopped_early = stopped_early

    def __repr__(self) -> str:
        status = "MAX_ITER" if self.stopped_early else "DONE"
        return (
            f"AgentResult(status={status}, iterations={self.iterations}, "
            f"tool_calls={len(self.tool_calls)})\n"
            f"ANSWER:\n{self.answer}"
        )


# ── Implementation ─────────────────────────────────────────────────────────────

def build_system_prompt(semantic_memory: SemanticMemory, task: str) -> str:
    """
    Inject relevant long-term memories into the system prompt.

    Design notes
    ────────────
    • We only add the memory block when the store is non-empty.  An empty
      "## Relevant memories" section with no bullet points would confuse
      the LLM, so we skip it entirely.
    • top_k=3 balances relevance against context-window pressure.
    • The memory block is appended at the END of SYSTEM_PROMPT so that the
      core format instructions (which the LLM must follow) always come first.
    """
    if semantic_memory.count() == 0:
        return SYSTEM_PROMPT

    memories = semantic_memory.retrieve(task, top_k=3)
    if not memories:
        return SYSTEM_PROMPT

    memory_block = "\n## Relevant memories\n"
    for mem in memories:
        memory_block += f"- {mem}\n"

    return SYSTEM_PROMPT + memory_block


def parse_llm_response(response_text: str) -> dict[str, Any]:
    """
    Parse a raw LLM response into a structured dict.

    Design notes
    ────────────
    • We split on recognised keywords rather than using a regex so that
      multi-line values (e.g. code blocks in Action Input) are preserved.
    • If Action Input is not valid JSON we return a parse_error rather than
      raising.  The loop will feed the error back as an OBSERVATION so the
      LLM can self-correct — a key resilience property of the ReACT pattern.
    • We intentionally do NOT raise on any parse failure; the loop guard
      (MAX_ITERATIONS) provides the backstop.
    """
    result: dict[str, Any] = {}

    # ── Extract Thought ───────────────────────────────────────────────────────
    thought = ""
    if "Thought:" in response_text:
        after_thought = response_text.split("Thought:", 1)[1]
        # Thought ends at the next recognised keyword.
        for keyword in ("Action:", "FINAL ANSWER:"):
            if keyword in after_thought:
                after_thought = after_thought.split(keyword, 1)[0]
        thought = after_thought.strip()
    result["thought"] = thought

    # ── FORMAT B: FINAL ANSWER ────────────────────────────────────────────────
    if "FINAL ANSWER:" in response_text:
        final = response_text.split("FINAL ANSWER:", 1)[1].strip()
        result["final_answer"] = final
        return result

    # ── FORMAT A: Action + Action Input ──────────────────────────────────────
    action = ""
    if "Action:" in response_text:
        after_action = response_text.split("Action:", 1)[1]
        if "Action Input:" in after_action:
            after_action = after_action.split("Action Input:", 1)[0]
        action = after_action.strip().splitlines()[0].strip()
    result["action"] = action

    action_input_raw = ""
    if "Action Input:" in response_text:
        action_input_raw = response_text.split("Action Input:", 1)[1].strip()

    try:
        result["action_input"] = json.loads(action_input_raw)
    except json.JSONDecodeError as exc:
        # Return an error dict so the loop can send an OBSERVATION back.
        result["parse_error"] = (
            f"Could not parse Action Input as JSON: {exc}. "
            f"Raw value was: {action_input_raw!r}"
        )

    return result


def run_agent(task: str) -> AgentResult:
    """
    Execute the ReACT loop.

    Design notes
    ────────────
    Memory writes
      • We store the final (task, answer) pair so future runs can recall
        what the agent previously produced for similar tasks.
      • Successful tool observations are also stored — they act as episodic
        memories that help the agent avoid repeating work.

    tool_choice="none"
      • We set tool_choice="none" to force the LLM to use the ReACT text
        format rather than native function-calling JSON.  This keeps all
        reasoning visible in the message content and makes it easy to log,
        debug, and feed back as observations.

    Loop guard
      • The iteration counter provides a hard ceiling.  When it is reached
        we return a result with stopped_early=True rather than raising — the
        caller can decide how to handle it.
    """
    buffer = SlidingWindowBuffer(max_messages=20)
    semantic_memory = SemanticMemory()

    system_prompt = build_system_prompt(semantic_memory, task)
    buffer.set_system(system_prompt)
    buffer.add("user", task)

    tool_calls_log: list[dict[str, Any]] = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        # ── LLM call ──────────────────────────────────────────────────────────
        response = client.chat.completions.create(
            model=MODEL,
            messages=buffer.messages(),
            tools=TOOL_SCHEMAS,
            tool_choice="none",   # Force ReACT text format.
        )
        assistant_text = response.choices[0].message.content or ""
        buffer.add("assistant", assistant_text)

        # ── Parse ─────────────────────────────────────────────────────────────
        parsed = parse_llm_response(assistant_text)

        # ── FINAL ANSWER ──────────────────────────────────────────────────────
        if "final_answer" in parsed:
            final_answer = parsed["final_answer"]
            semantic_memory.store(f"Task: {task}\nAnswer: {final_answer}")
            return AgentResult(
                answer=final_answer,
                iterations=iteration,
                tool_calls=tool_calls_log,
                stopped_early=False,
            )

        # ── Parse error → feed back as observation ────────────────────────────
        if "parse_error" in parsed:
            observation = f"OBSERVATION: Parse error – {parsed['parse_error']}"
            buffer.add("user", observation)
            continue

        # ── Tool call ─────────────────────────────────────────────────────────
        action = parsed.get("action", "")
        action_input = parsed.get("action_input", {})

        tool_result = dispatch_tool(action, action_input)

        tool_calls_log.append({
            "tool": action,
            "args": action_input,
            "result": tool_result,
        })

        # Store successful observations in long-term memory.
        if not tool_result.startswith("ERROR"):
            semantic_memory.store(f"Tool {action} result: {tool_result[:200]}")

        observation = f"OBSERVATION: {tool_result}"
        buffer.add("user", observation)

    # ── MAX_ITERATIONS reached ────────────────────────────────────────────────
    return AgentResult(
        answer="MAX_ITERATIONS reached without a FINAL ANSWER.",
        iterations=MAX_ITERATIONS,
        tool_calls=tool_calls_log,
        stopped_early=True,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

DEMO_TASK = (
    "Write a Python function called word_frequency(filepath: str) -> dict "
    "that reads a text file and returns a dictionary mapping each word to "
    "the number of times it appears.  The function should be case-insensitive "
    "and ignore punctuation.  Write the function to workspace/word_frequency.py, "
    "then write a small test script to workspace/test_word_frequency.py that "
    "creates a sample file, calls the function, and prints the result.  "
    "Finally, run the test script and confirm it produces output."
)

if __name__ == "__main__":
    result = run_agent(DEMO_TASK)
    print(result)
