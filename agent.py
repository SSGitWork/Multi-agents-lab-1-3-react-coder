"""
agent.py  ·  Lab 1.3 – ReACT Agent
────────────────────────────────────────────────────────────────────────────────
YOUR TASK
─────────
Implement the three functions marked with  TODO  below.  They wire together
the memory system (memory.py) and the tool layer (tools.py) into a working
ReACT loop.

The ReACT pattern structures LLM reasoning as a sequence of:
    Thought  → what the agent is planning to do next
    Action   → the tool to call (name + arguments)
    Observation → the tool's return value

The loop repeats until:
    • The LLM returns a  FINAL ANSWER:  marker (task complete), OR
    • The iteration counter reaches MAX_ITERATIONS (loop guard).

Files you will edit
───────────────────
    agent.py         ← THIS FILE.  Implement the three TODO functions.

Files provided (do not modify)
──────────────────────────────
    llm_client.py    – Helicone-routed OpenAI client + MODEL constant
    tools.py         – read_file, write_file, exec_python + dispatch_tool()
    memory.py        – SlidingWindowBuffer + SemanticMemory

Running the agent
─────────────────
    python agent.py

    The agent will attempt the default DEMO_TASK defined at the bottom of
    this file.  Swap it out with your own task to experiment.

Running the tests
─────────────────
    pytest tests/
"""

from __future__ import annotations

import json
from typing import Any

from llm_client import client, MODEL
from memory import SemanticMemory, SlidingWindowBuffer
from tools import TOOL_SCHEMAS, dispatch_tool

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ITERATIONS = 10   # Hard ceiling on agentic loop iterations.

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
    """Structured result returned by run_agent()."""

    def __init__(
        self,
        answer: str,
        iterations: int,
        tool_calls: list[dict[str, Any]],
        stopped_early: bool,
    ) -> None:
        self.answer = answer          # The FINAL ANSWER text.
        self.iterations = iterations  # How many loop iterations ran.
        self.tool_calls = tool_calls  # List of {"tool": ..., "args": ..., "result": ...}.
        self.stopped_early = stopped_early  # True if MAX_ITERATIONS was reached.

    def __repr__(self) -> str:
        status = "MAX_ITER" if self.stopped_early else "DONE"
        return (
            f"AgentResult(status={status}, iterations={self.iterations}, "
            f"tool_calls={len(self.tool_calls)})\n"
            f"ANSWER:\n{self.answer}"
        )


# ── Functions for students to implement ───────────────────────────────────────

def build_system_prompt(semantic_memory: SemanticMemory, task: str) -> str:
    """
    Construct the system prompt by injecting relevant long-term memories.

    Steps
    -----
    1. Call  semantic_memory.retrieve(task, top_k=3)  to get up to three
       relevant memory strings.
    2. If any memories were returned, format them as a bulleted list and
       append them to SYSTEM_PROMPT under a  ## Relevant memories  heading.
    3. Return the final system prompt string.

    If the memory store is empty (semantic_memory.count() == 0), return
    SYSTEM_PROMPT unchanged.

    Parameters
    ----------
    semantic_memory : SemanticMemory
        The long-term vector store.  May be empty on the first run.
    task : str
        The user's task description.  Used as the retrieval query.

    Returns
    -------
    str
        The complete system prompt, with memories appended if available.
    """
    raise NotImplementedError("TODO: implement build_system_prompt()")


def parse_llm_response(response_text: str) -> dict[str, Any]:
    """
    Parse a single LLM response into a structured dict.

    The response will be in one of two formats (see SYSTEM_PROMPT):

    FORMAT A (tool call):
        Thought: <text>
        Action: <tool_name>
        Action Input: <json>

    FORMAT B (final answer):
        Thought: <text>
        FINAL ANSWER: <text>

    Steps
    -----
    1. Extract the  Thought:  line.
    2. Check whether  "FINAL ANSWER:"  appears in the response.
       • If yes → return {"thought": ..., "final_answer": <everything after "FINAL ANSWER:">}
    3. Otherwise, extract  Action:  and  Action Input: .
       • Parse Action Input as JSON with json.loads().
       • If json.loads() raises, return an error dict so the loop can
         feed an error observation back to the LLM.
    4. Return {"thought": ..., "action": ..., "action_input": <dict>}

    Error return format (malformed Action Input):
        {"thought": ..., "action": ..., "parse_error": <error message string>}

    Hints
    -----
    • Use  str.split("Thought:", 1)  /  str.split("Action:", 1)  etc. to
      extract each field.
    • Strip whitespace from every extracted value.
    • The Action Input JSON may contain newlines — extract everything after
      "Action Input:" up to the end of the string (or next recognised keyword).

    Parameters
    ----------
    response_text : str
        Raw text from the LLM's message content.

    Returns
    -------
    dict
        A structured dict with the keys described above.
    """
    raise NotImplementedError("TODO: implement parse_llm_response()")


def run_agent(task: str) -> AgentResult:
    """
    Run the full ReACT agentic loop for the given *task*.

    Algorithm
    ---------
    1. Initialise a SlidingWindowBuffer and a SemanticMemory.
    2. Call build_system_prompt() and pass the result to buffer.set_system().
    3. Add the user task as the first message:  buffer.add("user", task)
    4. Enter the loop (up to MAX_ITERATIONS):
        a. Call the LLM with  client.chat.completions.create()  passing
           buffer.messages() and TOOL_SCHEMAS as the tools list.
           Use  tool_choice="none"  so the model uses the ReACT text format
           rather than native function-calling.
        b. Extract the assistant's text: response.choices[0].message.content
        c. Add the assistant message to the buffer:
               buffer.add("assistant", <assistant_text>)
        d. Parse the response with  parse_llm_response().
        e. If the parsed result contains "final_answer":
               • Store the final answer in semantic memory:
                     semantic_memory.store("Task: " + task + "\\nAnswer: " + final_answer)
               • Return an AgentResult with stopped_early=False.
        f. If the parsed result contains "parse_error":
               • Build an observation string:
                     "OBSERVATION: Parse error – " + parse_error
               • Add it to the buffer as a user message and continue.
        g. Otherwise (normal tool call):
               • Call  dispatch_tool(action, action_input)  to get the result.
               • Record the call in the tool_calls list.
               • Build the observation:
                     "OBSERVATION: " + tool_result
               • Store noteworthy observations in semantic memory
                 (store if the observation does NOT start with "ERROR").
               • Add the observation to the buffer as a user message.
    5. If the loop ends without a FINAL ANSWER, return AgentResult with
       stopped_early=True and answer="MAX_ITERATIONS reached".

    Parameters
    ----------
    task : str
        Natural language description of the coding task to complete.

    Returns
    -------
    AgentResult
        Structured result including the final answer, iteration count,
        and a log of all tool calls made.
    """
    raise NotImplementedError("TODO: implement run_agent()")


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
