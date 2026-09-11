# Lab 1.3 – ReACT Agent

**Module 1 · Section 4 · Build Autonomous Multi-Agent Systems**
Saras AI Institute

---

## What you will build

A working ReACT (Reasoning + Acting) agent that solves multi-step coding
tasks end-to-end.  The agent ties together the memory system from Lab 1.1
and the tools from Lab 1.2 into a continuous loop:

```
Sense → Think → Act → Remember → repeat
```

At the end of this lab you will have a Coder Agent that can accept a
natural-language coding task, plan and execute it across multiple tool
calls, and return structured output — the first building block of the
Week 1 capstone deliverable.

---

## Repo layout

```
lab-1.3-react-agent/
├── agent.py          ← YOUR WORK.  Three functions to implement.
├── llm_client.py     ← Pre-provided. Do not modify.
├── memory.py         ← Pre-provided. Do not modify.
├── tools.py          ← Pre-provided. Do not modify.
├── requirements.txt
├── tests/
│   └── test_agent.py ← Run with: pytest tests/
└── workspace/        ← Created at runtime. The agent's sandbox.
```

---

## Your task

Open `agent.py`.  You will implement exactly **three functions**:

### 1. `build_system_prompt(semantic_memory, task) → str`

Retrieves relevant memories from the vector store and appends them to
the base system prompt so the agent can recall facts from previous runs.

### 2. `parse_llm_response(response_text) → dict`

Parses the LLM's raw text output into a structured dict.  The LLM
responds in one of two formats:

**Format A — tool call:**
```
Thought: I need to write the function first.
Action: write_file
Action Input: {"path": "utils.py", "content": "def foo(): ..."}
```

**Format B — final answer:**
```
Thought: The task is complete.
FINAL ANSWER: workspace/utils.py contains the completed function.
```

### 3. `run_agent(task) → AgentResult`

The main loop.  Calls the LLM, parses its response, dispatches tool
calls, records observations, and repeats until the agent signals it is
done or the iteration limit is reached.

Read the docstrings in `agent.py` carefully — they contain step-by-step
hints for each function.

---

## Running the agent

```bash
python agent.py
```

The default `DEMO_TASK` at the bottom of `agent.py` asks the agent to
write and test a `word_frequency` function.  Replace it with any coding
task you like.

---

## Running the tests

```bash
# Fast (no LLM calls — mocked):
pytest tests/

# End-to-end (real LLM calls — slow and costs tokens):
pytest tests/ --run-llm
```

All 18 mocked tests should pass before you run the full end-to-end test.

---

## Success criteria

Your implementation passes when:

1. `pytest tests/` reports **all mocked tests passing** with no errors.
2. `python agent.py` produces a `FINAL ANSWER` for the default task
   within 10 iterations.
3. `workspace/word_frequency.py` exists after the run and contains a
   `def word_frequency` function definition.

---

## The ReACT format in detail

The agent communicates with the LLM using plain text, not native
function-calling.  Each LLM response must contain:

| Field          | Format A (tool call)    | Format B (final answer) |
|----------------|-------------------------|-------------------------|
| `Thought:`     | Required                | Required                |
| `Action:`      | Tool name               | —                       |
| `Action Input:`| JSON object             | —                       |
| `FINAL ANSWER:`| —                       | The deliverable         |

After each tool call, the agent appends the tool's result as:

```
OBSERVATION: <tool result>
```

This observation becomes part of the next LLM input, closing the loop.

---

## Common mistakes to avoid

| Mistake | Fix |
|---|---|
| `Action Input` is not valid JSON | `parse_llm_response` should return a `parse_error` key, not raise |
| Loop never terminates | Always check for `FINAL ANSWER` first; enforce `MAX_ITERATIONS` |
| Same action repeated | The loop guard catches this, but also check: is the observation being added to the buffer? |
| Memory block added even when empty | Check `semantic_memory.count() == 0` before calling `.retrieve()` |
