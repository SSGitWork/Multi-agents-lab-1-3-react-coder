"""
tools.py
────────
Pre-provided. Do NOT modify this file.

Implements the two tools the ReACT agent uses:
  • read_file  – reads a file from the sandbox workspace
  • write_file – writes content to a file in the sandbox workspace
  • exec_python – executes a Python snippet in a subprocess sandbox

These are the same tools completed in Lab 1.2.  They are provided here
so that Lab 1.3 can focus entirely on the ReACT loop.
"""

import subprocess
import tempfile
import textwrap
from pathlib import Path

# All file I/O is confined to this directory so the agent cannot escape the workspace.
WORKSPACE = Path("workspace")
WORKSPACE.mkdir(exist_ok=True)


# ── Tool schemas (OpenAI function-calling format) ────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file from the sandbox workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the file to read, relative to the workspace root.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the sandbox workspace, creating it if it does not exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the file to write, relative to the workspace root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text content to write to the file.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exec_python",
            "description": (
                "Execute a Python code snippet in a sandboxed subprocess and return its stdout. "
                "The code runs in an isolated process with a 10-second timeout. "
                "Use this to verify that generated code actually works."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python source code to execute.",
                    }
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
]


# ── Tool handlers ─────────────────────────────────────────────────────────────

def read_file(path: str) -> str:
    """Return file contents or a descriptive error string."""
    target = WORKSPACE / path
    try:
        return target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"ERROR: File not found: {path}"
    except Exception as exc:
        return f"ERROR: Could not read {path}: {exc}"


def write_file(path: str, content: str) -> str:
    """Write content to a file; return a confirmation or error string."""
    target = WORKSPACE / path
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return f"File written successfully: {path} ({line_count} lines)"
    except Exception as exc:
        return f"ERROR: Could not write {path}: {exc}"


def exec_python(code: str) -> str:
    """
    Execute *code* in a subprocess with a 10-second timeout.
    Returns stdout on success, or a combined stderr / timeout message on failure.
    The code runs in a temporary file so multi-line scripts work correctly.
    """
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
        tmp.write(textwrap.dedent(code))
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout
        if result.returncode != 0:
            output += f"\nSTDERR:\n{result.stderr}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Execution timed out after 10 seconds."
    except Exception as exc:
        return f"ERROR: Execution failed: {exc}"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Dispatch helper ───────────────────────────────────────────────────────────

TOOL_HANDLERS: dict = {
    "read_file": lambda args: read_file(**args),
    "write_file": lambda args: write_file(**args),
    "exec_python": lambda args: exec_python(**args),
}


def dispatch_tool(name: str, args: dict) -> str:
    """
    Call the tool identified by *name* with *args*.
    Returns the tool's result as a string, or an error message if the
    tool name is unknown.
    """
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"ERROR: Unknown tool '{name}'. Available tools: {list(TOOL_HANDLERS)}"
    return handler(args)
