"""
llm_client.py
─────────────
Pre-provided. Do NOT modify this file.

Provides a single OpenAI client instance that routes every request
through the Helicone proxy for logging and cost tracking.

Usage
-----
    from llm_client import client, MODEL

    response = client.chat.completions.create(
        model=MODEL,
        messages=[...],
    )
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

_HELICONE_BASE = os.getenv("HELICONE_BASE_URL")
_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
_HELICONE_API_KEY = os.getenv("HELICONE_API_KEY")

# Default model for all agent calls in Module 1.
MODEL = "gpt-4.1-mini"

client = OpenAI(
    # Helicone key doubles as the auth token.
    api_key=_OPENROUTER_API_KEY,
    base_url=_HELICONE_BASE,
    default_headers={
        "Helicone-Auth": f"Bearer {_HELICONE_API_KEY}",
    },
)
