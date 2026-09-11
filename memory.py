"""
memory.py
─────────
Pre-provided. Do NOT modify this file.

Provides the two-tier memory system completed in Lab 1.1:
  • SlidingWindowBuffer  – short-term message history (in-context)
  • SemanticMemory       – long-term storage and retrieval via Chroma

The ReACT agent (agent.py) calls these directly; you do not need to
understand the internals for Lab 1.3.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any

import chromadb
from chromadb import Collection
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

_HELICONE_BASE = os.getenv("HELICONE_BASE_URL")
_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
_HELICONE_API_KEY = os.getenv("HELICONE_API_KEY")

_embed_client = OpenAI(
    api_key=_OPENROUTER_API_KEY,
    base_url=_HELICONE_BASE,
    default_headers={"Helicone-Auth": f"Bearer {_HELICONE_API_KEY}"},
)


def _embed(text: str) -> list[float]:
    response = _embed_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


class HeliconeEmbeddingFunction(chromadb.EmbeddingFunction):
    # type: ignore[override]
    def __call__(self, input: list[str]) -> list[list[float]]:
        return [_embed(text) for text in input]


# ── Short-term memory ─────────────────────────────────────────────────────────

class SlidingWindowBuffer:
    """
    Keeps the most recent *max_messages* messages as OpenAI message dicts.
    The system prompt (role=system) is always kept regardless of window size.
    """

    def __init__(self, max_messages: int = 20) -> None:
        self._max = max_messages
        self._system: dict[str, str] | None = None
        self._window: deque[dict[str, Any]] = deque()

    def set_system(self, content: str) -> None:
        self._system = {"role": "system", "content": content}

    def add(self, role: str, content: str) -> None:
        self._window.append({"role": role, "content": content})
        while len(self._window) > self._max:
            self._window.popleft()

    def messages(self) -> list[dict[str, Any]]:
        base = [self._system] if self._system else []
        return base + list(self._window)

    def clear(self) -> None:
        self._window.clear()


# ── Long-term semantic memory ─────────────────────────────────────────────────

class SemanticMemory:
    """
    Persists text snippets in a Chroma vector store and retrieves the
    most relevant ones by embedding similarity.
    """

    def __init__(self, collection_name: str = "agent_memory") -> None:
        self._chroma = chromadb.Client()
        self._collection: Collection = self._chroma.get_or_create_collection(
            name=collection_name,
            embedding_function=HeliconeEmbeddingFunction(),
        )
        self._counter = 0

    def store(self, text: str, metadata: dict | None = None) -> None:
        self._counter += 1
        self._collection.add(
            documents=[text],
            ids=[f"mem_{self._counter}"],
            metadatas=[{"source": "agent"}]
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count() or 1),
        )
        docs = results.get("documents", [[]])[0]
        return docs if docs else []

    def count(self) -> int:
        return self._collection.count()
