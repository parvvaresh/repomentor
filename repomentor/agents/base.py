"""
The contract every RepoMentor agent implements.

An agent receives a question plus the indexed repository and produces a
JSON-serializable result. It does so as a stream of events, so the web
interface can show progress and partial answers:

    {"type": "status", "label": "Reading app/crud.py…"}
    {"type": "sources", "sources": [...]}       sources it is working from
    {"type": "delta", "text": "..."}             a piece of Markdown answer
    {"type": "fact", "fact": {...}, "source": {...}}
    {"type": "result", "result": {...}}          always last

The server stores the final result with the chat message, and the
frontend picks a renderer from `result["agent"]`.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .. import retrieval
from ..citations import describe_source
from ..sources import RepositorySource


@dataclass
class AgentContext:
    """Everything an agent may use to answer one request."""

    # Loaded search index of the chat's repository (see repomentor.index)
    index: Any
    project_path: str
    top_k: int
    retrieval_only: bool = False
    # Serializes use of the shared embedding model across threads
    retrieve_lock: Any = None
    # Earlier messages of the chat, oldest first:
    # [{"role": "user" | "assistant", "content": str, "agent": str | None}]
    history: list[dict] = field(default_factory=list)
    # Live client for the repository host (history, merge requests), acting
    # as the signed-in user when there is one. None for offline use.
    source: RepositorySource | None = None
    # Project identifier on the host (for local checkouts, the directory)
    source_project: str | None = None


class Agent:
    """
    Base class for agents. Subclasses set the class attributes and
    implement `stream` (or `run` for agents that cannot report progress).
    Register an instance with `repomentor.agents.register`.
    """

    # Stable identifier, stored with every message; never rename it
    id: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str]
    # Example prompts shown on an empty chat
    examples: ClassVar[list[str]] = []
    # Text the frontend shows while the agent is working
    working_label: ClassVar[str] = "Working…"
    # Placeholder for the message box
    placeholder: ClassVar[str | None] = None

    def stream(self, question: str, context: AgentContext) -> Iterator[dict]:
        yield {"type": "result", "result": self.run(question, context)}

    def run(self, question: str, context: AgentContext) -> dict:
        """
        Return a JSON-serializable dict. It must not contain the key
        "error" (reserved for failures) and should include "sources"
        (see citations.describe_source) when it cites the repository.
        The server adds "agent" automatically.
        """

        if type(self).stream is Agent.stream:
            raise NotImplementedError

        for event in self.stream(question, context):
            if event["type"] == "result":
                return event["result"]

        raise RuntimeError(f"Agent '{self.id}' produced no result.")

    def format_text(self, result: dict) -> str:
        """Plain-text version of a result, used for copying and history."""

        return str(result.get("answer", ""))

    def describe(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "examples": list(self.examples),
            "working_label": self.working_label,
            "placeholder": self.placeholder,
        }


# ---------------------------------------------------------------------------
# Helpers shared by agents
# ---------------------------------------------------------------------------

def run_retrieval(query: str, context: AgentContext, top_k: int | None = None):
    top_k = top_k or context.top_k

    if context.retrieve_lock is None:
        return retrieval.retrieve(query, context.index, top_k)

    with context.retrieve_lock:
        return retrieval.retrieve(query, context.index, top_k)


def retrieve_sources(query: str, context: AgentContext, top_k: int | None = None):
    """(chunks, described sources, related identifiers) for a query."""

    result = run_retrieval(query, context, top_k)
    chunks = [context.index.chunks[position] for position in result.indices]
    return chunks, [describe_source(chunk) for chunk in chunks], result.related_identifiers


def recent_history(history: list[dict], messages: int = 6, budget: int = 6000) -> list[dict]:
    """Keep the last few turns, newest first, within a character budget."""

    kept = []

    for message in reversed(history[-messages:]):
        content = message["content"]

        if len(content) > budget:
            break

        budget -= len(content)
        kept.append({"role": message["role"], "content": content})

    kept.reverse()
    return kept


def retrieval_query(question: str, history: list[dict]) -> str:
    """
    A follow-up like "add tests for it" retrieves poorly on its own, so the
    previous request is included in the search query.
    """

    for message in reversed(history):
        if message["role"] == "user":
            return f"{message['content']}\n{question}"

    return question


def format_sources(chunks: list[dict], tag: str) -> str:
    """Numbered, delimited sources for prompts that cite them as [n]."""

    from ..safety import wrap

    return "\n\n".join(
        wrap(
            chunk["content"],
            tag,
            number=number,
            file=chunk["file_path"],
            lines=f"{chunk['start_line']}-{chunk['end_line']}",
        )
        for number, chunk in enumerate(chunks, start=1)
    )


def cited_numbers(answer: str, source_count: int) -> list[int]:
    import re

    numbers = {int(match) - 1 for match in re.findall(r"\[(\d+)\]", answer)}
    return sorted(number for number in numbers if 0 <= number < source_count)
