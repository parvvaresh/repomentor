"""
Coder agent: writes or changes code that fits the indexed repository.

It retrieves the most relevant sources, shows them to the model with
numbered labels and asks for an implementation that follows the
repository's existing conventions. The model refers to sources as [n];
the frontend turns those into links to the exact files and lines.
"""

import os
import re

import rag_answer

from .base import Agent, AgentContext


# A larger model usually writes much better code than the default one
CODER_MODEL = os.getenv("OLLAMA_CODER_MODEL") or rag_answer.MODEL_NAME
CODER_CONTEXT_LENGTH = int(os.getenv("OLLAMA_CODER_CONTEXT", "16384"))
CODER_TIMEOUT = 600

# Earlier turns kept so follow-ups such as "now add tests" work
HISTORY_MESSAGES = 6
MAX_HISTORY_CHARACTERS = 6000

SOURCE_REFERENCE_PATTERN = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = """
You are a senior software engineer working inside the repository {project}.
You write code changes that fit this repository.

Rules:
1. Base your work on the numbered repository sources below. Reuse the
   functions, classes, naming and style they show instead of inventing
   new ones.
2. When you rely on a source, refer to it as [n], for example [2].
3. Start with a short plan (at most five bullets).
4. Then give the code. Put each file in its own fenced code block with a
   language tag, and write the file path on the line just before the
   block, e.g. **path/to/file.py**. For an existing file, show only the
   changed or new functions with enough surrounding context to place them.
5. End with any steps needed to try the change (commands, migrations).
6. If the sources do not show something you need, say so and state the
   assumption you made. Never claim code exists in the repository unless
   a source shows it.
"""


def format_sources(chunks: list[dict]) -> str:
    parts = []

    for number, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{number}] {chunk['file_path']} "
            f"(lines {chunk['start_line']}-{chunk['end_line']})\n"
            f"```\n{chunk['content']}\n```"
        )

    return "\n\n".join(parts)


def recent_history(history: list[dict]) -> list[dict]:
    """Keep the last few turns, newest first, within a character budget."""

    kept = []
    budget = MAX_HISTORY_CHARACTERS

    for message in reversed(history[-HISTORY_MESSAGES:]):
        content = message["content"]

        if len(content) > budget:
            break

        budget -= len(content)
        kept.append({"role": message["role"], "content": content})

    kept.reverse()

    return kept


def retrieval_query(question: str, history: list[dict]) -> str:
    """
    A follow-up like "add tests for it" retrieves poorly on its own,
    so the previous request is included in the search query.
    """

    for message in reversed(history):
        if message["role"] == "user":
            return f"{message['content']}\n{question}"

    return question


def cited_sources(answer: str, source_count: int) -> list[int]:
    numbers = {
        int(match) - 1
        for match in SOURCE_REFERENCE_PATTERN.findall(answer)
    }

    return sorted(number for number in numbers if 0 <= number < source_count)


class CoderAgent(Agent):
    id = "coder"
    name = "Coder"
    description = (
        "Writes new code or changes existing code, following the "
        "repository's own patterns and citing the files it builds on."
    )
    examples = [
        "Add an endpoint that returns the current user's profile.",
        "Write unit tests for the password hashing helpers.",
        "Add input validation to the item creation API.",
        "Refactor the database session handling into a helper.",
    ]
    working_label = "Reading sources and writing code…"

    def run(self, question: str, context: AgentContext) -> dict:
        query = retrieval_query(question, context.history)

        if context.retrieve_lock is None:
            indices, identifiers = rag_answer.retrieve(
                query,
                context.index,
                context.top_k,
            )
        else:
            with context.retrieve_lock:
                indices, identifiers = rag_answer.retrieve(
                    query,
                    context.index,
                    context.top_k,
                )

        chunks = [context.index["chunks"][i] for i in indices]
        sources = [rag_answer.describe_source(chunk) for chunk in chunks]

        if context.retrieval_only:
            return {
                "answer": "",
                "sources": sources,
                "cited_sources": [],
                "related_identifiers": identifiers,
                "retrieval_only": True,
                "model": None,
            }

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(project=context.project_path),
            },
            *recent_history(context.history),
            {
                "role": "user",
                "content": (
                    f"Repository sources:\n\n{format_sources(chunks)}\n\n"
                    f"Task:\n{question}"
                ),
            },
        ]

        answer = rag_answer.call_ollama(
            messages,
            temperature=0.2,
            context_length=CODER_CONTEXT_LENGTH,
            timeout=CODER_TIMEOUT,
            model=CODER_MODEL,
        )

        return {
            "answer": answer,
            "sources": sources,
            "cited_sources": cited_sources(answer, len(sources)),
            "related_identifiers": identifiers,
            "retrieval_only": False,
            "model": CODER_MODEL,
        }

    def format_text(self, result: dict) -> str:
        if result["retrieval_only"]:
            return "\n".join(
                source["citation"] for source in result["sources"]
            )

        return result["answer"]
