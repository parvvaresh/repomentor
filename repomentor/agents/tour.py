"""
Tour agent: onboarding for developers who are new to a repository.

On first use it builds hierarchical directory summaries (see
repomentor.summaries). It then either gives the tour (an overview, an
architecture map of directories and a learning path) or answers a broad
question from the directory summaries most similar to it plus the best
matching code, citing both.
"""

import numpy as np

from .. import llm, safety
from ..citations import describe_source
from ..retrieval import tokenize
from ..summaries import build_summaries
from .base import Agent, AgentContext, cited_numbers, format_sources, retrieve_sources


TOUR_WORDS = {
    "tour", "overview", "onboard", "onboarding", "architecture", "start",
    "begin", "learning", "path", "map", "structure", "introduce", "introduction",
}
SUMMARY_SOURCES = 3

ANSWER_PROMPT = """
You help a developer who is new to the repository {project}. Answer the
question using the numbered sources: directory summaries (written by an
assistant from the code) and code excerpts. Prefer code excerpts as
evidence, and cite every claim as [n]. Start with a short direct answer,
then the main pieces involved and where they live. If the sources do not
answer the question, say so.
{policy}"""


def wants_tour(question: str) -> bool:
    words = set(tokenize(question))
    return not question.strip() or (bool(words & TOUR_WORDS) and len(words) <= 12)


def summary_chunk(index, directory: str, summary: str) -> dict:
    web = index.manifest.get("web_url") or ""
    provider = index.provider
    tree = "-/tree" if provider == "gitlab" or "gitlab" in web else "tree"
    commit = index.manifest.get("head_commit") or "HEAD"
    path = "" if directory == "." else directory

    return {
        "chunk_id": f"summary:{directory}",
        "kind": "summary",
        "provider": provider,
        "project_path": index.project_path,
        "web_url": web,
        "file_path": f"{directory}/" if directory != "." else "(repository)",
        "title": f"Summary of {directory}/",
        "url": f"{web}/{tree}/{commit}/{path}".rstrip("/") if web else "",
        "last_commit_id": commit,
        "start_line": 1,
        "end_line": 1,
        "symbols": [],
        "content": summary,
    }


def tour_markdown(summaries: dict) -> str:
    lines = [summaries["overview"], "", "## Learning path"]

    for number, step in enumerate(summaries["learning_path"], start=1):
        lines.append(f"{number}. `{step['path']}` - {step['why']}")

    lines += ["", "## Architecture map"]

    for directory, node in summaries["directories"].items():
        depth = 0 if directory == "." else directory.count("/") + 1
        name = "(repository root)" if directory == "." else f"{directory}/"
        lines.append(f"{'  ' * depth}- **{name}** {node['summary']}")

    return "\n".join(lines)


class TourAgent(Agent):
    id = "tour"
    name = "Tour"
    description = (
        "Onboarding: an overview, an architecture map and a learning path, "
        "and answers to broad questions no single file answers."
    )
    examples = [
        "Give me a tour of this repository.",
        "How is the application deployed?",
        "What are the main parts of the system and how do they fit together?",
    ]
    working_label = "Reading the repository's structure…"

    def stream(self, question: str, context: AgentContext):
        index = context.index
        summaries = index.summaries()

        if summaries is None:
            yield {
                "type": "status",
                "label": "Summarizing the repository for the first time (this takes a while)…",
            }

            for event in build_summaries(index):
                if event["type"] == "summaries":
                    summaries = event["summaries"]
                else:
                    yield event

        if wants_tour(question):
            yield {"type": "result", "result": {
                "mode": "tour",
                "answer": tour_markdown(summaries),
                "overview": summaries["overview"],
                "learning_path": summaries["learning_path"],
                "directories": summaries["directories"],
                "sources": [],
                "cited_sources": [],
                "generated_at": summaries["generated_at"],
            }}
            return

        yield {"type": "status", "label": "Finding relevant parts of the repository…"}
        summary_chunks = self._similar_summaries(question, context, summaries)
        code_chunks, _, identifiers = retrieve_sources(question, context)
        chunks = summary_chunks + code_chunks
        sources = [describe_source(chunk) for chunk in chunks]
        yield {"type": "sources", "sources": sources}

        tag = safety.new_tag()
        parts = []

        for piece in llm.chat_stream([
            {
                "role": "system",
                "content": ANSWER_PROMPT.format(
                    project=context.project_path, policy=safety.policy(tag)
                ),
            },
            {
                "role": "user",
                "content": f"Sources:\n\n{format_sources(chunks, tag)}\n\nQuestion:\n{question}",
            },
        ]):
            parts.append(piece)
            yield {"type": "delta", "text": piece}

        answer = "".join(parts).strip()

        yield {"type": "result", "result": {
            "mode": "answer",
            "answer": answer,
            "sources": sources,
            "cited_sources": cited_numbers(answer, len(sources)),
            "related_identifiers": identifiers,
        }}

    def _similar_summaries(self, question, context: AgentContext, summaries: dict) -> list[dict]:
        directories = list(summaries["directories"].items())

        if not directories:
            return []

        texts = [f"{directory}/: {node['summary']}" for directory, node in directories]
        model = context.index.embedding_model

        def encode():
            return (
                model.encode(question, normalize_embeddings=True),
                model.encode(texts, normalize_embeddings=True),
            )

        if context.retrieve_lock is None:
            question_vector, vectors = encode()
        else:
            with context.retrieve_lock:
                question_vector, vectors = encode()

        scores = np.asarray(vectors) @ question_vector
        best = np.argsort(scores)[::-1][:SUMMARY_SOURCES]

        return [
            summary_chunk(context.index, directories[position][0], directories[position][1]["summary"])
            for position in best
        ]

    def format_text(self, result: dict) -> str:
        return result["answer"]

