"""
Coder agent: writes or changes code that fits the indexed repository.

It retrieves the most relevant sources, shows them to the model with
numbered labels and asks for an implementation that follows the
repository's existing conventions. Changes to existing files come back as
SEARCH/REPLACE blocks, which are applied to the indexed files to produce
unified diffs and syntax-checked results. The model refers to sources as
[n]; the frontend turns those into links to the exact files and lines.
"""

from .. import config, llm, safety
from ..coding import build_changes
from .base import (
    Agent,
    AgentContext,
    cited_numbers,
    format_sources,
    recent_history,
    retrieval_query,
    retrieve_sources,
)


CODER_TIMEOUT = 600

SYSTEM_PROMPT = """
You are a senior software engineer working inside the repository {project}.
You write code changes that fit this repository.

Rules:
1. Base your work on the numbered repository sources. Reuse the functions,
   classes, naming and style they show instead of inventing new ones.
2. When you rely on a source, refer to it as [n], for example [2].
3. Start with a short plan (at most five bullets).
4. Then give the changes, one file at a time. Write the file path in bold
   on its own line, then one fenced code block:
   - For an EXISTING file, the block contains one or more edits:
     <<<<<<< SEARCH
     lines copied exactly from the current file
     =======
     the new lines
     >>>>>>> REPLACE
     Keep each SEARCH short but unique, and copy it exactly.
   - For a NEW file, write **path/to/file.py** (new file) and the complete
     content of the file in the block.
5. End with any steps needed to try the change (commands, migrations).
6. If the sources do not show something you need, say so and state the
   assumption you made. Never claim code exists in the repository unless a
   source shows it.
{policy}"""


class CoderAgent(Agent):
    id = "coder"
    name = "Coder"
    description = (
        "Writes new code or changes existing code, following the "
        "repository's own patterns. Produces reviewable diffs and can open "
        "a merge request."
    )
    examples = [
        "Add an endpoint that returns the current user's profile.",
        "Write unit tests for the password hashing helpers.",
        "Add input validation to the item creation API.",
        "Refactor the database session handling into a helper.",
    ]
    working_label = "Reading sources and writing code…"

    def stream(self, question: str, context: AgentContext):
        yield {"type": "status", "label": "Retrieving sources…"}
        chunks, sources, identifiers = retrieve_sources(
            retrieval_query(question, context.history), context
        )
        yield {"type": "sources", "sources": sources}

        if context.retrieval_only:
            yield {"type": "result", "result": {
                "answer": "",
                "sources": sources,
                "cited_sources": [],
                "related_identifiers": identifiers,
                "retrieval_only": True,
                "model": None,
                "changes": [],
            }}
            return

        tag = safety.new_tag()
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    project=context.project_path, policy=safety.policy(tag)
                ),
            },
            *recent_history(context.history),
            {
                "role": "user",
                "content": (
                    f"Repository sources:\n\n{format_sources(chunks, tag)}\n\n"
                    f"Task:\n{question}"
                ),
            },
        ]

        yield {"type": "status", "label": "Writing code…"}
        parts = []

        for piece in llm.chat_stream(
            messages,
            temperature=0.2,
            context_length=config.CODER_CONTEXT_LENGTH,
            timeout=CODER_TIMEOUT,
            model=config.CODER_MODEL,
        ):
            parts.append(piece)
            yield {"type": "delta", "text": piece}

        answer = "".join(parts).strip()

        def read_file(path: str) -> str | None:
            record = context.index.file(path)
            return record["content"] if record else None

        yield {"type": "status", "label": "Applying and checking changes…"}

        yield {"type": "result", "result": {
            "answer": answer,
            "sources": sources,
            "cited_sources": cited_numbers(answer, len(sources)),
            "related_identifiers": identifiers,
            "retrieval_only": False,
            "model": config.CODER_MODEL,
            "changes": build_changes(answer, read_file),
            "base_commit": context.index.manifest.get("head_commit"),
        }}

    def format_text(self, result: dict) -> str:
        if result["retrieval_only"]:
            return "\n".join(source["citation"] for source in result["sources"])

        return result["answer"]
