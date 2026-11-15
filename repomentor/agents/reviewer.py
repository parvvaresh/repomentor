"""
Reviewer agent: reviews a merge request against the repository's own
conventions.

The change comes from the host (a merge request number or URL) or from a
diff pasted into the chat. For every changed file, the code most similar
to the change is retrieved from the index, so the model can compare the
change with how the repository already does the same things: naming,
error handling, validation, tests, structure.
"""

import re

from .. import config, llm, safety
from ..citations import describe_source
from .base import Agent, AgentContext, cited_numbers, format_sources, run_retrieval


MAX_FILES = 8
MAX_DIFF_CHARACTERS = 14000
SOURCES_PER_FILE = 2

REFERENCE_PATTERNS = [
    re.compile(r"/(?:-/)?merge_requests/(\d+)"),
    re.compile(r"/pull/(\d+)"),
    re.compile(r"(?:^|\s)[!#](\d+)\b"),
    re.compile(r"\b(?:mr|pr|merge request|pull request)\s*#?(\d+)\b", re.IGNORECASE),
]

DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.MULTILINE)
PLUS_HEADER = re.compile(r"^\+\+\+ (?:b/)?(\S+)", re.MULTILINE)

SYSTEM_PROMPT = """
You are a careful senior reviewer for the repository {project}. Review the
change against the conventions the repository already follows, shown in the
numbered sources.

Write Markdown with these sections:
## Summary
One or two sentences on what the change does.
## Findings
Bullets, most important first. Start each with a severity in bold
(**bug**, **security**, **convention**, **test**, **nit**), name the file,
and explain the problem and the fix. When a finding relies on how the
repository does something elsewhere, cite the source as [n].
## Looks good
Short bullets on what follows the repository's conventions.

Rules:
- Only report problems you can see in the diff. Do not invent code.
- Prefer a few precise findings over many vague ones.
- If the change looks correct and consistent, say so.
{policy}"""


def find_reference(text: str) -> int | None:
    for pattern in REFERENCE_PATTERNS:
        match = pattern.search(text)

        if match:
            return int(match.group(1))

    return None


def parse_pasted_diff(text: str) -> list[dict]:
    """Split a pasted unified diff into per-file changes."""

    text = re.sub(r"^```\w*\n|```\s*$", "", text.strip(), flags=re.MULTILINE)
    starts = [match.start() for match in DIFF_HEADER.finditer(text)]

    if not starts:
        starts = [match.start() for match in re.finditer(r"^--- ", text, re.MULTILINE)]

    changes = []

    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(text)
        section = text[start:end]
        header = DIFF_HEADER.match(section) or PLUS_HEADER.search(section)

        if header:
            path = header.group(header.lastindex)
            changes.append({"path": path, "old_path": path, "diff": section})

    return changes


def added_lines(diff: str, limit: int = 40) -> str:
    lines = [
        line[1:] for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    return "\n".join(lines[:limit])


class ReviewerAgent(Agent):
    id = "reviewer"
    name = "Reviewer"
    description = (
        "Reviews a merge request (or a pasted diff) against the conventions "
        "this repository already follows."
    )
    examples = [
        "Review merge request !1",
        "Review https://github.com/owner/repo/pull/1",
    ]
    working_label = "Reading the change and comparing it with the repository…"
    placeholder = "Paste a diff, or give a merge request number such as !12"

    def stream(self, question: str, context: AgentContext):
        changes = parse_pasted_diff(question)
        request = None

        if not changes:
            number = find_reference(question)

            if number is None:
                raise ValueError(
                    "Give a merge request number (for example !12 or #12), "
                    "a merge request URL, or paste a unified diff."
                )

            if context.source is None:
                raise ValueError("This repository's host is not available for reviews.")

            yield {"type": "status", "label": f"Fetching merge request {number}…"}
            request = context.source.get_merge_request(
                context.source_project or context.project_path, number
            )
            changes = [change for change in request["changes"] if not change["deleted_file"]]

        if not changes:
            raise ValueError("The change has no files to review.")

        reviewed = changes[:MAX_FILES]

        yield {"type": "status", "label": "Finding related code in the repository…"}
        chunks = []
        seen = set()

        for change in reviewed:
            query = f"{change['path']}\n{added_lines(change['diff'])}"
            result = run_retrieval(query, context, SOURCES_PER_FILE + 1)

            for position in result.indices[:SOURCES_PER_FILE]:
                chunk = context.index.chunks[position]

                if chunk["chunk_id"] not in seen:
                    seen.add(chunk["chunk_id"])
                    chunks.append(chunk)

        chunks = chunks[: max(context.top_k, 6)]
        sources = [describe_source(chunk) for chunk in chunks]
        yield {"type": "sources", "sources": sources}

        diff_text = "\n".join(change["diff"] for change in reviewed)[:MAX_DIFF_CHARACTERS]
        tag = safety.new_tag()
        title = f"{request['title']}\n\n{request['description']}" if request else ""

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    project=context.project_path, policy=safety.policy(tag)
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Repository sources:\n\n{format_sources(chunks, tag)}\n\n"
                    + safety.wrap(title, tag, kind="merge request description")
                    + "\n\n"
                    + safety.wrap(diff_text, tag, kind="diff")
                    + "\n\nReview this change."
                ),
            },
        ]

        yield {"type": "status", "label": "Writing the review…"}
        parts = []

        for piece in llm.chat_stream(
            messages,
            temperature=0.1,
            context_length=config.CODER_CONTEXT_LENGTH,
            model=config.CODER_MODEL,
        ):
            parts.append(piece)
            yield {"type": "delta", "text": piece}

        answer = "".join(parts).strip()

        yield {"type": "result", "result": {
            "answer": answer,
            "sources": sources,
            "cited_sources": cited_numbers(answer, len(sources)),
            "merge_request": {
                "number": request["number"],
                "title": request["title"],
                "url": request["url"],
                "author": request["author"],
            } if request else None,
            "files": [change["path"] for change in reviewed],
            "skipped_files": [change["path"] for change in changes[MAX_FILES:]],
            "model": config.CODER_MODEL,
        }}
