"""
Cited, fact-by-fact answers (the mentor agent's pipeline).

1. Retrieve the best chunks (one per file).
2. Refuse early when nothing in the repository is close to the question.
3. Ask the model for at most two facts per source, one source at a time,
   with numbered lines so each fact can point at the lines it came from.
4. Drop facts that name identifiers the source does not contain, and
   facts that repeat an earlier one.
5. Attach citations in Python, narrowed to the lines the model named when
   those lines are inside the chunk.

The error analysis of the first evaluation drove steps 2, 4 and 5: facts
were attributed to the wrong file, invented identifiers appeared in
otherwise plausible facts, and the same fact was often repeated.
"""

import re
from collections.abc import Iterator

from . import config, llm, safety
from .citations import create_citation, describe_source, line_url
from .retrieval import retrieve, tokenize


MAX_FACTS_PER_SOURCE = 2
MAX_TOTAL_FACTS = 6
DUPLICATE_THRESHOLD = 0.75

NO_EVIDENCE = "I couldn't find enough evidence in the retrieved repository sources."

# Citations a model may append on its own, e.g. "[app/crud.py:1-60]",
# "[1]" or "[Source 2]", without touching code such as "dict[str, int]"
MODEL_CITATION_PATTERN = re.compile(
    r"(?:\s*\[(?:[^\]\s]+:\d+(?:-\d+)?|\d+|source[^\]]*)\])+\s*$",
    re.IGNORECASE,
)
# Line references small models write in front of a fact: "(L5-7)", "L5-L7:",
# "[L5]", "<77>", "(L106)-<L112>"
LINE_PREFIX_PATTERN = re.compile(
    r"^(?:[(\[<]\s*L?|L)(\d+)\s*[)\]>]?"
    r"(?:\s*[-–]\s*[(\[<]?\s*L?(\d+)\s*[)\]>]?)?"
    r"\s*[:\-–]?\s*"
)
CODE_TOKEN_PATTERN = re.compile(
    r"`([^`]+)`"
    r"|\b([A-Za-z_][A-Za-z0-9]*_[A-Za-z0-9_]+)\b"
    r"|\b([a-z]+[A-Z][A-Za-z0-9]*)\b"
    r"|\b([A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*)\b"
)
IDENTIFIER_IN_SPAN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

EXTRACTION_PROMPT = """
You extract facts from exactly one repository source to answer a question.

Rules:
1. Use only the supplied source. Do not use outside knowledge or guess.
2. Extract at most {max_facts} facts that directly help answer the question.
3. Write each fact as a bullet: "- (L<first>-<last>) <fact>", where the
   line numbers are the source lines that show the fact.
4. Name functions, classes and variables exactly as the source writes them.
5. Describe only what this source shows. Do not describe what other files
   probably do.
6. Do not add citations; the program adds them.
7. If the source does not directly help answer the question, output only:
NONE
{policy}"""


# ---------------------------------------------------------------------------
# Parsing and checking facts
# ---------------------------------------------------------------------------

def is_none_answer(text: str) -> bool:
    # Small models write "NONE.", "- NONE" or "**NONE**"
    return text.strip().strip("*").lstrip("-* ").rstrip(".").upper() == "NONE"


def parse_facts(result: str, max_facts: int = MAX_FACTS_PER_SOURCE) -> list[dict]:
    """
    Turn the model's bullet list into [{"text", "lines": (start, end) | None}].
    Citations the model wrote itself are removed.
    """

    if is_none_answer(result):
        return []

    facts = []

    for line in result.splitlines():
        line = line.strip()

        if not line.startswith(("- ", "* ")):
            continue

        # Nested bullets ("- - fact") happen too
        text = re.sub(r"^(?:[-*]\s+)+", "", line[2:].strip())
        lines = None
        prefix = LINE_PREFIX_PATTERN.match(text)

        if prefix:
            start = int(prefix.group(1))
            end = int(prefix.group(2) or start)
            lines = (min(start, end), max(start, end))
            text = text[prefix.end():]

        text = MODEL_CITATION_PATTERN.sub("", text).strip()

        if text and not is_none_answer(text):
            facts.append({"text": text, "lines": lines})

        if len(facts) == max_facts:
            break

    return facts


def code_identifiers(text: str) -> set[str]:
    """Identifiers a fact claims exist (backticked or obviously code-like)."""

    found = set()

    for match in CODE_TOKEN_PATTERN.finditer(text):
        span = match.group(1)

        if span is not None:
            found.update(IDENTIFIER_IN_SPAN.findall(span))
        else:
            found.add(next(group for group in match.groups()[1:] if group))

    return found


def is_grounded(fact: str, source_text: str) -> bool:
    """
    Every identifier the fact names must appear in the source. Case is
    ignored so "FastAPI" still matches "from fastapi import ...".
    """

    lowered = source_text.lower()
    return all(identifier.lower() in lowered for identifier in code_identifiers(fact))


FILLER_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "by", "with", "to",
    "of", "in", "on", "for", "and", "or", "it", "its", "this", "that",
    "uses", "used", "using", "which", "when", "from", "as",
}


def content_words(text: str) -> set[str]:
    """Tokens without filler words, with simple suffix stripping."""

    words = set()

    for token in tokenize(text):
        if token in FILLER_WORDS:
            continue

        for suffix in ("ing", "ed", "es", "s"):
            if len(token) > len(suffix) + 3 and token.endswith(suffix):
                token = token[: -len(suffix)]
                break

        words.add(token)

    return words


def similarity(first: str, second: str) -> float:
    a, b = content_words(first), content_words(second)
    return len(a & b) / len(a | b) if a and b else 0.0


def is_duplicate(fact: str, earlier: list[str]) -> bool:
    return any(similarity(fact, other) >= DUPLICATE_THRESHOLD for other in earlier)


def numbered(chunk: dict) -> str:
    return "\n".join(
        f"{chunk['start_line'] + offset}| {line}"
        for offset, line in enumerate(chunk["content"].splitlines())
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_facts(question: str, chunk: dict) -> list[dict]:
    tag = safety.new_tag()
    result = llm.chat(
        [
            {
                "role": "system",
                "content": EXTRACTION_PROMPT.format(
                    max_facts=MAX_FACTS_PER_SOURCE,
                    policy=safety.policy(tag),
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    + safety.wrap(numbered(chunk), tag, file=chunk["file_path"])
                ),
            },
        ],
        temperature=0,
    )

    return parse_facts(result)


def cited_lines(chunk: dict, lines: tuple[int, int] | None) -> tuple[int, int]:
    """The model's line range when it is inside the chunk, else the chunk's."""

    if lines and chunk["start_line"] <= lines[0] <= lines[1] <= chunk["end_line"]:
        return lines

    return chunk["start_line"], chunk["end_line"]


def stream_answer(
    question: str,
    index,
    top_k: int = config.TOP_K,
    retrieval_only: bool = False,
    retrieve_lock=None,
    method: str | None = None,
) -> Iterator[dict]:
    """
    Yield progress events and finally {"type": "result", "result": {...}}:

    {
      "related_identifiers": [...],
      "facts": [{"text", "source", "start_line", "end_line", "citation", "url"}],
      "sources": [describe_source(...)],
      "retrieval_only": bool,
      "retrieval_method": str,
      "refusal": None | "low_relevance" | "no_facts",
      "dropped_facts": int,
    }
    """

    yield {"type": "status", "label": "Retrieving sources…"}

    if retrieve_lock is None:
        retrieval = retrieve(question, index, top_k, method)
    else:
        with retrieve_lock:
            retrieval = retrieve(question, index, top_k, method)

    chunks = [index.chunks[position] for position in retrieval.indices]
    base = {
        "related_identifiers": retrieval.related_identifiers,
        "retrieval_method": retrieval.method,
        "best_similarity": round(retrieval.best_similarity, 3),
    }

    if retrieval_only:
        yield {"type": "result", "result": {
            **base,
            "facts": [],
            "sources": [describe_source(chunk) for chunk in chunks],
            "retrieval_only": True,
            "refusal": None,
            "dropped_facts": 0,
        }}
        return

    if config.MIN_RELEVANCE and retrieval.best_similarity < config.MIN_RELEVANCE:
        yield {"type": "result", "result": {
            **base,
            "facts": [],
            "sources": [],
            "retrieval_only": False,
            "refusal": "low_relevance",
            "dropped_facts": 0,
        }}
        return

    facts: list[dict] = []
    used: list[dict] = []
    dropped = 0

    for position, chunk in enumerate(chunks, start=1):
        yield {
            "type": "status",
            "label": f"Reading {chunk['file_path']} ({position}/{len(chunks)})…",
        }

        accepted = []

        for fact in extract_facts(question, chunk):
            if not is_grounded(fact["text"], chunk["content"]) or is_duplicate(
                fact["text"], [earlier["text"] for earlier in facts + accepted]
            ):
                dropped += 1
                continue

            start, end = cited_lines(chunk, fact["lines"])
            accepted.append({
                "text": fact["text"],
                "source": len(used),
                "start_line": start,
                "end_line": end,
                "citation": create_citation(chunk, start, end),
                "url": line_url(chunk, start, end),
            })

        accepted = accepted[:MAX_TOTAL_FACTS - len(facts)]

        if accepted:
            used.append(chunk)
            facts.extend(accepted)

            for fact in accepted:
                yield {"type": "fact", "fact": fact, "source": describe_source(chunk)}

        if len(facts) >= MAX_TOTAL_FACTS:
            break

    yield {"type": "result", "result": {
        **base,
        "facts": facts,
        "sources": [describe_source(chunk) for chunk in used],
        "retrieval_only": False,
        "refusal": None if facts else "no_facts",
        "dropped_facts": dropped,
    }}


def generate_answer(question: str, index, **options) -> dict:
    result = None

    for event in stream_answer(question, index, **options):
        if event["type"] == "result":
            result = event["result"]

    return result


def format_answer(result: dict) -> str:
    """Plain-text answer with citations, as the command line prints it."""

    sources = result["sources"]

    if result["retrieval_only"]:
        return "\n".join(source["citation"] for source in sources)

    if not result["facts"]:
        return NO_EVIDENCE

    return "\n".join(
        f"- {fact['text']} {fact.get('citation') or sources[fact['source']]['citation']}"
        for fact in result["facts"]
    )
