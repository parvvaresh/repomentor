"""
Retrieval methods.

- semantic: cosine similarity between question and chunk embeddings.
- bm25: keyword matching, strongest for exact identifiers.
- hybrid: Reciprocal Rank Fusion of semantic and BM25 rankings.
- expanded-hybrid: hybrid plus a second pass with code identifiers found
  in the first results.
- auto: semantic for natural-language questions, hybrid when the question
  names code identifiers, file names or calls (where BM25 helps).

Every method keeps at most one chunk per file, ranks test files lower
unless the question is about tests, and ranks commit and merge-request
history lower unless the question asks why or when something changed.
An optional cross-encoder can rerank the best candidates.
"""

import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import numpy as np

from . import config


INITIAL_CANDIDATES = 12
RRF_K = 60
TEST_FILE_PENALTY = 0.75
HISTORY_PENALTY = 0.6

METHODS = ("semantic", "bm25", "hybrid", "expanded-hybrid", "auto")

STOPWORDS = {
    "what", "when", "where", "which", "who", "why", "how",
    "are", "does", "the", "this", "that", "with", "from",
    "into", "about", "handled",
}

TEST_DIRECTORIES = {"test", "tests", "__tests__", "spec", "specs"}

HISTORY_WORDS = {
    "why", "when", "changed", "change", "changes", "introduced", "added",
    "removed", "history", "commit", "commits", "merge", "pr", "reason",
    "decision", "decided", "motivation", "renamed", "replaced", "migrated",
}

# snake_case, camelCase, calls, file names and anything in backticks.
# PascalCase is left out: in questions it is mostly product names such as
# FastAPI or GitHub, which semantic search handles well.
IDENTIFIER_PATTERN = re.compile(
    r"`[^`]+`"
    r"|\b[A-Za-z]+_[A-Za-z0-9_]+\b"
    r"|\b[a-z]+[A-Z][A-Za-z0-9]*\b"
    r"|\b\w+\(\)"
    r"|\b[\w/-]+\.(?:py|ts|tsx|js|jsx|md|ya?ml|toml|sql)\b"
)


@dataclass
class Retrieval:
    indices: list[int]
    related_identifiers: list[str]
    method: str
    # Semantic similarity of each returned chunk, used to refuse questions
    # the repository cannot answer
    similarities: list[float] = field(default_factory=list)
    best_similarity: float = 0.0


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    # camelCase -> camel Case, so "getUser" also matches "get_user"
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    expanded = []

    for token in tokens:
        expanded.append(token)
        expanded.extend(part for part in token.split("_") if part)

        if len(token) > 3 and token.endswith("s"):
            expanded.append(token[:-1])

    return expanded


def is_test_file(file_path: str) -> bool:
    path = PurePosixPath(file_path)
    name = path.name.lower()

    if any(part.lower() in TEST_DIRECTORIES for part in path.parts[:-1]):
        return True

    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
        or ".test." in name
        or ".spec." in name
    )


def asks_about_tests(question: str) -> bool:
    return any(token.startswith("test") for token in tokenize(question))


def asks_about_history(question: str) -> bool:
    return bool(HISTORY_WORDS & set(tokenize(question)))


def mentions_identifiers(question: str) -> bool:
    return IDENTIFIER_PATTERN.search(question) is not None


def choose_method(question: str, method: str | None = None) -> str:
    method = (method or config.RETRIEVAL_METHOD).lower()

    if method != "auto":
        return method

    return "hybrid" if mentions_identifiers(question) else "semantic"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def semantic_scores(index, text: str) -> np.ndarray:
    vector = index.embedding_model.encode(text, normalize_embeddings=True)
    return index.store.scores(vector, len(index.chunks))


def keyword_ranking(scores: np.ndarray) -> list[int]:
    """
    Rank chunks by BM25 score, keeping only chunks that share at least one
    term with the query: zero-score chunks have an arbitrary order and must
    not receive rank-fusion credit.
    """

    ranking = np.argsort(scores)[::-1]
    return [int(position) for position in ranking if scores[position] > 0]


def semantic_ranking(scores: np.ndarray) -> list[int]:
    ranking = np.argsort(scores)[::-1]
    return [int(position) for position in ranking if np.isfinite(scores[position])]


def reciprocal_rank_fusion(rankings: list[list[int]], item_count: int) -> np.ndarray:
    combined = np.zeros(item_count)

    for ranking in rankings:
        for rank, position in enumerate(ranking, start=1):
            combined[position] += 1 / (RRF_K + rank)

    return combined


def find_related_identifiers(
    question: str,
    chunks: list[dict],
    candidate_indices: list[int],
) -> list[str]:
    """Identifiers in the given chunks that share words with the question."""

    question_terms = {
        token
        for token in tokenize(question)
        if len(token) >= 4 and token not in STOPWORDS
    }
    counts = Counter()

    for position in candidate_indices:
        for identifier in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", chunks[position]["content"]):
            if "_" in identifier and question_terms & set(tokenize(identifier)):
                counts[identifier] += 1

    return [identifier for identifier, _ in counts.most_common(25)]


def adjust_scores(scores: np.ndarray, chunks: list[dict], question: str) -> np.ndarray:
    """Apply the test-file and history penalties."""

    adjusted = scores.astype(float).copy()
    penalize_tests = not asks_about_tests(question)
    penalize_history = not asks_about_history(question)

    for position, chunk in enumerate(chunks):
        if not np.isfinite(adjusted[position]):
            continue

        kind = chunk.get("kind", "code")
        factor = 1.0

        if penalize_tests and kind == "code" and is_test_file(chunk["file_path"]):
            factor = TEST_FILE_PENALTY

        if penalize_history and kind in ("commit", "merge_request"):
            factor = HISTORY_PENALTY

        if factor != 1.0:
            # Multiplying a negative cosine score would raise it
            adjusted[position] = (
                adjusted[position] * factor if adjusted[position] > 0
                else adjusted[position] / factor
            )

    return adjusted


def unique_files(
    scores: np.ndarray,
    chunks: list[dict],
    limit: int,
    positive_only: bool = False,
) -> list[int]:
    best = []
    seen = set()

    for position in np.argsort(scores)[::-1]:
        if not np.isfinite(scores[position]):
            break

        if positive_only and scores[position] <= 0:
            continue

        file_path = chunks[position]["file_path"]

        if file_path in seen:
            continue

        best.append(int(position))
        seen.add(file_path)

        if len(best) == limit:
            break

    return best


# ---------------------------------------------------------------------------
# Methods: each returns (scores, positive_only, related_identifiers)
# ---------------------------------------------------------------------------

def score_semantic(question, index):
    return semantic_scores(index, question), False, []


def score_bm25(question, index):
    return index.bm25.get_scores(tokenize(question)), True, []


def score_hybrid(question, index):
    semantic = semantic_scores(index, question)
    keyword = index.bm25.get_scores(tokenize(question))
    combined = reciprocal_rank_fusion(
        [semantic_ranking(semantic), keyword_ranking(keyword)],
        len(index.chunks),
    )
    return combined, False, []


def score_expanded_hybrid(question, index):
    initial_semantic = semantic_scores(index, question)
    initial_keyword = index.bm25.get_scores(tokenize(question))

    candidates = list(dict.fromkeys(
        semantic_ranking(initial_semantic)[:INITIAL_CANDIDATES]
        + keyword_ranking(initial_keyword)[:INITIAL_CANDIDATES]
    ))
    identifiers = find_related_identifiers(question, index.chunks, candidates)
    expanded_query = f"{question} {' '.join(identifiers)}"

    semantic = np.maximum(initial_semantic, semantic_scores(index, expanded_query))
    keyword = index.bm25.get_scores(tokenize(expanded_query))
    combined = reciprocal_rank_fusion(
        [semantic_ranking(semantic), keyword_ranking(keyword)],
        len(index.chunks),
    )
    return combined, False, identifiers


SCORERS = {
    "semantic": score_semantic,
    "bm25": score_bm25,
    "hybrid": score_hybrid,
    "expanded-hybrid": score_expanded_hybrid,
}


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

_rerankers: dict[str, object] = {}
_reranker_lock = threading.Lock()


def load_reranker(name: str):
    with _reranker_lock:
        if name not in _rerankers:
            from sentence_transformers import CrossEncoder

            _rerankers[name] = CrossEncoder(name)

    return _rerankers[name]


def rerank(question: str, chunks: list[dict], candidates: list[int], model) -> list[int]:
    if len(candidates) < 2:
        return candidates

    scores = model.predict(
        [
            (question, f"{chunks[position]['file_path']}\n{chunks[position]['content']}")
            for position in candidates
        ]
    )
    order = np.argsort(np.asarray(scores))[::-1]
    return [candidates[position] for position in order]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def retrieve(
    question: str,
    index,
    top_k: int = config.TOP_K,
    method: str | None = None,
    reranker=None,
) -> Retrieval:
    """
    Return the best chunks for a question (at most one per file).
    `reranker` is a CrossEncoder-like object; by default the one named by
    RERANKER_MODEL is used, if any. Pass reranker=False to disable it.
    """

    chosen = choose_method(question, method)

    if chosen not in SCORERS:
        raise ValueError(f"Unknown retrieval method '{chosen}'.")

    scores, positive_only, identifiers = SCORERS[chosen](question, index)
    adjusted = adjust_scores(scores, index.chunks, question)

    if reranker is None and config.RERANKER_MODEL:
        reranker = load_reranker(config.RERANKER_MODEL)

    if reranker:
        candidates = unique_files(
            adjusted, index.chunks, max(top_k, config.RERANK_CANDIDATES), positive_only
        )
        best = rerank(question, index.chunks, candidates, reranker)[:top_k]
    else:
        best = unique_files(adjusted, index.chunks, top_k, positive_only)

    # Similarity is reported with the plain question, whatever the method
    similarity = semantic_scores(index, question) if chosen != "semantic" else scores
    finite = similarity[np.isfinite(similarity)]

    if not identifiers:
        identifiers = find_related_identifiers(question, index.chunks, best)

    return Retrieval(
        indices=best,
        related_identifiers=identifiers,
        method=chosen + ("+rerank" if reranker else ""),
        similarities=[float(similarity[position]) for position in best],
        best_similarity=float(finite.max()) if finite.size else 0.0,
    )
