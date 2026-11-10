import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import numpy as np
import requests
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


load_dotenv()

# Any chat model pulled into Ollama can be used, e.g. OLLAMA_MODEL=qwen2.5:1.5b
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 5
INITIAL_CANDIDATES = 12
MAX_FACTS_PER_SOURCE = 2
MAX_TOTAL_FACTS = 6

# Standard Reciprocal Rank Fusion constant
RRF_K = 60
TEST_FILE_PENALTY = 0.75

CHUNKS_PATH = Path("data/chunks.jsonl")
EMBEDDINGS_PATH = Path("data/embeddings.npy")
OLLAMA_URL = "http://localhost:11434/api/chat"

# Source links must point at the configured GitLab instance,
# which may be self-hosted rather than gitlab.com
GITLAB_BASE_URL = os.getenv("GITLAB_BASE_URL", "https://gitlab.com").rstrip("/")

STOPWORDS = {
    "what", "when", "where", "which", "who", "why", "how",
    "are", "does", "the", "this", "that", "with", "from",
    "into", "about", "handled",
}

TEST_DIRECTORIES = {"test", "tests", "__tests__", "spec", "specs"}

# Matches citations a model may append on its own, e.g.
# "[backend/app/crud.py:1-60]", "[1]" or "[Source 2]", without
# touching code such as "dict[str, int]"
MODEL_CITATION_PATTERN = re.compile(
    r"(?:\s*\[(?:[^\]\s]+:\d+(?:-\d+)?|\d+|source[^\]]*)\])+\s*$",
    re.IGNORECASE,
)


def call_ollama(
    messages: list[dict],
    temperature: float = 0,
) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {"temperature": temperature},
            },
            timeout=180,
        )
    except requests.ConnectionError as error:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}. "
            "Make sure Ollama is running, or use --retrieval-only."
        ) from error

    response.raise_for_status()

    return response.json()["message"]["content"].strip()


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    expanded = []

    for token in tokens:
        expanded.append(token)
        expanded.extend(token.split("_"))

        if len(token) > 3 and token.endswith("s"):
            expanded.append(token[:-1])

    return expanded


def load_chunks(chunks_path: Path = CHUNKS_PATH) -> list[dict]:
    chunks = []

    with chunks_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def load_index(
    chunks_path: Path = CHUNKS_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
    embedding_model: SentenceTransformer | None = None,
) -> dict:
    """
    Load chunks, embeddings, the embedding model and the BM25 index once
    so that several questions can be answered without reloading them.
    Pass an already loaded `embedding_model` to share it between indexes.
    """

    if not chunks_path.exists() or not embeddings_path.exists():
        raise RuntimeError(
            "Index files are missing. "
            "Run chunk_dataset.py and create_embeddings.py first."
        )

    chunks = load_chunks(chunks_path)
    embeddings = np.load(embeddings_path)

    if not chunks:
        raise RuntimeError(f"No chunks found in {chunks_path}.")

    if len(chunks) != len(embeddings):
        raise RuntimeError(
            "Chunks and embeddings do not match. "
            "Run create_embeddings.py again."
        )

    documents = [
        tokenize(f"{chunk['file_path']}\n{chunk['content']}")
        for chunk in chunks
    ]

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "embedding_model": (
            embedding_model or SentenceTransformer(EMBEDDING_MODEL)
        ),
        "bm25": BM25Okapi(documents),
    }


def find_related_identifiers(
    question: str,
    chunks: list[dict],
    candidate_indices: list[int],
) -> list[str]:
    """
    Find code identifiers in initially retrieved chunks that overlap
    with words from the user's question.
    """

    question_terms = {
        token
        for token in tokenize(question)
        if len(token) >= 4 and token not in STOPWORDS
    }

    identifier_counts = Counter()

    for index in candidate_indices:
        content = chunks[index]["content"]

        identifiers = re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
            content,
        )

        for identifier in identifiers:
            identifier_terms = set(tokenize(identifier))

            if "_" in identifier and question_terms & identifier_terms:
                identifier_counts[identifier] += 1

    return [
        identifier
        for identifier, _ in identifier_counts.most_common(25)
    ]


def is_test_file(file_path: str) -> bool:
    path = PurePosixPath(file_path)
    name = path.name.lower()

    if any(part.lower() in TEST_DIRECTORIES for part in path.parts[:-1]):
        return True

    # Common naming conventions for Python and JavaScript/TypeScript tests
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
        or ".test." in name
        or ".spec." in name
    )


def asks_about_tests(question: str) -> bool:
    # Match whole words such as "test", "tests" or "testing",
    # so that words like "latest" do not count
    return any(token.startswith("test") for token in tokenize(question))


def keyword_ranking(scores: np.ndarray) -> list[int]:
    """
    Rank chunks by BM25 score, keeping only chunks that share at least
    one term with the query. Chunks with a zero score have an arbitrary
    order and must not receive rank-fusion credit.
    """

    ranking = np.argsort(scores)[::-1]

    return [int(index) for index in ranking if scores[index] > 0]


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    item_count: int,
) -> np.ndarray:
    combined_scores = np.zeros(item_count)

    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            combined_scores[index] += 1 / (RRF_K + rank)

    return combined_scores


def retrieve(
    question: str,
    index: dict,
    top_k: int = TOP_K,
) -> tuple[list[int], list[str]]:
    """
    Run the two-pass hybrid retrieval and return the indices of the best
    chunks (at most one per file) plus the related identifiers found.
    """

    chunks = index["chunks"]
    embeddings = index["embeddings"]
    embedding_model = index["embedding_model"]
    bm25 = index["bm25"]

    # First retrieval pass
    question_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
    )

    initial_semantic_scores = embeddings @ question_embedding
    initial_keyword_scores = bm25.get_scores(tokenize(question))

    initial_semantic_ranking = np.argsort(initial_semantic_scores)[::-1]
    initial_keyword_ranking = keyword_ranking(initial_keyword_scores)

    initial_candidates = list(
        dict.fromkeys(
            initial_semantic_ranking[:INITIAL_CANDIDATES].tolist()
            + initial_keyword_ranking[:INITIAL_CANDIDATES]
        )
    )

    # Extract code identifiers from the initial results
    related_identifiers = find_related_identifiers(
        question,
        chunks,
        initial_candidates,
    )

    # Second retrieval pass using discovered identifiers
    expanded_query = f"{question} {' '.join(related_identifiers)}"

    expanded_embedding = embedding_model.encode(
        expanded_query,
        normalize_embeddings=True,
    )

    semantic_scores = np.maximum(
        initial_semantic_scores,
        embeddings @ expanded_embedding,
    )

    keyword_scores = bm25.get_scores(tokenize(expanded_query))

    combined_scores = reciprocal_rank_fusion(
        [
            np.argsort(semantic_scores)[::-1].tolist(),
            keyword_ranking(keyword_scores),
        ],
        len(chunks),
    )

    # Prefer implementation files unless the user asks about tests
    if not asks_about_tests(question):
        for chunk_index, chunk in enumerate(chunks):
            if is_test_file(chunk["file_path"]):
                combined_scores[chunk_index] *= TEST_FILE_PENALTY

    # Select different files for broader evidence
    best_indices = []
    seen_files = set()

    for chunk_index in np.argsort(combined_scores)[::-1]:
        file_path = chunks[chunk_index]["file_path"]

        if file_path in seen_files:
            continue

        best_indices.append(int(chunk_index))
        seen_files.add(file_path)

        if len(best_indices) == top_k:
            break

    return best_indices, related_identifiers


def create_citation(chunk: dict) -> str:
    return (
        f"[{chunk['file_path']}:"
        f"{chunk['start_line']}-{chunk['end_line']}]"
    )


def create_source_url(chunk: dict) -> str:
    encoded_path = quote(chunk["file_path"])

    return (
        f"{GITLAB_BASE_URL}/{chunk['project_path']}/-/blob/"
        f"{chunk['last_commit_id']}/{encoded_path}"
        f"#L{chunk['start_line']}-{chunk['end_line']}"
    )


def is_none_answer(text: str) -> bool:
    # Small models sometimes write "NONE." or "- NONE" instead of "NONE"
    return text.strip().lstrip("-* ").rstrip(".").upper() == "NONE"


def parse_facts(result: str) -> list[str]:
    """
    Turn the model's bullet list into plain facts. Any citation the
    model wrote is removed; callers attach the real source themselves.
    """

    if is_none_answer(result):
        return []

    facts = []

    for line in result.splitlines():
        line = line.strip()

        if not line.startswith(("- ", "* ")):
            continue

        # Remove any citation the model added itself.
        fact = MODEL_CITATION_PATTERN.sub("", line[2:]).strip()

        if fact and not is_none_answer(fact):
            facts.append(fact)

        if len(facts) == MAX_FACTS_PER_SOURCE:
            break

    return facts


def extract_facts(question: str, chunk: dict) -> list[str]:
    """
    Ask the model to extract facts from only one source.
    Python attaches the citation afterward.
    """

    result = call_ollama(
        [
            {
                "role": "system",
                "content": f"""
You are extracting facts from exactly one repository source.

Rules:
1. Use only the supplied source.
2. Extract at most {MAX_FACTS_PER_SOURCE} facts relevant to the question.
3. Output each fact as a bullet beginning with "- ".
4. Do not add citations; the program adds them automatically.
5. Do not use outside knowledge or make assumptions.
6. If the source does not directly help answer the question, output:
NONE
""",
            },
            {
                "role": "user",
                "content": f"""
Question:
{question}

File:
{chunk['file_path']}

Source:
{chunk['content']}
""",
            },
        ],
        temperature=0,
    )

    return parse_facts(result)


def describe_source(chunk: dict) -> dict:
    return {
        "chunk_id": chunk.get("chunk_id"),
        "project_path": chunk["project_path"],
        "file_path": chunk["file_path"],
        "start_line": chunk["start_line"],
        "end_line": chunk["end_line"],
        "commit_id": chunk["last_commit_id"],
        "citation": create_citation(chunk),
        "url": create_source_url(chunk),
        "content": chunk["content"],
    }


def generate_answer(
    question: str,
    index: dict,
    top_k: int = TOP_K,
    retrieval_only: bool = False,
    retrieve_lock=None,
) -> dict:
    """
    Answer a question and return the result as data:

    {
      "related_identifiers": [...],
      "facts": [{"text": "...", "source": 0}, ...],
      "sources": [{citation, url, file_path, ...}, ...],
      "retrieval_only": bool,
    }

    Each fact's "source" is a position in "sources". In retrieval-only
    mode "facts" is empty and "sources" lists everything retrieved.
    `retrieve_lock` optionally serializes access to the shared embedding
    model when several threads answer questions at once.
    """

    if retrieve_lock is None:
        best_indices, related_identifiers = retrieve(question, index, top_k)
    else:
        with retrieve_lock:
            best_indices, related_identifiers = retrieve(
                question,
                index,
                top_k,
            )

    retrieved_chunks = [index["chunks"][i] for i in best_indices]

    # Skip the language model and only show what retrieval found
    if retrieval_only:
        return {
            "related_identifiers": related_identifiers,
            "facts": [],
            "sources": [describe_source(chunk) for chunk in retrieved_chunks],
            "retrieval_only": True,
        }

    # Extract grounded facts one source at a time
    answer_facts = []
    used_chunks = []

    for chunk in retrieved_chunks:
        facts = extract_facts(question, chunk)

        if facts:
            facts = facts[:MAX_TOTAL_FACTS - len(answer_facts)]
            answer_facts.extend(
                {"text": fact, "source": len(used_chunks)}
                for fact in facts
            )
            used_chunks.append(chunk)

        if len(answer_facts) >= MAX_TOTAL_FACTS:
            break

    return {
        "related_identifiers": related_identifiers,
        "facts": answer_facts,
        "sources": [describe_source(chunk) for chunk in used_chunks],
        "retrieval_only": False,
    }


def print_sources(sources: list[dict]) -> None:
    print("\nSources:\n")

    for source in sources:
        print(source["citation"])
        print(source["url"])
        print()


def answer_question(
    question: str,
    index: dict,
    top_k: int = TOP_K,
    retrieval_only: bool = False,
) -> None:
    result = generate_answer(question, index, top_k, retrieval_only)
    sources = result["sources"]

    print("\nRelated code identifiers:")

    if result["related_identifiers"]:
        print(", ".join(result["related_identifiers"]))
    else:
        print("None found")

    if retrieval_only:
        print_sources(sources)
        return

    print("\nAnswer:\n")

    if result["facts"]:
        for fact in result["facts"]:
            citation = sources[fact["source"]]["citation"]
            print(f"- {fact['text']} {citation}")
    else:
        print(
            "- I couldn't find enough evidence in the "
            "retrieved repository sources."
        )

    print_sources(sources)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask grounded questions about the indexed repository.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to answer. Omit it to start an interactive session.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of source files to retrieve (default: {TOP_K}).",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Show retrieved sources without calling the language model.",
    )

    arguments = parser.parse_args()

    if arguments.top_k < 1:
        parser.error("--top-k must be at least 1.")

    return arguments


def main() -> None:
    arguments = parse_arguments()
    index = load_index()

    if arguments.question:
        answer_question(
            arguments.question.strip(),
            index,
            arguments.top_k,
            arguments.retrieval_only,
        )
        return

    # Interactive session: the index is loaded once for all questions
    print("Type a question, or press Enter / type 'exit' to quit.")

    while True:
        try:
            question = input("\nAsk RepoMentor: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question or question.lower() in {"exit", "quit"}:
            break

        answer_question(
            question,
            index,
            arguments.top_k,
            arguments.retrieval_only,
        )


if __name__ == "__main__":
    main()
