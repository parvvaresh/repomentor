import argparse
from pathlib import Path

import numpy as np

from evaluate_retrieval import (
    DEFAULT_DATASET_PATH,
    load_dataset,
    score_case,
)
from rag_answer import (
    TEST_FILE_PENALTY,
    asks_about_tests,
    keyword_ranking,
    load_index,
    reciprocal_rank_fusion,
    retrieve,
    tokenize,
)


DEFAULT_K_VALUES = [1, 3, 5, 10]


def select_unique_files(
    scores: np.ndarray,
    index: dict,
    question: str,
    top_k: int,
    positive_only: bool = False,
) -> list[int]:
    """
    Rank chunks and keep at most one chunk from each file.

    The test-file penalty and file diversity are shared across methods
    so the comparison focuses on the retrieval method itself.
    """

    adjusted_scores = scores.astype(float).copy()

    if not asks_about_tests(question):
        for chunk_index, chunk in enumerate(index["chunks"]):
            file_path = chunk["file_path"]

            if "/tests/" in file_path or file_path.startswith("tests/"):
                adjusted_scores[chunk_index] *= TEST_FILE_PENALTY

    best_indices = []
    seen_files = set()

    for chunk_index in np.argsort(adjusted_scores)[::-1]:
        if positive_only and scores[chunk_index] <= 0:
            continue

        file_path = index["chunks"][chunk_index]["file_path"]

        if file_path in seen_files:
            continue

        best_indices.append(int(chunk_index))
        seen_files.add(file_path)

        if len(best_indices) == top_k:
            break

    return best_indices


def retrieve_semantic(
    question: str,
    index: dict,
    top_k: int,
) -> list[int]:
    """Retrieve using embedding similarity only."""

    question_embedding = index["embedding_model"].encode(
        question,
        normalize_embeddings=True,
    )

    scores = index["embeddings"] @ question_embedding

    return select_unique_files(scores, index, question, top_k)


def retrieve_bm25(
    question: str,
    index: dict,
    top_k: int,
) -> list[int]:
    """Retrieve using BM25 keyword matching only."""

    scores = index["bm25"].get_scores(tokenize(question))

    return select_unique_files(
        scores,
        index,
        question,
        top_k,
        positive_only=True,
    )


def retrieve_hybrid(
    question: str,
    index: dict,
    top_k: int,
) -> list[int]:
    """Combine semantic and BM25 rankings without query expansion."""

    question_embedding = index["embedding_model"].encode(
        question,
        normalize_embeddings=True,
    )

    semantic_scores = index["embeddings"] @ question_embedding
    keyword_scores = index["bm25"].get_scores(tokenize(question))

    semantic_ranking = np.argsort(semantic_scores)[::-1].tolist()
    bm25_ranking = keyword_ranking(keyword_scores)

    combined_scores = reciprocal_rank_fusion(
        [semantic_ranking, bm25_ranking],
        len(index["chunks"]),
    )

    return select_unique_files(
        combined_scores,
        index,
        question,
        top_k,
    )


def retrieve_expanded_hybrid(
    question: str,
    index: dict,
    top_k: int,
) -> list[int]:
    """Use RepoMentor's complete two-pass retrieval pipeline."""

    best_indices, _ = retrieve(question, index, top_k)
    return best_indices


METHODS = {
    "semantic": retrieve_semantic,
    "bm25": retrieve_bm25,
    "hybrid": retrieve_hybrid,
    "expanded-hybrid": retrieve_expanded_hybrid,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RepoMentor retrieval methods.",
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )

    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=DEFAULT_K_VALUES,
    )

    arguments = parser.parse_args()

    if any(k < 1 for k in arguments.k_values):
        parser.error("Every K value must be at least 1.")

    return arguments


def main() -> None:
    arguments = parse_arguments()
    k_values = sorted(set(arguments.k_values))
    max_k = max(k_values)

    cases = load_dataset(arguments.dataset)
    index = load_index()

    results = {
        method: {k: [] for k in k_values}
        for method in METHODS
    }

    print(
        f"Evaluating {len(cases)} questions "
        f"with {len(METHODS)} retrieval methods..."
    )

    for case in cases:
        for method_name, retrieval_function in METHODS.items():
            best_indices = retrieval_function(
                case["question"],
                index,
                max_k,
            )

            retrieved_files = [
                index["chunks"][i]["file_path"]
                for i in best_indices
            ]

            for k in k_values:
                result = score_case(
                    retrieved_files[:k],
                    case["expected_files"],
                )
                results[method_name][k].append(result)

    print("\nAblation results:\n")
    print(
        f"{'Method':<20}"
        f"{'K':>4}"
        f"{'Hit@K':>10}"
        f"{'Recall@K':>12}"
        f"{'MRR@K':>10}"
    )
    print("-" * 56)

    for method_name in METHODS:
        for k in k_values:
            method_results = results[method_name][k]
            count = len(method_results)

            hit = sum(result["hit"] for result in method_results) / count
            recall = (
                sum(result["recall"] for result in method_results) / count
            )
            mrr = (
                sum(
                    result["reciprocal_rank"]
                    for result in method_results
                )
                / count
            )

            print(
                f"{method_name:<20}"
                f"{k:>4}"
                f"{hit:>10.3f}"
                f"{recall:>12.3f}"
                f"{mrr:>10.3f}"
            )


if __name__ == "__main__":
    main()