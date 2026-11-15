"""
File-level retrieval metrics: Hit@K, Recall@K and MRR@K.

    python -m repomentor eval retrieval --split test
    python -m repomentor eval ablation --split dev
"""

from pathlib import Path

from ..retrieval import retrieve
from .benchmarks import benchmark_paths, load_benchmark, prepare, select


DEFAULT_K_VALUES = (1, 3, 5, 10)


def score_case(retrieved_files: list[str], expected_files: list[str]) -> dict:
    """Compute file-level retrieval metrics for one question."""

    expected = set(expected_files)
    found = expected.intersection(retrieved_files)

    # Rank of the first relevant file, used for Mean Reciprocal Rank
    first_rank = next(
        (rank for rank, path in enumerate(retrieved_files, start=1) if path in expected),
        None,
    )

    return {
        "hit": 1.0 if found else 0.0,
        "recall": len(found) / len(expected),
        "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
        "missing": sorted(expected - found),
    }


def summarize(results: list[dict]) -> dict:
    count = len(results) or 1
    return {
        "questions": len(results),
        "hit": sum(r["hit"] for r in results) / count,
        "recall": sum(r["recall"] for r in results) / count,
        "mrr": sum(r["reciprocal_rank"] for r in results) / count,
    }


def run_method(index, cases: list[dict], method: str, k_values=DEFAULT_K_VALUES, reranker=False):
    """
    Returns ({k: [per-question scores, in case order]}, per-question details).
    `reranker` is False (off), or a CrossEncoder-like object.
    """

    max_k = max(k_values)
    per_k = {k: [] for k in k_values}
    details = []

    for case in cases:
        retrieval = retrieve(case["question"], index, max_k, method, reranker=reranker)
        files = [index.chunks[position]["file_path"] for position in retrieval.indices]

        for k in k_values:
            per_k[k].append(score_case(files[:k], case["expected_files"]))

        details.append({
            "id": case.get("id"),
            "question": case["question"],
            "expected": case["expected_files"],
            "retrieved": files,
            "method": retrieval.method,
            **score_case(files, case["expected_files"]),
        })

    return per_k, details


def load_benchmarks(paths: list[Path] | None) -> list[dict]:
    return [load_benchmark(path) for path in (paths or benchmark_paths())]


def format_row(label: str, k: int, metrics: dict) -> str:
    return (
        f"{label:<28}{k:>4}{metrics['hit']:>10.3f}"
        f"{metrics['recall']:>12.3f}{metrics['mrr']:>10.3f}"
    )


HEADER = f"{'Method':<28}{'K':>4}{'Hit@K':>10}{'Recall@K':>12}{'MRR@K':>10}"


def command_retrieval(arguments) -> None:
    for benchmark in load_benchmarks(arguments.benchmark):
        cases = select(benchmark, arguments.split, arguments.type)

        if not cases:
            continue

        index = load_prepared(benchmark)
        per_k, details = run_method(index, cases, arguments.method, (arguments.top_k,))

        print(f"\n{benchmark['name']} ({arguments.split}, {len(cases)} questions, "
              f"method {arguments.method})\n")

        for detail in details:
            status = "HIT " if detail["hit"] else "MISS"
            print(f"{status} [{detail['method']}] {detail['question']}")

            if detail["missing"]:
                print(f"     missing: {', '.join(detail['missing'])}")

        metrics = summarize(per_k[arguments.top_k])
        print(f"\nHit@{arguments.top_k}: {metrics['hit']:.3f}  "
              f"Recall@{arguments.top_k}: {metrics['recall']:.3f}  "
              f"MRR@{arguments.top_k}: {metrics['mrr']:.3f}")


def load_prepared(benchmark: dict):
    from ..index import load_index

    return load_index(prepare(benchmark))
