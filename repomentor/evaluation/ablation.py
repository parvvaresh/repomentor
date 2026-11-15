"""
Compare retrieval methods (and optional reranking or chunking variants)
on every benchmark, per split and question type.

    python -m repomentor eval ablation --split dev
    python -m repomentor eval ablation --split dev --rerank cross-encoder/ms-marco-MiniLM-L-6-v2
    CHUNKER=lines python -m repomentor eval ablation --variant lines

Results are printed and written as Markdown to evaluation/results/.
"""

from collections import defaultdict
from datetime import date
from pathlib import Path

from .. import config
from ..index import load_index
from ..retrieval import METHODS, load_reranker
from .benchmarks import prepare, select
from .retrieval import HEADER, format_row, load_benchmarks, run_method, summarize


RESULTS_DIRECTORY = Path("evaluation/results")


def table(title: str, rows: list[tuple[str, int, dict]]) -> list[str]:
    lines = [f"## {title}", "", "| Method | K | Hit@K | Recall@K | MRR@K |",
             "| --- | ---: | ---: | ---: | ---: |"]
    lines += [
        f"| {label} | {k} | {m['hit']:.3f} | {m['recall']:.3f} | {m['mrr']:.3f} |"
        for label, k, m in rows
    ]
    return lines + [""]


def print_table(title: str, rows: list[tuple[str, int, dict]]) -> None:
    print(f"\n{title}\n")
    print(HEADER)
    print("-" * len(HEADER))

    for label, k, metrics in rows:
        print(format_row(label, k, metrics))


def command_ablation(arguments) -> None:
    k_values = tuple(sorted(set(arguments.k_values)))
    methods = arguments.methods or list(METHODS)
    variants = [(method, False) for method in methods]

    if arguments.rerank:
        reranker = load_reranker(arguments.rerank)
        variants += [(method, reranker) for method in methods]

    lines = [
        f"# Retrieval ablation ({date.today().isoformat()})",
        "",
        f"Split: `{arguments.split}` · chunker: `{config.CHUNKER}` · "
        f"embedding model: `{config.EMBEDDING_MODEL}` · "
        f"path/symbol context: `{'on' if config.EMBEDDING_CONTEXT else 'off'}`"
        + (f" · reranker: `{arguments.rerank}`" if arguments.rerank else ""),
        "",
    ]
    # (label, kind) -> k -> pooled per-question scores
    pooled = defaultdict(lambda: defaultdict(list))
    benchmarks = load_benchmarks(arguments.benchmark)

    for benchmark in benchmarks:
        cases = select(benchmark, arguments.split)

        if not cases:
            continue

        index = load_index(prepare(benchmark, variant=arguments.variant))
        rows = []

        for method, reranker in variants:
            label = method + ("+rerank" if reranker else "")
            per_k, _ = run_method(index, cases, method, k_values, reranker)

            for k in k_values:
                rows.append((label, k, summarize(per_k[k])))

                for case, result in zip(cases, per_k[k]):
                    pooled[(label, "all")][k].append(result)
                    pooled[(label, case["type"])][k].append(result)

        title = f"{benchmark['name']} ({len(cases)} questions)"
        print_table(title, rows)
        lines += table(title, rows)

    for kind in ("all", "natural", "identifier"):
        rows = [
            (label, k, summarize(results))
            for (label, pooled_kind), per_k in pooled.items() if pooled_kind == kind
            for k, results in per_k.items()
        ]

        if rows:
            title = f"Pooled, {kind} questions ({rows[0][2]['questions']})"
            print_table(title, rows)
            lines += table(title, rows)

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    suffix = f"-{arguments.variant}" if arguments.variant else ""
    suffix += "-rerank" if arguments.rerank else ""
    output = RESULTS_DIRECTORY / f"ablation-{arguments.split}{suffix}.md"
    output.write_text("\n".join(lines))
    print(f"\nSaved to {output}")
