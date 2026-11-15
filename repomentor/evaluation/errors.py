"""
Per-question comparison of two retrieval methods.

    python -m repomentor eval errors --split dev --first semantic --second expanded-hybrid
"""

from ..index import load_index
from .benchmarks import prepare, select
from .retrieval import load_benchmarks, run_method


def command_errors(arguments) -> None:
    k = arguments.top_k
    differences = 0
    total = 0

    for benchmark in load_benchmarks(arguments.benchmark):
        cases = select(benchmark, arguments.split)

        if not cases:
            continue

        index = load_index(prepare(benchmark))
        _, first = run_method(index, cases, arguments.first, (k,))
        _, second = run_method(index, cases, arguments.second, (k,))
        total += len(cases)

        for a, b in zip(first, second):
            if (a["recall"], a["reciprocal_rank"]) == (b["recall"], b["reciprocal_rank"]):
                continue

            differences += 1
            print(f"\n[{benchmark['name']}] {a['question']}")
            print(f"Expected: {a['expected']}")

            for label, result in ((arguments.first, a), (arguments.second, b)):
                print(f"\n{label}:")

                for rank, path in enumerate(result["retrieved"], start=1):
                    marker = "*" if path in result["expected"] else " "
                    print(f"  {marker}{rank}. {path}")

                print(f"  Missing: {result['missing']}")

            print("-" * 70)

    print(f"\nQuestions with different results: {differences}/{total}")
