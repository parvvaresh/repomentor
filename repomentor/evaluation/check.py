"""
Quality regression gate for continuous integration.

    python -m repomentor eval check --retrieval
    python -m repomentor eval check --answers      # needs Ollama

Retrieval runs every benchmark with the default retrieval settings and
fails when Hit@K, Recall@K or MRR@K drop below evaluation/thresholds.json.
Answers are generated for the answer gold set on the pinned FastAPI
template, labeled by the model judge, and compared with the answer
thresholds. Thresholds are set a little below the last accepted run, so
the gate catches regressions without failing on noise.
"""

import json
import sys
from pathlib import Path

from .. import config
from ..index import load_index
from .answers import DATASET_PATH, generate
from .benchmarks import load_benchmark, prepare, select
from .judge import judge_result
from .retrieval import load_benchmarks, run_method, summarize
from .score import compute_metrics


THRESHOLDS_PATH = Path("evaluation/thresholds.json")


def check_retrieval(thresholds: dict) -> list[str]:
    failures = []
    settings = thresholds["retrieval"]
    k = settings["k"]

    for benchmark in load_benchmarks(None):
        minimum = settings["benchmarks"].get(benchmark["name"])

        if minimum is None:
            continue

        cases = select(benchmark, settings.get("split", "all"))
        index = load_index(prepare(benchmark))
        per_k, _ = run_method(index, cases, config.RETRIEVAL_METHOD, (k,))
        metrics = summarize(per_k[k])

        print(f"{benchmark['name']}: hit@{k} {metrics['hit']:.3f}  "
              f"recall@{k} {metrics['recall']:.3f}  mrr@{k} {metrics['mrr']:.3f}")

        for name in ("hit", "recall", "mrr"):
            if metrics[name] < minimum[name]:
                failures.append(
                    f"{benchmark['name']} {name}@{k} {metrics[name]:.3f} < {minimum[name]:.3f}"
                )

    return failures


def check_answers(thresholds: dict) -> list[str]:
    settings = thresholds["answers"]
    benchmark = load_benchmark(Path(settings["benchmark"]))
    index = load_index(prepare(benchmark))
    cases = json.loads(DATASET_PATH.read_text())
    results = generate(cases, index, config.TOP_K)

    for result in results:
        result["judge_review"] = judge_result(result, config.JUDGE_MODEL)

    metrics = compute_metrics(results, "judge_review")
    output = Path("evaluation/results/answers-check.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": metrics, "results": results}, indent=2))

    failures = []

    for name, minimum in settings["minimums"].items():
        value = metrics[name]
        print(f"{name}: {value:.3f} (minimum {minimum:.3f})")

        if value < minimum:
            failures.append(f"{name} {value:.3f} < {minimum:.3f}")

    for name, maximum in settings.get("maximums", {}).items():
        value = metrics[name]
        print(f"{name}: {value:.3f} (maximum {maximum:.3f})")

        if value > maximum:
            failures.append(f"{name} {value:.3f} > {maximum:.3f}")

    return failures


def command_check(arguments) -> None:
    thresholds = json.loads(THRESHOLDS_PATH.read_text())
    failures = []

    if arguments.retrieval or not arguments.answers:
        print("Retrieval regression check\n")
        failures += check_retrieval(thresholds)

    if arguments.answers:
        print("\nAnswer quality regression check (model judge)\n")
        failures += check_answers(thresholds)

    if failures:
        print("\nREGRESSIONS:")

        for failure in failures:
            print(f"- {failure}")

        sys.exit(1)

    print("\nAll quality checks passed.")
