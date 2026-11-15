import json
from pathlib import Path

import pytest

from repomentor.evaluation.benchmarks import benchmark_paths, load_benchmark, select
from repomentor.evaluation.judge import calibration_report, cohen_kappa, judge_result, parse_verdict
from repomentor.evaluation.retrieval import score_case, summarize
from repomentor.evaluation.score import compute_metrics


ROOT = Path(__file__).resolve().parent.parent


def test_score_case():
    result = score_case(["b.py", "a.py", "c.py"], ["a.py", "d.py"])
    assert result == {"hit": 1.0, "recall": 0.5, "reciprocal_rank": 0.5, "missing": ["d.py"]}
    assert summarize([result])["mrr"] == 0.5


def test_benchmarks_are_well_formed():
    paths = [ROOT / path for path in benchmark_paths()] or sorted((ROOT / "evaluation/benchmarks").glob("*.json"))
    assert len(paths) >= 2

    total = 0

    for path in paths:
        benchmark = load_benchmark(path)
        ids = [case["id"] for case in benchmark["questions"]]

        assert len(ids) == len(set(ids)), f"duplicate ids in {path.name}"
        assert len(benchmark["repository"]["commit"]) == 40
        assert select(benchmark, "dev") and select(benchmark, "test")
        total += len(benchmark["questions"])

    assert total > 16


def test_human_baseline_metrics_are_reproduced():
    """The README's answer-quality numbers come from this file."""

    results = json.loads((ROOT / "evaluation/generated_answers.json").read_text())
    metrics = compute_metrics(results)

    assert metrics["supported_required_facts"] == 8
    assert metrics["required_facts"] == 29
    assert metrics["faithful_substantive_answer_rate"] == pytest.approx(0.5)
    assert metrics["known_incorrect_claims_present"] == 2


def test_cohen_kappa():
    assert cohen_kappa([(True, True), (False, False)]) == 1.0
    assert cohen_kappa([(True, False), (False, True)]) == -1.0
    assert cohen_kappa([]) is None


@pytest.mark.parametrize(("text", "verdict"), [("YES. It does.", True), ("no", False), ("maybe", None)])
def test_parse_verdict(text, verdict):
    assert parse_verdict(text) is verdict


def test_judge_and_calibration(monkeypatch):
    from repomentor import llm

    results = json.loads((ROOT / "evaluation/generated_answers.json").read_text())[:2]
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "NO")

    for result in results:
        result["judge_review"] = judge_result(result, "fake")

    report = calibration_report(results, "fake")
    facts = report["labels"]["required_fact"]

    assert facts["judge_yes"] == 0
    assert facts["items"] == sum(len(r["manual_review"]["required_facts"]) for r in results)
    assert 0 <= facts["agreement"] <= 1
