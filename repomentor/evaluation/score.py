"""
Answer-quality metrics from reviewed answers.

    python -m repomentor eval score evaluation/generated_answers.json
    python -m repomentor eval score evaluation/runs/answers.json --labels judge_review

Labels come from a person (`manual_review`, see `eval review`) or from the
calibrated model judge (`judge_review`, see `eval judge`).
"""

import json


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(results: list[dict], labels: str = "manual_review") -> dict:
    for result in results:
        review = result.get(labels)

        if not review:
            raise ValueError(f"{result['id']} has no {labels}.")

        unreviewed = (
            any(fact["present_and_supported"] is None for fact in review["required_facts"])
            or any(claim["present"] is None for claim in review["incorrect_claims"])
            or review.get("all_claims_supported") is None
            or review.get("all_citations_support_claims") is None
        )

        if unreviewed:
            raise ValueError(f"{result['id']} is not fully reviewed in {labels}.")

    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    substantive = [r for r in results if not r["abstained"]]

    facts = [fact for r in answerable for fact in r[labels]["required_facts"]]
    claims = [claim for r in results for claim in r[labels]["incorrect_claims"]]
    supported = sum(fact["present_and_supported"] for fact in facts)
    complete = sum(
        all(fact["present_and_supported"] for fact in r[labels]["required_facts"])
        for r in answerable
    )
    triggered = sum(claim["present"] for claim in claims)
    answered = sum(not r["abstained"] for r in answerable)
    abstained = sum(r["abstained"] for r in unanswerable)
    faithful = sum(r[labels]["all_claims_supported"] for r in substantive)
    citation_correct = sum(r[labels]["all_citations_support_claims"] for r in substantive)
    recalls = [
        r["expected_source_recall"] for r in answerable
        if r["expected_source_recall"] is not None
    ]

    return {
        "labels": labels,
        "questions": len(results),
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "required_facts": len(facts),
        "supported_required_facts": supported,
        "gold_fact_coverage": ratio(supported, len(facts)),
        "complete_answers": complete,
        "complete_answer_rate": ratio(complete, len(answerable)),
        "answerable_response_rate": ratio(answered, len(answerable)),
        "unanswerable_abstention_accuracy": ratio(abstained, len(unanswerable)),
        "faithful_substantive_answers": faithful,
        "faithful_substantive_answer_rate": ratio(faithful, len(substantive)),
        "citation_correct_substantive_answers": citation_correct,
        "citation_correct_substantive_answer_rate": ratio(citation_correct, len(substantive)),
        "known_incorrect_claims_present": triggered,
        "known_incorrect_claims_checked": len(claims),
        "known_error_trigger_rate": ratio(triggered, len(claims)),
        "mean_expected_source_recall": ratio(sum(recalls), len(recalls)),
    }


def print_metrics(metrics: dict) -> None:
    def percent(value: float) -> str:
        return f"{value * 100:.1f}%"

    print(f"\nAnswer evaluation ({metrics['labels']}):\n")
    print(f"Questions: {metrics['questions']}")
    print(f"Gold fact coverage: {metrics['supported_required_facts']}/{metrics['required_facts']} "
          f"({percent(metrics['gold_fact_coverage'])})")
    print(f"Complete answer rate: {percent(metrics['complete_answer_rate'])}")
    print(f"Answerable response rate: {percent(metrics['answerable_response_rate'])}")
    print(f"Unanswerable abstention accuracy: {percent(metrics['unanswerable_abstention_accuracy'])}")
    print(f"Faithful substantive answer rate: {percent(metrics['faithful_substantive_answer_rate'])}")
    print(f"Citation-correct substantive answer rate: "
          f"{percent(metrics['citation_correct_substantive_answer_rate'])}")
    print(f"Known error trigger rate: {metrics['known_incorrect_claims_present']}/"
          f"{metrics['known_incorrect_claims_checked']} "
          f"({percent(metrics['known_error_trigger_rate'])})")
    print(f"Mean expected-source recall: {percent(metrics['mean_expected_source_recall'])}")


def command_score(arguments) -> None:
    results = json.loads(arguments.answers.read_text())
    metrics = compute_metrics(results, arguments.labels)
    print_metrics(metrics)

    if arguments.output:
        arguments.output.write_text(json.dumps(metrics, indent=2))
        print(f"\nSaved metrics to {arguments.output}")
