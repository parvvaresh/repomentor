import json
from pathlib import Path


INPUT_PATH = Path("evaluation/generated_answers.json")
OUTPUT_PATH = Path("evaluation/answer_evaluation_results.json")


def percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    results = json.loads(INPUT_PATH.read_text())

    answerable = [result for result in results if result["answerable"]]
    unanswerable = [result for result in results if not result["answerable"]]
    substantive = [result for result in results if not result["abstained"]]

    required_fact_reviews = [
        fact
        for result in answerable
        for fact in result["manual_review"]["required_facts"]
    ]

    incorrect_claim_reviews = [
        claim
        for result in results
        for claim in result["manual_review"]["incorrect_claims"]
    ]

    # Ensure manual review is complete.
    for fact in required_fact_reviews:
        if fact["present_and_supported"] is None:
            raise ValueError("A required fact has not been reviewed.")

    for claim in incorrect_claim_reviews:
        if claim["present"] is None:
            raise ValueError("An incorrect claim has not been reviewed.")

    for result in results:
        review = result["manual_review"]

        if review["all_claims_supported"] is None:
            raise ValueError(
                f"Claim support was not reviewed for {result['id']}."
            )

        if review.get("all_citations_support_claims") is None:
            raise ValueError(
                f"Citation support was not reviewed for {result['id']}."
            )

    supported_facts = sum(
        fact["present_and_supported"]
        for fact in required_fact_reviews
    )

    complete_answers = sum(
        all(
            fact["present_and_supported"]
            for fact in result["manual_review"]["required_facts"]
        )
        for result in answerable
    )

    known_incorrect_claims = sum(
        claim["present"]
        for claim in incorrect_claim_reviews
    )

    correct_answerable_responses = sum(
        not result["abstained"]
        for result in answerable
    )

    correct_unanswerable_abstentions = sum(
        result["abstained"]
        for result in unanswerable
    )

    faithful_substantive_answers = sum(
        result["manual_review"]["all_claims_supported"]
        for result in substantive
    )

    citation_correct_substantive_answers = sum(
        result["manual_review"]["all_citations_support_claims"]
        for result in substantive
    )

    source_recalls = [
        result["expected_source_recall"]
        for result in answerable
        if result["expected_source_recall"] is not None
    ]

    metrics = {
        "questions": len(results),
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "required_facts": len(required_fact_reviews),
        "supported_required_facts": supported_facts,
        "gold_fact_coverage": (
            supported_facts / len(required_fact_reviews)
        ),
        "complete_answers": complete_answers,
        "complete_answer_rate": complete_answers / len(answerable),
        "answerable_response_rate": (
            correct_answerable_responses / len(answerable)
        ),
        "unanswerable_abstention_accuracy": (
            correct_unanswerable_abstentions / len(unanswerable)
        ),
        "faithful_substantive_answers": faithful_substantive_answers,
        "faithful_substantive_answer_rate": (
            faithful_substantive_answers / len(substantive)
        ),
        "citation_correct_substantive_answers": (
            citation_correct_substantive_answers
        ),
        "citation_correct_substantive_answer_rate": (
            citation_correct_substantive_answers / len(substantive)
        ),
        "known_incorrect_claims_present": known_incorrect_claims,
        "known_incorrect_claims_checked": len(incorrect_claim_reviews),
        "known_error_trigger_rate": (
            known_incorrect_claims / len(incorrect_claim_reviews)
        ),
        "mean_expected_source_recall": (
            sum(source_recalls) / len(source_recalls)
        ),
    }

    OUTPUT_PATH.write_text(json.dumps(metrics, indent=2))

    print("\nAnswer evaluation results:\n")
    print(f"Questions: {metrics['questions']}")
    print(
        "Gold fact coverage: "
        f"{supported_facts}/{len(required_fact_reviews)} "
        f"({percentage(metrics['gold_fact_coverage'])})"
    )
    print(
        "Complete answer rate: "
        f"{complete_answers}/{len(answerable)} "
        f"({percentage(metrics['complete_answer_rate'])})"
    )
    print(
        "Answerable response rate: "
        f"{correct_answerable_responses}/{len(answerable)} "
        f"({percentage(metrics['answerable_response_rate'])})"
    )
    print(
        "Unanswerable abstention accuracy: "
        f"{correct_unanswerable_abstentions}/{len(unanswerable)} "
        f"({percentage(metrics['unanswerable_abstention_accuracy'])})"
    )
    print(
        "Faithful substantive answer rate: "
        f"{faithful_substantive_answers}/{len(substantive)} "
        f"({percentage(metrics['faithful_substantive_answer_rate'])})"
    )
    print(
        "Citation-correct substantive answer rate: "
        f"{citation_correct_substantive_answers}/{len(substantive)} "
        f"({percentage(metrics['citation_correct_substantive_answer_rate'])})"
    )
    print(
        "Known error trigger rate: "
        f"{known_incorrect_claims}/{len(incorrect_claim_reviews)} "
        f"({percentage(metrics['known_error_trigger_rate'])})"
    )
    print(
        "Mean expected-source recall: "
        f"{percentage(metrics['mean_expected_source_recall'])}"
    )
    print(f"\nSaved metrics to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()