import json
from pathlib import Path


RESULTS_PATH = Path("evaluation/generated_answers.json")


def ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} [y/n]: ").strip().lower()

        if answer in {"y", "yes"}:
            return True

        if answer in {"n", "no"}:
            return False

        print("Please enter y or n.")


def save_results(results: list[dict]) -> None:
    RESULTS_PATH.write_text(json.dumps(results, indent=2))


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text())

    for position, result in enumerate(results, start=1):
        review = result["manual_review"]

        print("\n" + "=" * 80)
        print(f"QUESTION {position}/{len(results)}")
        print(result["question"])
        print(f"Expected to be answerable: {result['answerable']}")
        print(f"Automatically detected abstention: {result['abstained']}")
        print("\nGENERATED ANSWER:\n")
        print(result["answer"])

        print("\nREQUIRED FACTS")

        for fact_number, fact_review in enumerate(
            review["required_facts"],
            start=1,
        ):
            print(f"\n{fact_number}. {fact_review['fact']}")
            print(f"   Expected source: {fact_review['expected_source']}")

            fact_review["present_and_supported"] = ask_yes_no(
                "   Is this fact stated accurately and supported?"
            )

        print("\nKNOWN INCORRECT CLAIMS")

        for claim_number, claim_review in enumerate(
            review["incorrect_claims"],
            start=1,
        ):
            print(f"\n{claim_number}. {claim_review['claim']}")

            claim_review["present"] = ask_yes_no(
                "   Does the generated answer make this mistake?"
            )

        review["all_claims_supported"] = ask_yes_no(
            "\nAre all other factual claims supported by repository evidence?"
        )

        review["all_citations_support_claims"] = ask_yes_no(
            "Do all citations support the claims attached to them?"
        )

        review["notes"] = input(
            "Optional notes about this answer: "
        ).strip()

        save_results(results)
        print("Review saved.")

    print(f"\nCompleted manual review: {RESULTS_PATH}")


if __name__ == "__main__":
    main()