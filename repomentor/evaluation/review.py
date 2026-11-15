"""
Interactive human review of generated answers.

    python -m repomentor eval review evaluation/runs/answers.json
"""

import json


def ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(f"{prompt} [y/n]: ").strip().lower()

        if answer in {"y", "yes"}:
            return True

        if answer in {"n", "no"}:
            return False

        print("Please enter y or n.")


def command_review(arguments) -> None:
    path = arguments.answers
    results = json.loads(path.read_text())

    for position, result in enumerate(results, start=1):
        review = result["manual_review"]

        if review.get("all_citations_support_claims") is not None and not arguments.again:
            continue

        print("\n" + "=" * 80)
        print(f"QUESTION {position}/{len(results)}")
        print(result["question"])
        print(f"Expected to be answerable: {result['answerable']}")
        print(f"Automatically detected abstention: {result['abstained']}")
        print("\nGENERATED ANSWER:\n")
        print(result["answer"])

        print("\nREQUIRED FACTS")

        for number, fact in enumerate(review["required_facts"], start=1):
            print(f"\n{number}. {fact['fact']}")
            print(f"   Expected source: {fact['expected_source']}")
            fact["present_and_supported"] = ask_yes_no(
                "   Is this fact stated accurately and supported?"
            )

        print("\nKNOWN INCORRECT CLAIMS")

        for number, claim in enumerate(review["incorrect_claims"], start=1):
            print(f"\n{number}. {claim['claim']}")
            claim["present"] = ask_yes_no("   Does the generated answer make this mistake?")

        review["all_claims_supported"] = ask_yes_no(
            "\nAre all other factual claims supported by repository evidence?"
        )
        review["all_citations_support_claims"] = ask_yes_no(
            "Do all citations support the claims attached to them?"
        )
        review["notes"] = input("Optional notes about this answer: ").strip()

        path.write_text(json.dumps(results, indent=2))
        print("Review saved.")

    print(f"\nCompleted review: {path}")
