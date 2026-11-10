import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path

from rag_answer import answer_question, load_index


DATASET_PATH = Path("evaluation/answer_questions.json")
OUTPUT_PATH = Path("evaluation/generated_answers.json")
TOP_K = 5

CITATION_PATTERN = re.compile(r"\[([^\[\]\n]+):\d+-\d+\]")


def extract_answer(full_output: str) -> str:
    """Remove identifiers and the final source list from the captured output."""
    if "\nAnswer:\n" not in full_output:
        return full_output.strip()

    answer = full_output.split("\nAnswer:\n", maxsplit=1)[1]

    if "\nSources:" in answer:
        answer = answer.split("\nSources:", maxsplit=1)[0]

    return answer.strip()


def extract_cited_sources(answer: str) -> list[str]:
    """Extract unique file paths from citations such as [file.py:1-36]."""
    return sorted(set(CITATION_PATTERN.findall(answer)))


def expected_sources(case: dict) -> list[str]:
    """Get the unique gold source files for an answerable question."""
    return sorted(
        {
            required_fact["source"]
            for required_fact in case["required_facts"]
        }
    )


def calculate_source_recall(
    expected: list[str],
    cited: list[str],
) -> float | None:
    """Measure how many expected source files appeared in the citations."""
    if not expected:
        return None

    return len(set(expected) & set(cited)) / len(expected)


def generate_answer(question: str, index: dict) -> tuple[str, str]:
    """Run RepoMentor and capture its printed output."""
    output_buffer = io.StringIO()

    with redirect_stdout(output_buffer):
        answer_question(
            question=question,
            index=index,
            top_k=TOP_K,
            retrieval_only=False,
        )

    full_output = output_buffer.getvalue()
    answer = extract_answer(full_output)

    return answer, full_output


def main() -> None:
    cases = json.loads(DATASET_PATH.read_text())
    index = load_index()
    results = []

    print(f"Generating answers for {len(cases)} evaluation questions...\n")

    for position, case in enumerate(cases, start=1):
        print(f"[{position}/{len(cases)}] {case['question']}")

        answer, full_output = generate_answer(case["question"], index)
        cited_sources = extract_cited_sources(answer)
        gold_sources = expected_sources(case)

        abstained = "couldn't find enough evidence" in answer.lower()

        results.append(
            {
                "id": case["id"],
                "question": case["question"],
                "answerable": case["answerable"],
                "answer": answer,
                "full_output": full_output,
                "expected_sources": gold_sources,
                "cited_sources": cited_sources,
                "expected_source_recall": calculate_source_recall(
                    gold_sources,
                    cited_sources,
                ),
                "abstained": abstained,
                "manual_review": {
                    "required_facts": [
                        {
                            "fact": item["fact"],
                            "expected_source": item["source"],
                            "present_and_supported": None,
                        }
                        for item in case["required_facts"]
                    ],
                    "incorrect_claims": [
                        {
                            "claim": claim,
                            "present": None,
                        }
                        for claim in case["incorrect_claims"]
                    ],
                    "other_unsupported_claims": [],
                    "all_claims_supported": None,
                    "notes": "",
                },
            }
        )

    OUTPUT_PATH.write_text(json.dumps(results, indent=2))

    print(f"\nSaved generated answers to: {OUTPUT_PATH}")
    print(
        "The automatic fields only check citations and abstention. "
        "Correctness still requires manual review."
    )


if __name__ == "__main__":
    main()