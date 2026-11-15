"""
Generate answers for the answer-quality gold set.

    python -m repomentor eval answers --data data --output evaluation/runs/answers.json

The output has the same shape as evaluation/generated_answers.json: each
answer with automatic citation and abstention fields, plus an empty
`manual_review` block to fill in with `eval review` (a person) or
`eval judge` (a model). The human-reviewed baseline file is never
overwritten unless it is passed explicitly as --output.
"""

import json
import re
from pathlib import Path

from ..answer import NO_EVIDENCE, format_answer, generate_answer
from ..index import load_index


DATASET_PATH = Path("evaluation/answer_questions.json")
BASELINE_PATH = Path("evaluation/generated_answers.json")

CITATION_PATTERN = re.compile(r"\[([^\[\]\n]+):\d+-\d+\]")


def extract_cited_sources(answer: str) -> list[str]:
    """Unique file paths from citations such as [file.py:1-36]."""
    return sorted(set(CITATION_PATTERN.findall(answer)))


def expected_sources(case: dict) -> list[str]:
    return sorted({fact["source"] for fact in case["required_facts"]})


def source_recall(expected: list[str], cited: list[str]) -> float | None:
    if not expected:
        return None

    return len(set(expected) & set(cited)) / len(expected)


def empty_review(case: dict) -> dict:
    return {
        "required_facts": [
            {"fact": item["fact"], "expected_source": item["source"], "present_and_supported": None}
            for item in case["required_facts"]
        ],
        "incorrect_claims": [
            {"claim": claim, "present": None} for claim in case["incorrect_claims"]
        ],
        "other_unsupported_claims": [],
        "all_claims_supported": None,
        "all_citations_support_claims": None,
        "notes": "",
    }


def generate(cases: list[dict], index, top_k: int) -> list[dict]:
    results = []

    for position, case in enumerate(cases, start=1):
        print(f"[{position}/{len(cases)}] {case['question']}")
        result = generate_answer(case["question"], index, top_k=top_k)
        answer = format_answer(result)
        cited = extract_cited_sources(answer)
        gold = expected_sources(case)

        results.append({
            "id": case["id"],
            "question": case["question"],
            "answerable": case["answerable"],
            "answer": answer,
            "sources": [
                {"citation": s["citation"], "content": s["content"]} for s in result["sources"]
            ],
            "retrieval_method": result.get("retrieval_method"),
            "refusal": result.get("refusal"),
            "expected_sources": gold,
            "cited_sources": cited,
            "expected_source_recall": source_recall(gold, cited),
            "abstained": answer == NO_EVIDENCE,
            "manual_review": empty_review(case),
        })

    return results


def command_answers(arguments) -> None:
    if arguments.output.resolve() == BASELINE_PATH.resolve() and not arguments.overwrite_baseline:
        raise SystemExit(
            f"{BASELINE_PATH} holds the human-reviewed baseline. "
            "Choose another --output, or pass --overwrite-baseline."
        )

    cases = json.loads(arguments.dataset.read_text())
    index = load_index(arguments.data)
    results = generate(cases, index, arguments.top_k)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2))
    print(f"\nSaved generated answers to {arguments.output}")
    print("Review them with `eval review` (a person) or `eval judge` (a model).")
