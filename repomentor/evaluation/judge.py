"""
A language-model judge for answer quality, calibrated against people.

Manual review does not scale to large benchmarks or to every pull
request. The judge answers the same yes/no questions a human reviewer
answers (is this required fact stated and supported? does the answer make
this known mistake? are all claims supported by the cited sources?) with
one small, focused prompt per question, so its labels can be compared
item by item with the human labels.

    # label answers with the judge (adds `judge_review`)
    python -m repomentor eval judge evaluation/runs/answers.json

    # measure agreement with the human labels already in a file
    python -m repomentor eval judge evaluation/generated_answers.json --calibrate

Calibration reports raw agreement and Cohen's kappa per label type and
writes evaluation/results/judge_calibration.json. Only trust judge-based
metrics for label types whose agreement is acceptable.
"""

import json
import re
from pathlib import Path

from .. import config, llm, safety
from .answers import empty_review


CALIBRATION_PATH = Path("evaluation/results/judge_calibration.json")

JUDGE_PROMPT = """
You are a strict evaluator of answers about a code repository. Reply with
YES or NO on the first line, then one short sentence of justification.
{policy}"""

FACT_QUESTION = """Does the ANSWER state the following fact accurately (the same meaning,
possibly in other words)? Partial or vaguer statements count as NO.

FACT: {fact}"""

CLAIM_QUESTION = """Does the ANSWER make the following claim (or something that implies it)?

CLAIM: {claim}"""

SUPPORT_QUESTION = """Is every factual claim in the ANSWER directly supported by the SOURCES?
Claims about code that the sources do not show count as NO."""

CITATION_QUESTION = """Each bullet in the ANSWER ends with a citation [file:lines]. Does every
citation point to a source whose content supports the claim it is attached to?"""


def parse_verdict(text: str) -> bool | None:
    match = re.search(r"\b(YES|NO)\b", text.upper())
    return None if match is None else match.group(1) == "YES"


def ask(question: str, answer: str, sources: list[dict] | None, model: str) -> bool:
    tag = safety.new_tag()
    content = safety.wrap(answer, tag, kind="answer")

    if sources is not None:
        content += "\n\n" + "\n\n".join(
            safety.wrap(source["content"], tag, citation=source["citation"]) for source in sources
        )

    reply = llm.chat(
        [
            {"role": "system", "content": JUDGE_PROMPT.format(policy=safety.policy(tag))},
            {"role": "user", "content": f"{content}\n\n{question}"},
        ],
        model=model,
    )
    verdict = parse_verdict(reply)

    # An unparseable reply is treated as the cautious answer for each kind
    return bool(verdict)


def judge_result(result: dict, model: str) -> dict:
    review = empty_review({
        "required_facts": [
            {"fact": fact["fact"], "source": fact["expected_source"]}
            for fact in result["manual_review"]["required_facts"]
        ],
        "incorrect_claims": [claim["claim"] for claim in result["manual_review"]["incorrect_claims"]],
    })
    answer = result["answer"]

    if result["abstained"]:
        # An abstention states no facts and makes no claims
        for fact in review["required_facts"]:
            fact["present_and_supported"] = False

        for claim in review["incorrect_claims"]:
            claim["present"] = False

        review["all_claims_supported"] = True
        review["all_citations_support_claims"] = True
        return review

    sources = result.get("sources")

    for fact in review["required_facts"]:
        fact["present_and_supported"] = ask(FACT_QUESTION.format(fact=fact["fact"]), answer, None, model)

    for claim in review["incorrect_claims"]:
        claim["present"] = ask(CLAIM_QUESTION.format(claim=claim["claim"]), answer, None, model)

    if sources:
        review["all_claims_supported"] = ask(SUPPORT_QUESTION, answer, sources, model)
        review["all_citations_support_claims"] = ask(CITATION_QUESTION, answer, sources, model)
    else:
        # Older runs did not store source text; fall back to the human label
        review["all_claims_supported"] = result["manual_review"].get("all_claims_supported")
        review["all_citations_support_claims"] = result["manual_review"].get(
            "all_citations_support_claims"
        )

    review["judge_model"] = model
    return review


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    if not pairs:
        return None

    count = len(pairs)
    observed = sum(a == b for a, b in pairs) / count
    first_yes = sum(a for a, _ in pairs) / count
    second_yes = sum(b for _, b in pairs) / count
    expected = first_yes * second_yes + (1 - first_yes) * (1 - second_yes)

    if expected == 1:
        return 1.0 if observed == 1 else 0.0

    return (observed - expected) / (1 - expected)


def label_pairs(results: list[dict]) -> dict[str, list[tuple[bool, bool]]]:
    """(human, judge) label pairs per label type."""

    pairs = {"required_fact": [], "incorrect_claim": [], "all_claims_supported": [],
             "all_citations_support_claims": []}

    for result in results:
        human, judge = result["manual_review"], result.get("judge_review")

        if not judge:
            continue

        for h, j in zip(human["required_facts"], judge["required_facts"]):
            if h["present_and_supported"] is not None:
                pairs["required_fact"].append((h["present_and_supported"], j["present_and_supported"]))

        for h, j in zip(human["incorrect_claims"], judge["incorrect_claims"]):
            if h["present"] is not None:
                pairs["incorrect_claim"].append((h["present"], j["present"]))

        if result.get("sources"):
            for key in ("all_claims_supported", "all_citations_support_claims"):
                if human.get(key) is not None and judge.get(key) is not None:
                    pairs[key].append((human[key], judge[key]))

    return pairs


def calibration_report(results: list[dict], model: str) -> dict:
    report = {"judge_model": model, "labels": {}}

    for name, pairs in label_pairs(results).items():
        if not pairs:
            continue

        report["labels"][name] = {
            "items": len(pairs),
            "agreement": sum(a == b for a, b in pairs) / len(pairs),
            "cohen_kappa": cohen_kappa(pairs),
            "human_yes": sum(a for a, _ in pairs),
            "judge_yes": sum(b for _, b in pairs),
            "false_positive": sum((not a) and b for a, b in pairs),
            "false_negative": sum(a and not b for a, b in pairs),
        }

    return report


def command_judge(arguments) -> None:
    path = arguments.answers
    results = json.loads(path.read_text())
    model = arguments.model or config.JUDGE_MODEL

    for position, result in enumerate(results, start=1):
        print(f"[{position}/{len(results)}] judging {result['id']}")
        result["judge_review"] = judge_result(result, model)

    if not arguments.calibrate:
        path.write_text(json.dumps(results, indent=2))
        print(f"\nAdded judge_review to {path}")
        return

    report = calibration_report(results, model)
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(report, indent=2))

    print(f"\nJudge calibration ({model}) against human labels:\n")
    print(f"{'Label':<32}{'Items':>6}{'Agree':>8}{'Kappa':>8}{'FP':>5}{'FN':>5}")

    for name, row in report["labels"].items():
        kappa = "n/a" if row["cohen_kappa"] is None else f"{row['cohen_kappa']:.2f}"
        print(f"{name:<32}{row['items']:>6}{row['agreement']:>8.2f}{kappa:>8}"
              f"{row['false_positive']:>5}{row['false_negative']:>5}")

    print(f"\nSaved to {CALIBRATION_PATH}")
