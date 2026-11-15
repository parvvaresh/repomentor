"""
Who-to-ask agent: finds the people who know a part of the codebase.

It retrieves the files relevant to the question, reads the commit history
of each of those files from the repository host, and ranks authors by how
much, and how recently, they changed them. No language model is needed,
so the answer is fast and fully traceable to commits.
"""

import math
from collections import defaultdict
from datetime import datetime, timezone

from ..sources import SourceError
from .base import Agent, AgentContext, retrieve_sources


COMMITS_PER_FILE = 50
# A commit loses half its weight every this many days
HALF_LIFE_DAYS = 180
MAX_PEOPLE = 5

BOT_MARKERS = ("[bot]", "dependabot", "renovate", "github-actions", "pre-commit-ci")


def parse_date(value: str) -> datetime | None:
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None

    return date if date.tzinfo else date.replace(tzinfo=timezone.utc)


def recency_weight(date: datetime | None, now: datetime) -> float:
    if date is None:
        return 0.5

    age_days = max((now - date).total_seconds() / 86400, 0)
    return math.pow(0.5, age_days / HALF_LIFE_DAYS)


def is_bot(name: str, email: str) -> bool:
    text = f"{name} {email}".lower()
    return any(marker in text for marker in BOT_MARKERS)


def rank_people(histories: dict[str, list[dict]], now: datetime | None = None) -> list[dict]:
    """
    histories: {file_path: [commit, ...]} -> people sorted by score:
    [{"name", "email", "score", "commits", "last_commit", "files": [...],
      "recent_commits": [{"sha", "title", "url", "date", "file"}]}]
    """

    now = now or datetime.now(timezone.utc)
    people: dict[str, dict] = {}
    counted = defaultdict(set)

    for file_path, commits in histories.items():
        for commit in commits:
            name, email = commit["author_name"], commit["author_email"]

            if is_bot(name, email):
                continue

            key = (email or name).lower()
            person = people.setdefault(key, {
                "name": name,
                "email": email,
                "score": 0.0,
                "commits": 0,
                "last_commit": "",
                "files": {},
                "recent_commits": [],
            })
            date = parse_date(commit["date"])
            person["score"] += recency_weight(date, now)
            person["files"][file_path] = person["files"].get(file_path, 0) + 1

            if commit["sha"] not in counted[key]:
                counted[key].add(commit["sha"])
                person["commits"] += 1
                person["recent_commits"].append({
                    "sha": commit["sha"],
                    "title": commit["title"],
                    "url": commit["url"],
                    "date": commit["date"],
                    "file": file_path,
                })

            if commit["date"] > person["last_commit"]:
                person["last_commit"] = commit["date"]

    ranked = sorted(people.values(), key=lambda person: person["score"], reverse=True)

    for person in ranked:
        person["score"] = round(person["score"], 2)
        person["files"] = [
            {"path": path, "commits": count}
            for path, count in sorted(person["files"].items(), key=lambda item: -item[1])
        ]
        person["recent_commits"] = sorted(
            person["recent_commits"], key=lambda commit: commit["date"], reverse=True
        )[:3]

    return ranked


class ExpertsAgent(Agent):
    id = "experts"
    name = "Who to ask"
    description = (
        "Finds the people who have worked most, and most recently, on the "
        "code behind your question, from commit history."
    )
    examples = [
        "Who knows the authentication code?",
        "Who should review changes to the database models?",
        "Who worked on the deployment configuration?",
    ]
    working_label = "Reading commit history…"

    def stream(self, question: str, context: AgentContext):
        if context.source is None:
            raise ValueError("Commit history is not available for this repository.")

        yield {"type": "status", "label": "Finding the relevant files…"}
        chunks, sources, identifiers = retrieve_sources(question, context)
        code = [
            (chunk, source) for chunk, source in zip(chunks, sources)
            if chunk.get("kind", "code") == "code"
        ]
        yield {"type": "sources", "sources": [source for _, source in code]}

        histories = {}
        failures = []
        project = context.source_project or context.project_path

        for position, (chunk, _) in enumerate(code, start=1):
            yield {
                "type": "status",
                "label": f"Reading history of {chunk['file_path']} ({position}/{len(code)})…",
            }

            try:
                histories[chunk["file_path"]] = context.source.file_history(
                    project, chunk["file_path"], COMMITS_PER_FILE
                )
            except SourceError as error:
                failures.append(f"{chunk['file_path']}: {error}")

        if not histories and failures:
            raise SourceError(failures[0])

        people = rank_people(histories)[:MAX_PEOPLE]

        yield {"type": "result", "result": {
            "people": people,
            "sources": [source for _, source in code],
            "related_identifiers": identifiers,
            "files_checked": len(histories),
            "failures": failures,
            "answer": self._summary(people),
        }}

    def _summary(self, people: list[dict]) -> str:
        if not people:
            return "No commit authors were found for the relevant files."

        lines = []

        for person in people:
            files = ", ".join(item["path"] for item in person["files"][:3])
            lines.append(
                f"- {person['name']} ({person['commits']} commits, last "
                f"{person['last_commit'][:10]}): {files}"
            )

        return "\n".join(lines)

    def format_text(self, result: dict) -> str:
        return result["answer"]
