"""
Download a repository into a JSON Lines corpus.

Ingestion is incremental: when a previous corpus is supplied, files whose
blob ID has not changed are copied from it instead of being downloaded
again, so synchronizing after a push only fetches the files that changed.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .sources import NotSupported, RepositorySource, SourceError
from .sources.base import MAX_FILE_SIZE


Progress = Callable[[int, int, str, bool], None]


@dataclass
class IngestReport:
    head_commit: str
    downloaded: int = 0
    reused: int = 0
    failed: int = 0
    skipped: int = 0
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return self.downloaded + self.reused

    def to_dict(self) -> dict:
        return {**asdict(self), "file_count": self.file_count}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, records) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")
            count += 1

    return count


def ingest_repository(
    source: RepositorySource,
    project: str,
    output_path: Path,
    previous_path: Path | None = None,
    progress: Progress | None = None,
) -> IngestReport:
    """
    Write every useful file of `project` to `output_path`.

    `progress(position, total, file_path, succeeded)` is called after each
    file that had to be downloaded.
    """

    head = source.head_commit(project)
    entries = source.list_files(project, head)
    web_url = source.web_url(project)

    previous = {
        record["file_path"]: record
        for record in (read_jsonl(previous_path) if previous_path else [])
    }

    report = IngestReport(head_commit=head)
    current_paths = {entry.path for entry in entries}
    report.removed = sorted(set(previous) - current_paths)

    to_download = [
        entry for entry in entries
        if previous.get(entry.path, {}).get("blob_id") != entry.blob_id
    ]

    def records():
        position = 0

        for entry in entries:
            old = previous.get(entry.path)

            if old and old.get("blob_id") == entry.blob_id:
                report.reused += 1
                yield {**old, "web_url": web_url, "provider": source.provider}
                continue

            position += 1

            try:
                data = source.read_file(project, entry.path, head)
            except SourceError:
                report.failed += 1
                succeeded = False
            else:
                succeeded = True

                if data["size"] > MAX_FILE_SIZE:
                    report.skipped += 1
                else:
                    report.downloaded += 1
                    report.changed.append(entry.path)
                    yield {
                        "provider": source.provider,
                        "project_path": project,
                        "web_url": web_url,
                        "file_path": entry.path,
                        "blob_id": data["blob_id"],
                        "last_commit_id": data["last_commit_id"],
                        "size": data["size"],
                        "content": data["content"],
                    }

            if progress:
                progress(position, len(to_download), entry.path, succeeded)

    write_jsonl(output_path, records())
    return report


def ingest_history(
    source: RepositorySource,
    project: str,
    output_path: Path,
    commit_limit: int,
    merge_request_limit: int,
) -> int:
    """
    Save recent commit messages and merged merge-request descriptions so
    questions about *why* something changed can be answered. Hosts that do
    not offer one of them are skipped. Returns the number of records.
    """

    records = []
    web_url = source.web_url(project)
    base = {"provider": source.provider, "project_path": project, "web_url": web_url}

    try:
        for commit in source.list_commits(project, commit_limit):
            # Commits with a one-line message say little about intent, but
            # they still help "when did X change" questions.
            records.append({
                **base,
                "kind": "commit",
                "id": commit["sha"],
                "title": commit["title"],
                "body": commit["message"],
                "author": commit["author_name"],
                "date": commit["date"],
                "url": commit["url"],
            })
    except NotSupported:
        pass

    try:
        for request in source.list_merge_requests(project, merge_request_limit):
            records.append({
                **base,
                "kind": "merge_request",
                "id": str(request["number"]),
                "title": request["title"],
                "body": request["description"],
                "author": request["author"],
                "date": request["merged_at"],
                "url": request["url"],
            })
    except NotSupported:
        pass

    return write_jsonl(output_path, records)
