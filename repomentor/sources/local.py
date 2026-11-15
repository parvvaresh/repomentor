"""
A Git checkout on the local disk.

Useful for offline work, for evaluation (benchmarks pin an exact commit of
a public repository) and for repositories on hosts RepoMentor has no
client for. Source links point at the checkout's GitHub or GitLab remote
when it has one.
"""

import re
import subprocess
from pathlib import Path

from .base import (
    FileEntry,
    NotFound,
    NotSupported,
    RepositorySource,
    SourceError,
    is_useful_file,
)


SEPARATOR = "\x1f"
RECORD_END = "\x1e"
LOG_FORMAT = SEPARATOR.join(["%H", "%s", "%B", "%an", "%ae", "%cI"]) + RECORD_END

REMOTE_PATTERN = re.compile(
    r"^(?:https?://(?:[^@/]+@)?|git@)(?P<host>[^/:]+)[/:](?P<path>.+?)(?:\.git)?/?$"
)


def remote_web_url(remote: str) -> str | None:
    """Turn a clone URL such as git@github.com:owner/repo.git into a web URL."""

    match = REMOTE_PATTERN.match(remote.strip())

    if not match:
        return None

    return f"https://{match['host']}/{match['path']}"


class LocalGitSource(RepositorySource):
    provider = "local"

    def _git(self, project: str, *arguments: str) -> str:
        directory = Path(project)

        if not (directory / ".git").exists():
            raise NotFound(f"{project} is not a Git checkout.")

        result = subprocess.run(
            ["git", "-C", str(directory), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            raise SourceError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")

        return result.stdout

    def _remote(self, project: str) -> str | None:
        try:
            return remote_web_url(self._git(project, "remote", "get-url", "origin"))
        except SourceError:
            return None

    def web_url(self, project: str) -> str:
        return self._remote(project) or Path(project).resolve().as_uri()

    def head_commit(self, project: str) -> str:
        return self._git(project, "rev-parse", "HEAD").strip()

    def list_files(self, project: str, ref: str) -> list[FileEntry]:
        entries = []

        for line in self._git(project, "ls-tree", "-r", "-z", ref).split("\0"):
            if not line:
                continue

            meta, path = line.split("\t", 1)
            _, kind, blob_id = meta.split()

            if kind == "blob" and is_useful_file(path):
                entries.append(FileEntry(path=path, blob_id=blob_id))

        return entries

    def read_file(self, project: str, file_path: str, ref: str) -> dict:
        blob_id = self._git(project, "rev-parse", f"{ref}:{file_path}").strip()
        content = self._git(project, "cat-file", "blob", blob_id)

        return {
            "content": content,
            "blob_id": blob_id,
            "last_commit_id": ref,
            "size": len(content.encode()),
        }

    def _parse_log(self, project: str, output: str) -> list[dict]:
        remote = self._remote(project)
        commits = []

        for record in output.split(RECORD_END):
            record = record.strip("\n")

            if not record:
                continue

            sha, title, message, name, email, date = record.split(SEPARATOR)
            url = ""

            if remote:
                url = f"{remote}/-/commit/{sha}" if "gitlab" in remote else f"{remote}/commit/{sha}"

            commits.append({
                "sha": sha,
                "title": title,
                "message": message.strip(),
                "author_name": name,
                "author_email": email,
                "date": date,
                "url": url,
            })

        return commits

    def list_commits(self, project: str, limit: int) -> list[dict]:
        return self._parse_log(
            project,
            self._git(project, "log", f"-n{limit}", f"--format={LOG_FORMAT}"),
        )

    def file_history(self, project: str, file_path: str, limit: int) -> list[dict]:
        return self._parse_log(
            project,
            self._git(
                project, "log", f"-n{limit}", f"--format={LOG_FORMAT}", "--", file_path
            ),
        )

    def list_merge_requests(self, project: str, limit: int) -> list[dict]:
        # Merge requests live on the host, not in Git; merge commits are
        # indexed through list_commits instead.
        return []

    def get_merge_request(self, project: str, number: int) -> dict:
        raise NotSupported(
            "Local checkouts have no merge requests. Paste the diff instead."
        )
