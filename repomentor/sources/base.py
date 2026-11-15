"""
The interface every repository host implements.

Ingestion, the who-to-ask agent, the reviewer and merge-request creation
only talk to a `RepositorySource`, so GitLab, GitHub and local Git
checkouts are interchangeable.
"""

from dataclasses import dataclass
from pathlib import PurePosixPath


ALLOWED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".md", ".yml", ".yaml", ".toml", ".sql",
}

EXCLUDED_DIRECTORIES = {
    "node_modules", ".venv", "venv", "dist",
    "build", "coverage", "vendor", ".git",
}

# Files larger than this are almost always generated or data files
MAX_FILE_SIZE = 500_000


def is_useful_file(file_path: str) -> bool:
    path = PurePosixPath(file_path)

    if any(part in EXCLUDED_DIRECTORIES for part in path.parts):
        return False

    return path.name == "Dockerfile" or path.suffix.lower() in ALLOWED_EXTENSIONS


class SourceError(RuntimeError):
    """A readable error from a repository host."""


class AccessDenied(SourceError):
    pass


class NotFound(SourceError):
    pass


class NotSupported(SourceError):
    pass


@dataclass
class FileEntry:
    path: str
    # Content hash reported by the host (Git blob SHA); unchanged files
    # keep their blob ID, which makes incremental synchronization cheap.
    blob_id: str


@dataclass
class UserInfo:
    id: str
    username: str
    name: str


class RepositorySource:
    """
    Base class. `project` is the host's project identifier, e.g.
    "group/subgroup/project" on GitLab or "owner/repo" on GitHub.
    """

    provider: str = ""

    def web_url(self, project: str) -> str:
        """Browser URL of the project."""
        raise NotImplementedError

    def head_commit(self, project: str) -> str:
        """Commit SHA of the default branch."""
        raise NotImplementedError

    def list_files(self, project: str, ref: str) -> list[FileEntry]:
        """Every useful file at commit `ref`."""
        raise NotImplementedError

    def read_file(self, project: str, file_path: str, ref: str) -> dict:
        """
        Return {"content": str, "blob_id": str, "last_commit_id": str,
        "size": int}. `last_commit_id` may equal `ref` when the host
        cannot tell cheaply which commit last touched the file.
        """
        raise NotImplementedError

    # --- History ------------------------------------------------------------

    def list_commits(self, project: str, limit: int) -> list[dict]:
        """
        Recent commits on the default branch, newest first:
        [{"sha", "title", "message", "author_name", "author_email",
          "date", "url"}]
        """
        raise NotSupported(f"{self.provider} does not list commits.")

    def list_merge_requests(self, project: str, limit: int) -> list[dict]:
        """
        Recently merged merge/pull requests, newest first:
        [{"number", "title", "description", "author", "merged_at", "url"}]
        """
        raise NotSupported(f"{self.provider} does not list merge requests.")

    def file_history(self, project: str, file_path: str, limit: int) -> list[dict]:
        """Commits that touched a file, newest first (same shape as list_commits)."""
        raise NotSupported(f"{self.provider} does not report file history.")

    # --- Reviews and changes -----------------------------------------------

    def get_merge_request(self, project: str, number: int) -> dict:
        """
        {"number", "title", "description", "author", "url",
         "source_branch", "target_branch",
         "changes": [{"path", "old_path", "diff", "new_file", "deleted_file"}]}
        """
        raise NotSupported(f"{self.provider} does not support reviews.")

    def create_merge_request(
        self,
        project: str,
        branch: str,
        title: str,
        description: str,
        files: dict[str, str],
    ) -> dict:
        """
        Commit `files` ({path: new content}) to a new branch created from the
        default branch and open a merge request. Returns {"url", "number"}.
        """
        raise NotSupported(f"{self.provider} cannot open merge requests.")

    # --- Users ----------------------------------------------------------------

    def current_user(self) -> UserInfo:
        """The user the source's token belongs to."""
        raise NotSupported(f"{self.provider} has no users.")

    def can_read(self, project: str) -> bool:
        """Whether the source's token may read the project."""
        raise NotSupported(f"{self.provider} has no access control.")
