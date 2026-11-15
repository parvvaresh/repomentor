"""
Repository hosts RepoMentor can index.

`get_source(provider)` returns a client authenticated with the server's
configured token; pass `token` to act as a specific user instead.
"""

import re
from urllib.parse import urlparse

from .. import config
from .base import (
    AccessDenied,
    FileEntry,
    NotFound,
    NotSupported,
    RepositorySource,
    SourceError,
    UserInfo,
    is_useful_file,
)
from .github import GitHubSource
from .gitlab import GitLabSource
from .local import LocalGitSource


PROVIDERS = ("gitlab", "github", "local")

PROJECT_PATH_PATTERN = re.compile(r"^[\w.-]+(?:/[\w.-]+)+$")


def get_source(provider: str, token: str | None = None) -> RepositorySource:
    if provider == "gitlab":
        return GitLabSource(config.GITLAB_BASE_URL, token or config.GITLAB_TOKEN)

    if provider == "github":
        return GitHubSource(
            config.GITHUB_API_URL,
            config.GITHUB_WEB_URL,
            token or config.GITHUB_TOKEN,
        )

    if provider == "local":
        return LocalGitSource()

    raise ValueError(f"Unknown repository provider '{provider}'.")


def write_token(provider: str) -> str:
    """Token used to open merge requests when the user has not signed in."""

    return {
        "gitlab": config.GITLAB_WRITE_TOKEN,
        "github": config.GITHUB_WRITE_TOKEN,
    }.get(provider, "")


def configured_providers() -> list[str]:
    """Hosts the server can index. GitHub works without a token for public repos."""

    providers = []

    if config.GITLAB_TOKEN and config.GITLAB_BASE_URL:
        providers.append("gitlab")

    providers.append("github")
    return providers


def normalize_project_path(value: str, provider: str | None = None) -> tuple[str, str]:
    """
    Accept "group/project" as well as a pasted URL such as
    "https://gitlab.com/group/project.git", ".../-/tree/main" or
    "https://github.com/owner/repo/pull/3". Returns (provider, path).
    """

    value = value.strip()

    if "://" in value or value.startswith("git@"):
        if value.startswith("git@"):
            host, _, path = value[4:].partition(":")
        else:
            parsed = urlparse(value)
            host, path = parsed.hostname or "", parsed.path

        if provider is None:
            github_host = urlparse(config.GITHUB_WEB_URL).hostname

            provider = "github" if host == github_host else "gitlab"

        value = path

    provider = provider or "gitlab"
    value = value.split("/-/")[0].strip("/")

    if value.endswith(".git"):
        value = value[:-4]

    if provider == "github":
        # Drop /tree/main, /pull/3 and similar suffixes
        value = "/".join(value.split("/")[:2])

    if not PROJECT_PATH_PATTERN.match(value):
        raise ValueError(
            "Enter a project path such as group/project or owner/repo."
        )

    return provider, value


__all__ = [
    "AccessDenied",
    "FileEntry",
    "GitHubSource",
    "GitLabSource",
    "LocalGitSource",
    "NotFound",
    "NotSupported",
    "PROVIDERS",
    "RepositorySource",
    "SourceError",
    "UserInfo",
    "configured_providers",
    "get_source",
    "is_useful_file",
    "normalize_project_path",
    "write_token",
]
