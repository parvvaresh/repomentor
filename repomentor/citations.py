"""Turning chunks into citable sources with links to the exact lines."""

from urllib.parse import quote

from . import config
from .safety import scan


def project_web_url(chunk: dict) -> str:
    # Indexes built before multi-provider support only knew GitLab
    return chunk.get("web_url") or f"{config.GITLAB_BASE_URL}/{chunk['project_path']}"


def line_url(chunk: dict, start_line: int, end_line: int) -> str:
    if chunk.get("kind", "code") != "code":
        return chunk.get("url") or project_web_url(chunk)

    web = project_web_url(chunk)
    path = quote(chunk["file_path"])
    commit = chunk["last_commit_id"]

    if web.startswith("file:"):
        return f"{web}/{path}#L{start_line}"

    provider = chunk.get("provider", "gitlab")

    if provider == "gitlab" or (provider == "local" and "gitlab" in web):
        return f"{web}/-/blob/{commit}/{path}#L{start_line}-{end_line}"

    return f"{web}/blob/{commit}/{path}#L{start_line}-L{end_line}"


def create_citation(chunk: dict, start_line: int | None = None, end_line: int | None = None) -> str:
    if chunk.get("kind", "code") != "code":
        return f"[{chunk['file_path']}]"

    return (
        f"[{chunk['file_path']}:"
        f"{start_line or chunk['start_line']}-{end_line or chunk['end_line']}]"
    )


def describe_source(chunk: dict) -> dict:
    """The JSON shape the web interface and agents use for a source."""

    return {
        "chunk_id": chunk.get("chunk_id"),
        "kind": chunk.get("kind", "code"),
        "provider": chunk.get("provider", "gitlab"),
        "project_path": chunk["project_path"],
        "file_path": chunk["file_path"],
        "title": chunk.get("title"),
        "start_line": chunk["start_line"],
        "end_line": chunk["end_line"],
        "commit_id": chunk.get("last_commit_id", ""),
        "symbols": chunk.get("symbols", []),
        "citation": create_citation(chunk),
        "url": line_url(chunk, chunk["start_line"], chunk["end_line"]),
        "content": chunk["content"],
        "warnings": scan(chunk["content"]),
    }
