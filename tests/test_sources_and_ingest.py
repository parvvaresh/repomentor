import json

import pytest

from repomentor.ingest import ingest_history, ingest_repository, read_jsonl
from repomentor.sources import (
    GitHubSource,
    GitLabSource,
    LocalGitSource,
    normalize_project_path,
)
from repomentor.sources.local import remote_web_url

from .conftest import git


@pytest.mark.parametrize(
    ("value", "provider", "expected"),
    [
        ("group/project", None, ("gitlab", "group/project")),
        ("https://gitlab.com/group/sub/project.git", None, ("gitlab", "group/sub/project")),
        ("https://gitlab.com/group/project/-/tree/main", None, ("gitlab", "group/project")),
        ("https://github.com/owner/repo/pull/3", None, ("github", "owner/repo")),
        ("git@github.com:owner/repo.git", None, ("github", "owner/repo")),
        ("owner/repo", "github", ("github", "owner/repo")),
    ],
)
def test_normalize_project_path(value, provider, expected):
    assert normalize_project_path(value, provider) == expected


def test_normalize_project_path_rejects_garbage():
    with pytest.raises(ValueError):
        normalize_project_path("not a project")


def test_remote_web_url():
    assert remote_web_url("git@github.com:owner/repo.git") == "https://github.com/owner/repo"
    assert remote_web_url("https://token@gitlab.example.com/g/p.git") == "https://gitlab.example.com/g/p"


def test_local_source_lists_useful_files_and_history(local_repo):
    source = LocalGitSource()
    head = source.head_commit(str(local_repo))
    paths = {entry.path for entry in source.list_files(str(local_repo), head)}

    assert paths == {"app/security.py", "app/users.py", "tests/test_users.py", "README.md"}

    record = source.read_file(str(local_repo), "app/security.py", head)
    assert "def get_password_hash" in record["content"]

    commits = source.list_commits(str(local_repo), 10)
    assert [c["title"] for c in commits] == ["Add user service", "Add password hashing"]
    assert "plain text leaked" in commits[0]["message"]

    history = source.file_history(str(local_repo), "app/users.py", 10)
    assert [c["author_name"] for c in history] == ["Grace Hopper"]


def test_incremental_ingest_reuses_unchanged_files(tmp_path, local_repo):
    source = LocalGitSource()
    first = tmp_path / "first.jsonl"
    report = ingest_repository(source, str(local_repo), first)
    assert report.downloaded == 4 and report.reused == 0

    (local_repo / "app" / "users.py").write_text("print('changed')\n")
    (local_repo / "README.md").unlink()
    git(local_repo, "add", "-A")
    git(local_repo, "commit", "-q", "-m", "Change users")

    second = tmp_path / "second.jsonl"
    report = ingest_repository(source, str(local_repo), second, previous_path=first)

    assert report.downloaded == 1
    assert report.reused == 2
    assert report.changed == ["app/users.py"]
    assert report.removed == ["README.md"]
    records = {r["file_path"]: r for r in read_jsonl(second)}
    assert records["app/users.py"]["content"] == "print('changed')\n"
    assert records["app/security.py"]["provider"] == "local"


def test_ingest_history(tmp_path, local_repo):
    output = tmp_path / "history.jsonl"
    count = ingest_history(LocalGitSource(), str(local_repo), output, 10, 10)

    assert count == 2
    assert read_jsonl(output)[0]["kind"] == "commit"


class FakeResponse:
    def __init__(self, payload, headers=None, status=200):
        self.payload = payload
        self.headers = headers or {}
        self.status_code = status
        self.ok = status < 400
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


def test_gitlab_tree_is_paginated_and_filtered(monkeypatch):
    source = GitLabSource("https://gitlab.example.com", "token")
    pages = {
        1: FakeResponse(
            [{"type": "blob", "path": "app/main.py", "id": "a"},
             {"type": "tree", "path": "app", "id": "t"}],
            {"X-Next-Page": "2"},
        ),
        2: FakeResponse([
            {"type": "blob", "path": "node_modules/x.js", "id": "b"},
            {"type": "blob", "path": "image.png", "id": "c"},
            {"type": "blob", "path": "Dockerfile", "id": "d"},
        ]),
    }
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params)
        return pages[params["page"]]

    monkeypatch.setattr(source.session, "get", fake_get)
    entries = source.list_files("group/project", "abc")

    assert [entry.path for entry in entries] == ["app/main.py", "Dockerfile"]
    assert calls[0]["ref"] == "abc"
    assert source.session.headers["PRIVATE-TOKEN"] == "token"


def test_gitlab_maps_access_errors(monkeypatch):
    from repomentor.sources import AccessDenied

    source = GitLabSource("https://gitlab.example.com", "bad")
    monkeypatch.setattr(source.session, "get", lambda *a, **k: FakeResponse({}, status=401))

    with pytest.raises(AccessDenied):
        source.head_commit("group/project")

    assert source.can_read("group/project") is False


def test_github_lists_only_merged_pull_requests(monkeypatch):
    source = GitHubSource()
    pulls = [
        {"number": 1, "title": "merged", "body": "why", "user": {"login": "a"},
         "merged_at": "2024-01-01", "html_url": "u1"},
        {"number": 2, "title": "closed", "body": "", "user": {"login": "b"},
         "merged_at": None, "html_url": "u2"},
    ]
    monkeypatch.setattr(
        source.session, "request", lambda *a, **k: FakeResponse(pulls)
    )

    merged = source.list_merge_requests("owner/repo", 10)
    assert [request["number"] for request in merged] == [1]
    assert merged[0]["description"] == "why"
