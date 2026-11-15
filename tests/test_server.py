import hashlib
import hmac
import json
import shutil
import sqlite3

import pytest
from fastapi.testclient import TestClient

from repomentor import config, server
from repomentor.auth import access
from repomentor.sources import UserInfo


@pytest.fixture
def client(tmp_path, built_index, fake_model, monkeypatch):
    """A server whose data directory holds the demo repository's index."""

    data = config.DATA_DIRECTORY
    data.mkdir(parents=True)

    for path in built_index.directory.iterdir():
        shutil.copy(path, data / path.name)

    monkeypatch.setattr(server, "load_embedding_model", lambda *a, **k: fake_model)
    server.index_cache.clear()
    access.forget()

    with TestClient(server.app) as test_client:
        yield test_client


def first_repo(client):
    [repo] = client.get("/api/repos").json()
    return repo


def test_status_lists_agents_and_providers(client):
    status = client.get("/api/status").json()

    assert {agent["id"] for agent in status["agents"]} >= {"mentor", "coder", "reviewer", "experts", "tour"}
    assert "github" in status["providers"]
    assert status["auth_mode"] == "none"


def test_cli_index_is_registered(client):
    repo = first_repo(client)

    assert repo["provider"] == "local"
    assert repo["status"] == "ready"
    assert repo["head_commit"]


def test_chat_and_streamed_answer(client):
    repo = first_repo(client)
    chat = client.post("/api/chats", json={"repo_id": repo["id"]}).json()

    with client.stream(
        "POST",
        f"/api/chats/{chat['id']}/messages/stream",
        json={"question": "How is a password hashed?", "retrieval_only": True},
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    events = [block.split("\n")[0][len("event: "):] for block in body.strip().split("\n\n")]
    assert events[0] == "user"
    assert events[-1] == "message"
    assert "status" in events

    saved = client.get(f"/api/chats/{chat['id']}").json()
    assert [m["role"] for m in saved["messages"]] == ["user", "assistant"]
    assert saved["title"] == "How is a password hashed?"
    assert saved["messages"][1]["result"]["retrieval_only"] is True


def test_agent_errors_are_saved_in_the_chat(client):
    repo = first_repo(client)
    chat = client.post("/api/chats", json={"repo_id": repo["id"]}).json()
    response = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"question": "review this", "agent": "reviewer"},
    )

    assert response.status_code == 200
    assistant = response.json()["messages"][1]
    assert "merge request number" in assistant["result"]["error"]


def test_unknown_agent_is_rejected(client):
    repo = first_repo(client)
    chat = client.post("/api/chats", json={"repo_id": repo["id"]}).json()
    response = client.post(f"/api/chats/{chat['id']}/messages", json={"question": "x", "agent": "nope"})

    assert response.status_code == 422


def test_merge_requests_are_disabled_by_default(client):
    assert client.post("/api/messages/1/merge-request", json={}).status_code == 403


def test_github_webhook_checks_signature(client, monkeypatch):
    monkeypatch.setattr(config, "GITHUB_WEBHOOK_SECRET", "s3cret")
    body = json.dumps({
        "ref": "refs/heads/main",
        "repository": {"full_name": "owner/unknown", "default_branch": "main"},
    }).encode()

    bad = client.post("/api/webhooks/github", content=body, headers={
        "X-Hub-Signature-256": "sha256=wrong", "X-GitHub-Event": "push",
    })
    assert bad.status_code == 401

    signature = "sha256=" + hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
    good = client.post("/api/webhooks/github", content=body, headers={
        "X-Hub-Signature-256": signature, "X-GitHub-Event": "push",
    })
    assert good.status_code == 200
    assert good.json()["queued"] is False  # not an indexed repository


def test_gitlab_webhook_queues_a_sync(client, monkeypatch):
    monkeypatch.setattr(config, "GITLAB_WEBHOOK_SECRET", "token")
    scheduled = []
    monkeypatch.setattr(server, "schedule_indexing", lambda *args, **kwargs: scheduled.append(args) or True)

    with server.database() as connection:
        connection.execute(
            "INSERT INTO repos (provider, project_path, status, created_at) "
            "VALUES ('gitlab', 'group/app', 'ready', 'now')"
        )

    event = {
        "object_kind": "push",
        "ref": "refs/heads/main",
        "project": {"path_with_namespace": "group/app", "default_branch": "main"},
    }

    assert client.post("/api/webhooks/gitlab", json=event).status_code == 401
    response = client.post("/api/webhooks/gitlab", json=event, headers={"X-Gitlab-Token": "token"})

    assert response.json()["queued"] is True
    assert scheduled[0][1:] == ("gitlab", "group/app")


def test_version_1_database_is_migrated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIRECTORY", tmp_path / "old")
    (tmp_path / "old").mkdir()
    connection = sqlite3.connect(tmp_path / "old" / "repomentor.db")
    connection.executescript("""
        CREATE TABLE repos (id INTEGER PRIMARY KEY AUTOINCREMENT, project_path TEXT NOT NULL UNIQUE,
            data_dir TEXT, status TEXT NOT NULL, message TEXT, file_count INTEGER,
            chunk_count INTEGER, indexed_at TEXT, created_at TEXT NOT NULL);
        CREATE TABLE chats (id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_id INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
            title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
            role TEXT NOT NULL, content TEXT NOT NULL, payload TEXT, created_at TEXT NOT NULL);
        INSERT INTO repos (project_path, status, created_at) VALUES ('g/p', 'ready', 'x');
        INSERT INTO chats (repo_id, title, created_at, updated_at) VALUES (1, 'old', 'x', 'x');
    """)
    connection.commit()
    connection.close()

    server.initialize_database()

    with server.database() as db:
        repo = db.execute("SELECT * FROM repos").fetchone()
        assert repo["provider"] == "gitlab"
        assert db.execute("SELECT user_id FROM chats").fetchone()["user_id"] is None
        # Cascading deletes still work after the table rebuild
        db.execute("DELETE FROM repos WHERE id = 1")
        assert db.execute("SELECT COUNT(*) FROM chats").fetchone()[0] == 0


class FakeHost:
    def __init__(self, token):
        self.token = token

    def current_user(self):
        if self.token not in ("alice-token", "bob-token"):
            from repomentor.sources import AccessDenied

            raise AccessDenied("bad token")

        name = self.token.split("-")[0]
        return UserInfo(id=name, username=name, name=name.title())

    def can_read(self, project):
        return self.token == "alice-token" and project == "group/app"


def test_token_mode_limits_repos_and_chats(client, monkeypatch):
    monkeypatch.setattr(config, "AUTH_MODE", "token")
    monkeypatch.setattr("repomentor.auth.get_source", lambda provider, token: FakeHost(token))

    with server.database() as connection:
        connection.execute(
            "INSERT INTO repos (provider, project_path, status, created_at) "
            "VALUES ('gitlab', 'group/app', 'ready', 'now')"
        )

    alice = {"Authorization": "Bearer alice-token", "X-RepoMentor-Provider": "gitlab"}
    bob = {"Authorization": "Bearer bob-token", "X-RepoMentor-Provider": "gitlab"}

    assert client.get("/api/repos").status_code == 401
    assert client.get("/api/repos", headers={**alice, "Authorization": "Bearer nope"}).status_code == 401

    alice_repos = client.get("/api/repos", headers=alice).json()
    # The local index has no owner to ask, so it is hidden from everyone
    assert [repo["project_path"] for repo in alice_repos] == ["group/app"]
    assert client.get("/api/repos", headers=bob).json() == []

    repo_id = alice_repos[0]["id"]
    chat = client.post("/api/chats", json={"repo_id": repo_id}, headers=alice).json()
    assert client.get(f"/api/chats/{chat['id']}", headers=alice).status_code == 200
    assert client.get(f"/api/chats/{chat['id']}", headers=bob).status_code == 404
    assert client.post("/api/chats", json={"repo_id": repo_id}, headers=bob).status_code == 404
    assert client.get("/api/me", headers=alice).json()["name"] == "Alice"
