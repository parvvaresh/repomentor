"""
Shared fixtures. Tests never download models or call Ollama: a small
hashing "embedding model" stands in for Sentence Transformers, and a local
Git repository stands in for GitLab and GitHub.
"""

import hashlib
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from repomentor import config


DIMENSION = 128


class FakeEmbeddingModel:
    """Bag-of-words vectors: texts sharing words get similar vectors."""

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(DIMENSION, dtype=np.float32)

        for word in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.md5(word.encode()).digest()
            vector[digest[0] % DIMENSION] += 1.0

        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def encode(self, texts, normalize_embeddings=True, **_):
        if isinstance(texts, str):
            return self._vector(texts)

        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)


@pytest.fixture
def fake_model():
    return FakeEmbeddingModel()


def git(directory: Path, *arguments: str, author: str = "Ada Lovelace <ada@example.com>"):
    name, email = author[:-1].split(" <")
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": name,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": name,
        "GIT_COMMITTER_EMAIL": email,
    }
    subprocess.run(
        ["git", "-C", str(directory), *arguments],
        check=True,
        capture_output=True,
        env=environment,
    )


SECURITY_PY = '''"""Password hashing helpers."""

from passlib import hash_backend


def get_password_hash(password: str) -> str:
    """Hash a plaintext password."""
    return hash_backend.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a plaintext password against a stored hash."""
    return hash_backend.verify(plain_password, hashed_password)
'''

USERS_PY = '''from app.security import get_password_hash


class UserService:
    """Create and update users."""

    def create_user(self, email: str, password: str) -> dict:
        hashed = get_password_hash(password)
        return {"email": email, "hashed_password": hashed}

    def delete_user(self, email: str) -> None:
        self.storage.pop(email)
'''

TEST_USERS_PY = '''from app.users import UserService


def test_create_user_hashes_password():
    user = UserService().create_user("a@example.com", "secret")
    assert user["hashed_password"] != "secret"
'''

README_MD = """# Demo app

A tiny application used in tests.

## Deployment

Run the app with docker compose up.

## Development

Install dependencies and run the tests with pytest.
"""


@pytest.fixture
def local_repo(tmp_path) -> Path:
    """A Git repository with code, tests, docs and two authors."""

    repo = tmp_path / "demo"
    (repo / "app").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "node_modules").mkdir()

    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    (repo / "app" / "security.py").write_text(SECURITY_PY)
    (repo / "README.md").write_text(README_MD)
    (repo / "node_modules" / "ignored.js").write_text("module.exports = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "Add password hashing")

    (repo / "app" / "users.py").write_text(USERS_PY)
    (repo / "tests" / "test_users.py").write_text(TEST_USERS_PY)
    git(repo, "add", ".", author="Grace Hopper <grace@example.com>")
    git(
        repo, "commit", "-q", "-m",
        "Add user service\n\nUsers need hashed passwords because plain text leaked before.",
        author="Grace Hopper <grace@example.com>",
    )
    return repo


@pytest.fixture
def built_index(tmp_path, local_repo, fake_model):
    from repomentor.index import build_index, load_index
    from repomentor.sources import LocalGitSource

    directory = tmp_path / "index"
    build_index(LocalGitSource(), str(local_repo), directory, embedding_model=fake_model)
    return load_index(directory, fake_model)


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Keep tests away from the developer's .env, data and reranker."""

    monkeypatch.setattr(config, "DATA_DIRECTORY", tmp_path / "data")
    monkeypatch.setattr(config, "RERANKER_MODEL", "")
    monkeypatch.setattr(config, "RETRIEVAL_METHOD", "auto")
    monkeypatch.setattr(config, "VECTOR_STORE", "numpy")
    monkeypatch.setattr(config, "CHUNKER", "auto")
    monkeypatch.setattr(config, "MIN_RELEVANCE", 0.0)
    monkeypatch.setattr(config, "AUTH_MODE", "none")
    monkeypatch.setattr(config, "ALLOW_MERGE_REQUESTS", False)
    monkeypatch.setattr(config, "GITLAB_TOKEN", "")
    monkeypatch.setattr(config, "GITLAB_WEBHOOK_SECRET", "")
    monkeypatch.setattr(config, "GITHUB_WEBHOOK_SECRET", "")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "fake-model")
