"""
RepoMentor web API.

Serves the React frontend and exposes endpoints to index GitLab and GitHub
repositories, keep them in sync, ask questions (streamed as Server-Sent
Events) and keep the chat history.

Run with:
    uvicorn repomentor.server:app --reload
"""

import hashlib
import hmac
import json
import re
import shutil
import sqlite3
import threading
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, llm
from .agents import DEFAULT_AGENT, AgentContext, get_agent, list_agents
from .auth import AuthError, User, access
from .embeddings import load_embedding_model
from .index import (
    CHUNKS_FILE,
    EMBEDDINGS_FILE,
    RAW_FILE,
    build_index,
    load_index,
    now,
    read_manifest,
)
from .ingest import read_jsonl
from .sources import (
    SourceError,
    configured_providers,
    get_source,
    normalize_project_path,
    write_token,
)


FRONTEND_DIRECTORY = Path(__file__).resolve().parent.parent / "frontend" / "dist"

MAX_TITLE_LENGTH = 60
HISTORY_LIMIT = 20


def data_directory() -> Path:
    return config.DATA_DIRECTORY


def repos_directory() -> Path:
    return config.DATA_DIRECTORY / "repos"


def database_path() -> Path:
    return config.DATA_DIRECTORY / "repomentor.db"


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL DEFAULT 'gitlab',
    project_path TEXT NOT NULL,
    data_dir TEXT,
    status TEXT NOT NULL,
    message TEXT,
    file_count INTEGER,
    chunk_count INTEGER,
    head_commit TEXT,
    last_sync TEXT,
    indexed_at TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (provider, project_path)
);

CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    user_id TEXT,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    payload TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS chats_by_repo ON chats(repo_id, updated_at);
CREATE INDEX IF NOT EXISTS messages_by_chat ON messages(chat_id, id);
"""

# Version 1 (the first web interface) had GitLab-only repositories with a
# unique project path and no chat owners.
MIGRATION_1_TO_2 = """
CREATE TABLE repos_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL DEFAULT 'gitlab',
    project_path TEXT NOT NULL,
    data_dir TEXT,
    status TEXT NOT NULL,
    message TEXT,
    file_count INTEGER,
    chunk_count INTEGER,
    head_commit TEXT,
    last_sync TEXT,
    indexed_at TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (provider, project_path)
);
INSERT INTO repos_new (id, provider, project_path, data_dir, status, message,
    file_count, chunk_count, indexed_at, created_at)
SELECT id, 'gitlab', project_path, data_dir, status, message,
    file_count, chunk_count, indexed_at, created_at FROM repos;
DROP TABLE repos;
ALTER TABLE repos_new RENAME TO repos;
ALTER TABLE chats ADD COLUMN user_id TEXT;
"""


@contextmanager
def database():
    connection = sqlite3.connect(database_path(), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def migrate(connection: sqlite3.Connection) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    has_repos = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'repos'"
    ).fetchone()

    if has_repos and version < 2:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(repos)")}

        if "provider" not in columns:
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.executescript(MIGRATION_1_TO_2)
            connection.execute("PRAGMA foreign_keys = ON")

    connection.executescript(SCHEMA)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def initialize_database() -> None:
    data_directory().mkdir(parents=True, exist_ok=True)

    with database() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        migrate(connection)

        # A job cannot survive a restart, so mark it as failed
        connection.execute(
            "UPDATE repos SET status = 'error', "
            "message = 'Indexing was interrupted. Re-index the repository.' "
            "WHERE status = 'indexing'"
        )


def register_existing_index() -> None:
    """
    Make an index built with the command line in data/ available in the
    web interface.
    """

    directory = data_directory()

    if not (directory / CHUNKS_FILE).exists() or not (directory / EMBEDDINGS_FILE).exists():
        return

    manifest = read_manifest(directory)
    chunks = read_jsonl(directory / CHUNKS_FILE)

    if not chunks:
        return

    provider = manifest.get("provider") or chunks[0].get("provider", "gitlab")
    project_path = manifest.get("project_path") or chunks[0]["project_path"]
    file_count = manifest.get("file_count") or len({chunk["file_path"] for chunk in chunks})

    with database() as connection:
        connection.execute(
            "INSERT OR IGNORE INTO repos (provider, project_path, data_dir, status, "
            "file_count, chunk_count, head_commit, indexed_at, created_at) "
            "VALUES (?, ?, ?, 'ready', ?, ?, ?, ?, ?)",
            (
                provider,
                project_path,
                str(directory),
                file_count,
                len(chunks),
                manifest.get("head_commit"),
                manifest.get("indexed_at") or now(),
                now(),
            ),
        )


def repo_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "provider": row["provider"],
        "project_path": row["project_path"],
        "status": row["status"],
        "message": row["message"],
        "file_count": row["file_count"],
        "chunk_count": row["chunk_count"],
        "head_commit": row["head_commit"],
        "last_sync": json.loads(row["last_sync"]) if row["last_sync"] else None,
        "indexed_at": row["indexed_at"],
        "created_at": row["created_at"],
    }


def message_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "result": json.loads(row["payload"]) if row["payload"] else None,
        "created_at": row["created_at"],
    }


def update_repo(repo_id: int, **fields) -> None:
    columns = ", ".join(f"{name} = ?" for name in fields)

    with database() as connection:
        connection.execute(
            f"UPDATE repos SET {columns} WHERE id = ?",
            (*fields.values(), repo_id),
        )


# ---------------------------------------------------------------------------
# Users and access
# ---------------------------------------------------------------------------

def current_user(
    authorization: str | None = Header(default=None),
    x_repomentor_provider: str | None = Header(default=None),
) -> User:
    token = None

    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()

    try:
        return access.authenticate(x_repomentor_provider, token)
    except AuthError as error:
        raise HTTPException(401, str(error)) from error


def can_read(user: User, repo: sqlite3.Row) -> bool:
    if access.enabled and repo["provider"] == "local":
        return False

    return access.can_read(user, repo["provider"], repo["project_path"])


def get_repo_row(connection: sqlite3.Connection, repo_id: int, user: User) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM repos WHERE id = ?", (repo_id,)).fetchone()

    if row is None or not can_read(user, row):
        raise HTTPException(404, "Repository not found.")

    return row


def get_chat_row(connection: sqlite3.Connection, chat_id: int, user: User) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()

    if row is None or (access.enabled and row["user_id"] != user.id):
        raise HTTPException(404, "Chat not found.")

    # Losing access to a repository also hides its chats
    get_repo_row(connection, row["repo_id"], user)
    return row


# ---------------------------------------------------------------------------
# Search indexes and indexing jobs
# ---------------------------------------------------------------------------

# The embedding model is shared by every repository and by indexing jobs.
# Encoding is serialized because the model is not guaranteed to be
# thread-safe.
model_lock = threading.Lock()

index_cache: dict[int, object] = {}
index_cache_lock = threading.Lock()

jobs_lock = threading.Lock()
running_jobs: set[int] = set()
resync_requested: set[int] = set()


def get_embedding_model():
    with model_lock:
        return load_embedding_model()


def get_index(repo: sqlite3.Row):
    with index_cache_lock:
        index = index_cache.get(repo["id"])

        if index is None:
            index = load_index(Path(repo["data_dir"]), get_embedding_model())
            index_cache[repo["id"]] = index

    return index


def forget_index(repo_id: int) -> None:
    with index_cache_lock:
        index_cache.pop(repo_id, None)


def describe_error(error: Exception) -> str:
    return str(error) or error.__class__.__name__


def source_project(repo: sqlite3.Row) -> str:
    return repo["project_path"]


def index_repository(repo_id: int, provider: str, project_path: str, full: bool = False) -> None:
    """
    Build or synchronize a repository's index in a background thread.

    Files are built in a temporary directory and swapped in at the end, so
    the existing index keeps answering questions meanwhile. Unless `full`
    is set, unchanged files are reused from the existing index.
    """

    final_directory = repos_directory() / str(repo_id)
    build_directory = repos_directory() / f"{repo_id}.building"

    with database() as connection:
        row = connection.execute("SELECT data_dir FROM repos WHERE id = ?", (repo_id,)).fetchone()

    previous = Path(row["data_dir"]) if row and row["data_dir"] else None

    if full or previous is None or not (previous / RAW_FILE).exists():
        previous = None

    try:
        shutil.rmtree(build_directory, ignore_errors=True)
        build_directory.mkdir(parents=True)

        manifest = build_index(
            get_source(provider),
            project_path,
            build_directory,
            previous=previous,
            report=lambda message: update_repo(repo_id, message=message),
            embedding_model=get_embedding_model(),
            model_lock=model_lock,
            final_directory=final_directory,
        )

        if final_directory.exists():
            shutil.rmtree(final_directory)

        build_directory.rename(final_directory)
        forget_index(repo_id)

        sync = manifest["sync"]
        message = None

        if sync["failed"]:
            message = f"{sync['failed']} file(s) could not be downloaded."
        elif sync["incremental"]:
            changed = len(sync["changed"]) + len(sync["removed"])
            message = (
                f"Synced: {changed} file(s) changed." if changed
                else "Already up to date."
            )

        update_repo(
            repo_id,
            data_dir=str(final_directory),
            status="ready",
            message=message,
            file_count=manifest["file_count"],
            chunk_count=manifest["chunk_count"],
            head_commit=manifest["head_commit"],
            last_sync=json.dumps({
                key: sync[key] for key in (
                    "downloaded", "reused", "failed", "reused_vectors", "incremental",
                )
            } | {"changed": len(sync["changed"]), "removed": len(sync["removed"])}),
            indexed_at=manifest["indexed_at"],
        )

    except Exception as error:
        shutil.rmtree(build_directory, ignore_errors=True)
        update_repo(
            repo_id,
            # Keep serving the previous index after a failed sync
            status="ready" if previous else "error",
            message=f"Sync failed: {describe_error(error)}" if previous else describe_error(error),
        )

    finally:
        with jobs_lock:
            running_jobs.discard(repo_id)
            again = repo_id in resync_requested
            resync_requested.discard(repo_id)

        if again:
            schedule_indexing(repo_id, provider, project_path)


def schedule_indexing(repo_id: int, provider: str, project_path: str, full: bool = False) -> bool:
    """Start a job, or remember to run another one when the current ends."""

    with jobs_lock:
        if repo_id in running_jobs:
            resync_requested.add(repo_id)
            return False

        running_jobs.add(repo_id)

    update_repo(repo_id, status="indexing", message="Queued…")
    threading.Thread(
        target=index_repository,
        args=(repo_id, provider, project_path, full),
        daemon=True,
    ).start()
    return True


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class RepoCreate(BaseModel):
    project_path: str = Field(min_length=3, max_length=500)
    provider: str | None = Field(default=None, max_length=20)


class ChatCreate(BaseModel):
    repo_id: int


class ChatUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class QuestionCreate(BaseModel):
    question: str = Field(min_length=1, max_length=40000)
    top_k: int = Field(default=config.TOP_K, ge=1, le=config.MAX_TOP_K)
    retrieval_only: bool = False
    agent: str = Field(default=DEFAULT_AGENT, max_length=50)


class MergeRequestCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    # Open the merge request even if a syntax check failed
    force: bool = False


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    register_existing_index()
    yield


app = FastAPI(title="RepoMentor", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Status and repositories
# ---------------------------------------------------------------------------

@app.get("/api/status")
def get_status() -> dict:
    return {
        "ollama_available": llm.available(),
        "model": config.MODEL_NAME,
        "providers": configured_providers(),
        "gitlab_configured": "gitlab" in configured_providers(),
        "gitlab_base_url": config.GITLAB_BASE_URL,
        "github_web_url": config.GITHUB_WEB_URL,
        "auth_mode": config.AUTH_MODE,
        "allow_merge_requests": config.ALLOW_MERGE_REQUESTS,
        "default_top_k": config.TOP_K,
        "max_top_k": config.MAX_TOP_K,
        "agents": [agent.describe() for agent in list_agents()],
        "default_agent": DEFAULT_AGENT,
    }


@app.get("/api/me")
def get_me(user: User = Depends(current_user)) -> dict:
    return {
        "id": user.id,
        "provider": user.provider,
        "username": user.username,
        "name": user.name,
    }


@app.get("/api/repos")
def list_repos(user: User = Depends(current_user)) -> list[dict]:
    with database() as connection:
        rows = connection.execute("SELECT * FROM repos ORDER BY project_path").fetchall()

    return [repo_to_dict(row) for row in rows if can_read(user, row)]


@app.post("/api/repos", status_code=201)
def create_repo(body: RepoCreate, user: User = Depends(current_user)) -> dict:
    try:
        provider, project_path = normalize_project_path(body.project_path, body.provider)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error

    if provider not in configured_providers():
        raise HTTPException(422, f"{provider} is not configured on the server.")

    if access.enabled and not access.can_read(user, provider, project_path):
        raise HTTPException(404, "Project not found, or your token cannot read it.")

    with database() as connection:
        existing = connection.execute(
            "SELECT id FROM repos WHERE provider = ? AND project_path = ?",
            (provider, project_path),
        ).fetchone()

        if existing:
            raise HTTPException(409, "This repository is already added.")

        cursor = connection.execute(
            "INSERT INTO repos (provider, project_path, status, message, created_at) "
            "VALUES (?, ?, 'indexing', 'Queued…', ?)",
            (provider, project_path, now()),
        )
        repo_id = cursor.lastrowid

    schedule_indexing(repo_id, provider, project_path)

    with database() as connection:
        return repo_to_dict(get_repo_row(connection, repo_id, user))


@app.post("/api/repos/{repo_id}/reindex")
def reindex_repo(repo_id: int, full: bool = False, user: User = Depends(current_user)) -> dict:
    with database() as connection:
        repo = get_repo_row(connection, repo_id, user)

    if repo["status"] == "indexing":
        raise HTTPException(409, "This repository is already indexing.")

    schedule_indexing(repo_id, repo["provider"], repo["project_path"], full)

    with database() as connection:
        return repo_to_dict(get_repo_row(connection, repo_id, user))


@app.delete("/api/repos/{repo_id}", status_code=204)
def delete_repo(repo_id: int, user: User = Depends(current_user)) -> None:
    with database() as connection:
        repo = get_repo_row(connection, repo_id, user)

        if repo["status"] == "indexing":
            raise HTTPException(409, "Wait for indexing to finish before removing it.")

        connection.execute("DELETE FROM repos WHERE id = ?", (repo_id,))

    forget_index(repo_id)

    # Only remove indexes the server created, never the CLI's data/ files
    if repo["data_dir"]:
        directory = Path(repo["data_dir"]).resolve()

        if directory.parent == repos_directory().resolve():
            if read_manifest(directory).get("vector_store") == "pgvector":
                try:
                    load_index(directory, get_embedding_model()).store.delete()
                except Exception:
                    pass

            shutil.rmtree(directory, ignore_errors=True)


# ---------------------------------------------------------------------------
# Chats
# ---------------------------------------------------------------------------

@app.get("/api/chats")
def list_chats(repo_id: int | None = None, user: User = Depends(current_user)) -> list[dict]:
    query = (
        "SELECT chats.*, COUNT(messages.id) AS message_count "
        "FROM chats LEFT JOIN messages ON messages.chat_id = chats.id WHERE 1 = 1 "
    )
    parameters: list = []

    if repo_id is not None:
        with database() as connection:
            get_repo_row(connection, repo_id, user)

        query += "AND chats.repo_id = ? "
        parameters.append(repo_id)

    if access.enabled:
        query += "AND chats.user_id = ? "
        parameters.append(user.id)

    query += "GROUP BY chats.id ORDER BY chats.updated_at DESC, chats.id DESC"

    with database() as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [
        {
            "id": row["id"],
            "repo_id": row["repo_id"],
            "title": row["title"],
            "message_count": row["message_count"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


@app.post("/api/chats", status_code=201)
def create_chat(body: ChatCreate, user: User = Depends(current_user)) -> dict:
    timestamp = now()

    with database() as connection:
        get_repo_row(connection, body.repo_id, user)
        cursor = connection.execute(
            "INSERT INTO chats (repo_id, user_id, title, created_at, updated_at) "
            "VALUES (?, ?, 'New chat', ?, ?)",
            (body.repo_id, user.id, timestamp, timestamp),
        )

    return {
        "id": cursor.lastrowid,
        "repo_id": body.repo_id,
        "title": "New chat",
        "message_count": 0,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: int, user: User = Depends(current_user)) -> dict:
    with database() as connection:
        chat = get_chat_row(connection, chat_id, user)
        messages = connection.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()

    return {
        "id": chat["id"],
        "repo_id": chat["repo_id"],
        "title": chat["title"],
        "created_at": chat["created_at"],
        "updated_at": chat["updated_at"],
        "messages": [message_to_dict(row) for row in messages],
    }


@app.patch("/api/chats/{chat_id}")
def rename_chat(chat_id: int, body: ChatUpdate, user: User = Depends(current_user)) -> dict:
    with database() as connection:
        get_chat_row(connection, chat_id, user)
        connection.execute(
            "UPDATE chats SET title = ? WHERE id = ?",
            (body.title.strip(), chat_id),
        )

    return {"id": chat_id, "title": body.title.strip()}


@app.delete("/api/chats/{chat_id}", status_code=204)
def delete_chat(chat_id: int, user: User = Depends(current_user)) -> None:
    with database() as connection:
        get_chat_row(connection, chat_id, user)
        connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def save_message(chat_id: int, role: str, content: str, payload: dict | None = None) -> dict:
    timestamp = now()

    with database() as connection:
        cursor = connection.execute(
            "INSERT INTO messages (chat_id, role, content, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                chat_id,
                role,
                content,
                json.dumps(payload) if payload is not None else None,
                timestamp,
            ),
        )
        connection.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ?",
            (timestamp, chat_id),
        )
        row = connection.execute(
            "SELECT * FROM messages WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()

    return message_to_dict(row)


def load_history(chat_id: int) -> list[dict]:
    """Recent successful turns of a chat, oldest first, for agents."""

    with database() as connection:
        rows = connection.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
            (chat_id, HISTORY_LIMIT),
        ).fetchall()

    history = []

    for row in reversed(rows):
        payload = json.loads(row["payload"]) if row["payload"] else {}

        if "error" in payload:
            continue

        history.append({
            "role": row["role"],
            "content": row["content"],
            # Messages saved before agents existed came from the mentor
            "agent": payload.get("agent", DEFAULT_AGENT),
        })

    return history


def prepare_question(chat_id: int, body: QuestionCreate, user: User):
    """Validate a question and return what answering it needs."""

    question = body.question.strip()

    if not question:
        raise HTTPException(422, "The question is empty.")

    agent = get_agent(body.agent)

    if agent is None:
        raise HTTPException(422, f"Unknown agent '{body.agent}'.")

    with database() as connection:
        chat = get_chat_row(connection, chat_id, user)
        repo = get_repo_row(connection, chat["repo_id"], user)

        if not repo["data_dir"] or not (Path(repo["data_dir"]) / CHUNKS_FILE).exists():
            raise HTTPException(
                409,
                "This repository is not indexed yet. "
                "Wait for indexing to finish or re-index it.",
            )

        # Name the chat after its first question
        if chat["title"] == "New chat":
            has_messages = connection.execute(
                "SELECT 1 FROM messages WHERE chat_id = ? LIMIT 1",
                (chat_id,),
            ).fetchone()

            if not has_messages:
                title = question.splitlines()[0]

                if len(title) > MAX_TITLE_LENGTH:
                    title = title[:MAX_TITLE_LENGTH - 1].rstrip() + "…"

                connection.execute(
                    "UPDATE chats SET title = ? WHERE id = ?",
                    (title, chat_id),
                )

    return question, agent, repo


def answer_events(chat_id: int, question: str, agent, repo, body: QuestionCreate, user: User):
    """
    Run an agent and yield (event_type, data) pairs: the saved user message,
    the agent's progress events and finally the saved assistant message.
    The question and the answer (or the error that prevented one) are both
    kept in the chat history.
    """

    history = load_history(chat_id)
    user_message = save_message(
        chat_id,
        "user",
        question,
        {"top_k": body.top_k, "retrieval_only": body.retrieval_only, "agent": agent.id},
    )
    yield "user", user_message

    try:
        # Live host access acts as the signed-in user when there is one
        token = user.token if user.provider == repo["provider"] else None
        context = AgentContext(
            index=get_index(repo),
            project_path=repo["project_path"],
            top_k=body.top_k,
            retrieval_only=body.retrieval_only,
            retrieve_lock=model_lock,
            history=history,
            source=get_source(repo["provider"], token),
            source_project=source_project(repo),
        )
        result = None

        for event in agent.stream(question, context):
            if event["type"] == "result":
                result = event["result"]
            else:
                yield event["type"], event

        if result is None:
            raise RuntimeError("The agent finished without an answer.")

        result = {**result, "agent": agent.id}
        assistant = save_message(chat_id, "assistant", agent.format_text(result), result)

    except GeneratorExit:
        # The browser went away; keep the chat consistent
        save_message(
            chat_id,
            "assistant",
            "The answer was interrupted.",
            {"error": "The answer was interrupted.", "agent": agent.id},
        )
        raise

    except Exception as error:
        detail = describe_error(error)
        assistant = save_message(chat_id, "assistant", detail, {"error": detail, "agent": agent.id})

    yield "message", assistant


@app.post("/api/chats/{chat_id}/messages")
def ask_question(chat_id: int, body: QuestionCreate, user: User = Depends(current_user)) -> dict:
    """Answer a question and return both saved messages at once."""

    question, agent, repo = prepare_question(chat_id, body, user)
    messages = [
        data for kind, data in answer_events(chat_id, question, agent, repo, body, user)
        if kind in ("user", "message")
    ]
    return {"messages": messages}


def sse(kind: str, data) -> str:
    return f"event: {kind}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/chats/{chat_id}/messages/stream")
def ask_question_stream(chat_id: int, body: QuestionCreate, user: User = Depends(current_user)):
    """
    Answer a question as Server-Sent Events: `user` (the saved question),
    then `status`, `sources`, `delta` and `fact` events while the agent
    works, and `message` (the saved answer) at the end.
    """

    question, agent, repo = prepare_question(chat_id, body, user)

    def events():
        for kind, data in answer_events(chat_id, question, agent, repo, body, user):
            yield sse(kind, data)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Merge requests from coder answers
# ---------------------------------------------------------------------------

def branch_name(message_id: int, title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "change"
    return f"repomentor/{message_id}-{slug}"


@app.post("/api/messages/{message_id}/merge-request")
def open_merge_request(
    message_id: int,
    body: MergeRequestCreate,
    user: User = Depends(current_user),
) -> dict:
    if not config.ALLOW_MERGE_REQUESTS:
        raise HTTPException(403, "Opening merge requests is disabled (ALLOW_MERGE_REQUESTS).")

    with database() as connection:
        message = connection.execute(
            "SELECT * FROM messages WHERE id = ?", (message_id,)
        ).fetchone()

        if message is None:
            raise HTTPException(404, "Message not found.")

        chat = get_chat_row(connection, message["chat_id"], user)
        repo = get_repo_row(connection, chat["repo_id"], user)
        question = connection.execute(
            "SELECT content FROM messages WHERE chat_id = ? AND id < ? AND role = 'user' "
            "ORDER BY id DESC LIMIT 1",
            (chat["id"], message_id),
        ).fetchone()

    payload = json.loads(message["payload"] or "{}")

    if payload.get("agent") != "coder" or not payload.get("changes"):
        raise HTTPException(422, "This answer has no code changes.")

    if payload.get("merge_request_created"):
        return payload["merge_request_created"]

    changes = [c for c in payload["changes"] if c["status"] in ("created", "modified")]

    if not changes:
        raise HTTPException(422, "None of the changes could be applied.")

    if not body.force and any(change["check"]["ok"] is False for change in changes):
        raise HTTPException(422, "A syntax check failed. Fix the code or open it anyway.")

    token = user.token if user.provider == repo["provider"] else write_token(repo["provider"])

    if not token:
        raise HTTPException(409, "No token with write access is configured for this host.")

    title = body.title or (question["content"].splitlines()[0][:100] if question else "RepoMentor change")
    description = (
        f"{payload.get('answer', '')[:6000]}\n\n---\n"
        f"Generated by RepoMentor's coder agent ({payload.get('model')}) "
        f"from commit {payload.get('base_commit') or 'unknown'}. "
        "Review every change before merging."
    )

    try:
        created = get_source(repo["provider"], token).create_merge_request(
            source_project(repo),
            branch_name(message_id, title),
            title,
            description,
            {change["path"]: change["content"] for change in changes},
        )
    except SourceError as error:
        raise HTTPException(502, describe_error(error)) from error

    payload["merge_request_created"] = created

    with database() as connection:
        connection.execute(
            "UPDATE messages SET payload = ? WHERE id = ?",
            (json.dumps(payload), message_id),
        )

    return created


# ---------------------------------------------------------------------------
# Webhooks: keep indexes in sync with pushes
# ---------------------------------------------------------------------------

def sync_on_push(provider: str, project_path: str) -> dict:
    with database() as connection:
        repo = connection.execute(
            "SELECT * FROM repos WHERE provider = ? AND project_path = ?",
            (provider, project_path),
        ).fetchone()

    if repo is None:
        return {"queued": False, "reason": "Repository is not indexed by RepoMentor."}

    started = schedule_indexing(repo["id"], provider, project_path)
    return {"queued": True, "started": started}


@app.post("/api/webhooks/gitlab")
async def gitlab_webhook(request: Request, x_gitlab_token: str | None = Header(default=None)):
    secret = config.GITLAB_WEBHOOK_SECRET

    if not secret:
        raise HTTPException(404, "GitLab webhooks are not enabled.")

    if not x_gitlab_token or not hmac.compare_digest(x_gitlab_token, secret):
        raise HTTPException(401, "Invalid webhook token.")

    event = await request.json()

    if event.get("object_kind") != "push":
        return {"queued": False, "reason": "Only push events trigger a sync."}

    project = event.get("project") or {}
    default_branch = project.get("default_branch")

    if default_branch and event.get("ref") != f"refs/heads/{default_branch}":
        return {"queued": False, "reason": "Push was not to the default branch."}

    return sync_on_push("gitlab", project.get("path_with_namespace", ""))


@app.post("/api/webhooks/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
):
    secret = config.GITHUB_WEBHOOK_SECRET

    if not secret:
        raise HTTPException(404, "GitHub webhooks are not enabled.")

    raw = await request.body()
    expected = "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()

    if not x_hub_signature_256 or not hmac.compare_digest(x_hub_signature_256, expected):
        raise HTTPException(401, "Invalid webhook signature.")

    if x_github_event == "ping":
        return {"queued": False, "reason": "pong"}

    if x_github_event != "push":
        return {"queued": False, "reason": "Only push events trigger a sync."}

    event = json.loads(raw)
    repository = event.get("repository") or {}
    default_branch = repository.get("default_branch")

    if default_branch and event.get("ref") != f"refs/heads/{default_branch}":
        return {"queued": False, "reason": "Push was not to the default branch."}

    return sync_on_push("github", repository.get("full_name", ""))


# Serve the built React app (npm run build) from the same server
if FRONTEND_DIRECTORY.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIRECTORY, html=True), name="frontend")
