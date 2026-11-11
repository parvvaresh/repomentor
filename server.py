"""
RepoMentor web API.

Serves the React frontend and exposes endpoints to index several GitLab
repositories, ask questions about them and keep the chat history.

Run with:
    uvicorn server:app --reload
"""

import json
import os
import re
import shutil
import sqlite3
import threading
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import rag_answer
from agents import DEFAULT_AGENT, AgentContext, get_agent, list_agents
from chunk_dataset import chunk_dataset
from create_embeddings import create_embeddings
from ingest_repository import ingest_repository


DATA_DIRECTORY = Path("data")
REPOS_DIRECTORY = DATA_DIRECTORY / "repos"
DATABASE_PATH = DATA_DIRECTORY / "repomentor.db"
FRONTEND_DIRECTORY = Path("frontend/dist")

RAW_FILE = "repository_files.jsonl"
CHUNKS_FILE = "chunks.jsonl"
EMBEDDINGS_FILE = "embeddings.npy"

MAX_TITLE_LENGTH = 60
MAX_TOP_K = 20
HISTORY_LIMIT = 20

PROJECT_PATH_PATTERN = re.compile(r"^[\w.-]+(?:/[\w.-]+)+$")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_path TEXT NOT NULL UNIQUE,
    data_dir TEXT,
    status TEXT NOT NULL,
    message TEXT,
    file_count INTEGER,
    chunk_count INTEGER,
    indexed_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
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


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def database():
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_database() -> None:
    DATA_DIRECTORY.mkdir(exist_ok=True)

    with database() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(SCHEMA)

        # A job cannot survive a restart, so mark it as failed
        connection.execute(
            "UPDATE repos SET status = 'error', "
            "message = 'Indexing was interrupted. Re-index the repository.' "
            "WHERE status = 'indexing'"
        )


def register_existing_index() -> None:
    """
    Make an index built with the command-line scripts (data/chunks.jsonl
    and data/embeddings.npy) available in the web interface.
    """

    chunks_path = DATA_DIRECTORY / CHUNKS_FILE
    embeddings_path = DATA_DIRECTORY / EMBEDDINGS_FILE

    if not chunks_path.exists() or not embeddings_path.exists():
        return

    chunks = rag_answer.load_chunks(chunks_path)

    if not chunks:
        return

    project_path = chunks[0]["project_path"]
    file_count = len({chunk["file_path"] for chunk in chunks})

    with database() as connection:
        connection.execute(
            "INSERT OR IGNORE INTO repos (project_path, data_dir, status, "
            "message, file_count, chunk_count, indexed_at, created_at) "
            "VALUES (?, ?, 'ready', NULL, ?, ?, ?, ?)",
            (
                project_path,
                str(DATA_DIRECTORY),
                file_count,
                len(chunks),
                now(),
                now(),
            ),
        )


def repo_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "project_path": row["project_path"],
        "status": row["status"],
        "message": row["message"],
        "file_count": row["file_count"],
        "chunk_count": row["chunk_count"],
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


def get_repo_row(connection: sqlite3.Connection, repo_id: int) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM repos WHERE id = ?",
        (repo_id,),
    ).fetchone()

    if row is None:
        raise HTTPException(404, "Repository not found.")

    return row


def get_chat_row(connection: sqlite3.Connection, chat_id: int) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM chats WHERE id = ?",
        (chat_id,),
    ).fetchone()

    if row is None:
        raise HTTPException(404, "Chat not found.")

    return row


def update_repo(repo_id: int, **fields) -> None:
    columns = ", ".join(f"{name} = ?" for name in fields)

    with database() as connection:
        connection.execute(
            f"UPDATE repos SET {columns} WHERE id = ?",
            (*fields.values(), repo_id),
        )


# ---------------------------------------------------------------------------
# Search indexes
# ---------------------------------------------------------------------------

# The embedding model is shared by every repository and by indexing jobs.
# Encoding is serialized because the model is not guaranteed to be
# thread-safe.
model_lock = threading.Lock()
embedding_model = None

index_cache: dict[int, dict] = {}
index_cache_lock = threading.Lock()


def get_embedding_model():
    global embedding_model

    with model_lock:
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer

            embedding_model = SentenceTransformer(rag_answer.EMBEDDING_MODEL)

    return embedding_model


def get_index(repo: sqlite3.Row) -> dict:
    with index_cache_lock:
        index = index_cache.get(repo["id"])

        if index is None:
            data_directory = Path(repo["data_dir"])
            index = rag_answer.load_index(
                data_directory / CHUNKS_FILE,
                data_directory / EMBEDDINGS_FILE,
                get_embedding_model(),
            )
            index_cache[repo["id"]] = index

    return index


def forget_index(repo_id: int) -> None:
    with index_cache_lock:
        index_cache.pop(repo_id, None)


def describe_ingestion_error(error: Exception) -> str:
    if isinstance(error, requests.HTTPError) and error.response is not None:
        status = error.response.status_code

        if status in (401, 403):
            return "GitLab rejected the token. Check GITLAB_TOKEN in .env."

        if status == 404:
            return (
                "Project not found on GitLab, or the token "
                "cannot access it."
            )

    if isinstance(error, requests.RequestException):
        return f"Could not reach GitLab: {error}"

    return str(error) or error.__class__.__name__


def index_repository(repo_id: int, project_path: str) -> None:
    """
    Download, chunk and embed a repository in a background thread.
    Files are built in a temporary directory and swapped in at the end,
    so an existing index keeps working while it is being rebuilt.
    """

    final_directory = REPOS_DIRECTORY / str(repo_id)
    build_directory = REPOS_DIRECTORY / f"{repo_id}.building"

    try:
        shutil.rmtree(build_directory, ignore_errors=True)
        build_directory.mkdir(parents=True)

        update_repo(repo_id, message="Listing repository files…")

        def report(position: int, total: int, file_path: str, succeeded: bool):
            if position == total or position % 5 == 0:
                update_repo(
                    repo_id,
                    message=f"Downloading files ({position}/{total})…",
                )

        downloaded, failed = ingest_repository(
            project_path,
            build_directory / RAW_FILE,
            report,
        )

        if downloaded == 0:
            raise RuntimeError(
                "No supported source files were found in this repository."
            )

        update_repo(repo_id, message="Splitting files into chunks…")
        chunk_count, _ = chunk_dataset(
            build_directory / RAW_FILE,
            build_directory / CHUNKS_FILE,
        )

        if chunk_count == 0:
            raise RuntimeError("The repository files produced no chunks.")

        update_repo(
            repo_id,
            message=f"Creating embeddings for {chunk_count} chunks…",
        )
        model = get_embedding_model()

        with model_lock:
            create_embeddings(
                build_directory / CHUNKS_FILE,
                build_directory / EMBEDDINGS_FILE,
                model,
                show_progress_bar=False,
            )

        shutil.rmtree(final_directory, ignore_errors=True)
        build_directory.rename(final_directory)
        forget_index(repo_id)

        message = None

        if failed:
            message = f"{failed} file(s) could not be downloaded."

        update_repo(
            repo_id,
            data_dir=str(final_directory),
            status="ready",
            message=message,
            file_count=downloaded,
            chunk_count=chunk_count,
            indexed_at=now(),
        )

    except Exception as error:
        shutil.rmtree(build_directory, ignore_errors=True)
        update_repo(
            repo_id,
            status="error",
            message=describe_ingestion_error(error),
        )


def start_indexing(repo_id: int, project_path: str) -> None:
    thread = threading.Thread(
        target=index_repository,
        args=(repo_id, project_path),
        daemon=True,
    )
    thread.start()


def normalize_project_path(value: str) -> str:
    """
    Accept "group/project" as well as a pasted GitLab URL such as
    "https://gitlab.com/group/project.git" or ".../-/tree/main".
    """

    value = value.strip()

    if "://" in value:
        value = urlparse(value).path

    value = value.split("/-/")[0].strip("/")

    if value.endswith(".git"):
        value = value[:-4]

    if not PROJECT_PATH_PATTERN.match(value):
        raise HTTPException(
            422,
            "Enter a GitLab project path such as group/project.",
        )

    return value


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class RepoCreate(BaseModel):
    project_path: str = Field(min_length=3, max_length=300)


class ChatCreate(BaseModel):
    repo_id: int


class ChatUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class QuestionCreate(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=rag_answer.TOP_K, ge=1, le=MAX_TOP_K)
    retrieval_only: bool = False
    agent: str = Field(default=DEFAULT_AGENT, max_length=50)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    register_existing_index()
    yield


app = FastAPI(title="RepoMentor", lifespan=lifespan)


@app.get("/api/status")
def get_status() -> dict:
    try:
        response = requests.get(
            rag_answer.OLLAMA_URL.replace("/api/chat", "/api/tags"),
            timeout=2,
        )
        ollama_available = response.ok
    except requests.RequestException:
        ollama_available = False

    gitlab_configured = bool(
        os.getenv("GITLAB_TOKEN")
        and os.getenv("GITLAB_BASE_URL")
    )

    return {
        "ollama_available": ollama_available,
        "model": rag_answer.MODEL_NAME,
        "gitlab_configured": gitlab_configured,
        "gitlab_base_url": rag_answer.GITLAB_BASE_URL,
        "default_top_k": rag_answer.TOP_K,
        "max_top_k": MAX_TOP_K,
        "agents": [agent.describe() for agent in list_agents()],
        "default_agent": DEFAULT_AGENT,
    }


@app.get("/api/repos")
def list_repos() -> list[dict]:
    with database() as connection:
        rows = connection.execute(
            "SELECT * FROM repos ORDER BY project_path"
        ).fetchall()

    return [repo_to_dict(row) for row in rows]


@app.post("/api/repos", status_code=201)
def create_repo(body: RepoCreate) -> dict:
    project_path = normalize_project_path(body.project_path)

    with database() as connection:
        existing = connection.execute(
            "SELECT id FROM repos WHERE project_path = ?",
            (project_path,),
        ).fetchone()

        if existing:
            raise HTTPException(409, "This repository is already added.")

        cursor = connection.execute(
            "INSERT INTO repos (project_path, status, message, created_at) "
            "VALUES (?, 'indexing', 'Queued…', ?)",
            (project_path, now()),
        )
        repo_id = cursor.lastrowid
        row = get_repo_row(connection, repo_id)

    start_indexing(repo_id, project_path)

    return repo_to_dict(row)


@app.post("/api/repos/{repo_id}/reindex")
def reindex_repo(repo_id: int) -> dict:
    with database() as connection:
        repo = get_repo_row(connection, repo_id)

        if repo["status"] == "indexing":
            raise HTTPException(409, "This repository is already indexing.")

        connection.execute(
            "UPDATE repos SET status = 'indexing', message = 'Queued…' "
            "WHERE id = ?",
            (repo_id,),
        )
        row = get_repo_row(connection, repo_id)

    start_indexing(repo_id, repo["project_path"])

    return repo_to_dict(row)


@app.delete("/api/repos/{repo_id}", status_code=204)
def delete_repo(repo_id: int) -> None:
    with database() as connection:
        repo = get_repo_row(connection, repo_id)

        if repo["status"] == "indexing":
            raise HTTPException(
                409,
                "Wait for indexing to finish before removing it.",
            )

        connection.execute("DELETE FROM repos WHERE id = ?", (repo_id,))

    forget_index(repo_id)

    # Only remove indexes the server created, never the CLI's data/ files
    if repo["data_dir"]:
        data_directory = Path(repo["data_dir"]).resolve()

        if data_directory.parent == REPOS_DIRECTORY.resolve():
            shutil.rmtree(data_directory, ignore_errors=True)


@app.get("/api/chats")
def list_chats(repo_id: int | None = None) -> list[dict]:
    query = (
        "SELECT chats.*, COUNT(messages.id) AS message_count "
        "FROM chats LEFT JOIN messages ON messages.chat_id = chats.id "
    )
    parameters: tuple = ()

    if repo_id is not None:
        query += "WHERE chats.repo_id = ? "
        parameters = (repo_id,)

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
def create_chat(body: ChatCreate) -> dict:
    timestamp = now()

    with database() as connection:
        get_repo_row(connection, body.repo_id)
        cursor = connection.execute(
            "INSERT INTO chats (repo_id, title, created_at, updated_at) "
            "VALUES (?, 'New chat', ?, ?)",
            (body.repo_id, timestamp, timestamp),
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
def get_chat(chat_id: int) -> dict:
    with database() as connection:
        chat = get_chat_row(connection, chat_id)
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
def rename_chat(chat_id: int, body: ChatUpdate) -> dict:
    with database() as connection:
        get_chat_row(connection, chat_id)
        connection.execute(
            "UPDATE chats SET title = ? WHERE id = ?",
            (body.title.strip(), chat_id),
        )

    return {"id": chat_id, "title": body.title.strip()}


@app.delete("/api/chats/{chat_id}", status_code=204)
def delete_chat(chat_id: int) -> None:
    with database() as connection:
        get_chat_row(connection, chat_id)
        connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def save_message(
    chat_id: int,
    role: str,
    content: str,
    payload: dict | None = None,
) -> dict:
    timestamp = now()

    with database() as connection:
        cursor = connection.execute(
            "INSERT INTO messages (chat_id, role, content, payload, "
            "created_at) VALUES (?, ?, ?, ?, ?)",
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
            "SELECT * FROM messages WHERE chat_id = ? "
            "ORDER BY id DESC LIMIT ?",
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


@app.post("/api/chats/{chat_id}/messages")
def ask_question(chat_id: int, body: QuestionCreate) -> dict:
    """
    Answer a question in a chat with the chosen agent. The question and
    the answer (or the error that prevented one) are both kept in the
    chat history.
    """

    question = body.question.strip()

    if not question:
        raise HTTPException(422, "The question is empty.")

    agent = get_agent(body.agent)

    if agent is None:
        raise HTTPException(422, f"Unknown agent '{body.agent}'.")

    with database() as connection:
        chat = get_chat_row(connection, chat_id)
        repo = get_repo_row(connection, chat["repo_id"])

        if repo["status"] != "ready" or not repo["data_dir"]:
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

    history = load_history(chat_id)

    user_message = save_message(
        chat_id,
        "user",
        question,
        {
            "top_k": body.top_k,
            "retrieval_only": body.retrieval_only,
            "agent": agent.id,
        },
    )

    try:
        context = AgentContext(
            index=get_index(repo),
            project_path=repo["project_path"],
            top_k=body.top_k,
            retrieval_only=body.retrieval_only,
            retrieve_lock=model_lock,
            history=history,
        )
        result = {**agent.run(question, context), "agent": agent.id}
        assistant_message = save_message(
            chat_id,
            "assistant",
            agent.format_text(result),
            result,
        )

    except Exception as error:
        detail = str(error) or error.__class__.__name__

        if isinstance(error, requests.HTTPError):
            detail = f"Ollama returned an error: {error}"

        assistant_message = save_message(
            chat_id,
            "assistant",
            detail,
            {"error": detail, "agent": agent.id},
        )

    return {"messages": [user_message, assistant_message]}


# Serve the built React app (npm run build) from the same server
if FRONTEND_DIRECTORY.exists():
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_DIRECTORY, html=True),
        name="frontend",
    )
