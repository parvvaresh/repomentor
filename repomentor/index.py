"""
Building and loading a repository index.

An index is a directory:

    repository_files.jsonl   raw files (full content, used by the coder)
    history.jsonl            commit messages and merge-request descriptions
    chunks.jsonl             retrieval units with citation metadata
    embeddings.npy           one normalized vector per chunk
    summaries.json           optional directory summaries (tour agent)
    manifest.json            what was indexed, from where, and how

`build_index` is incremental: pass the previous directory and unchanged
files are neither downloaded nor embedded again.
"""

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .chunking import chunk_dataset, tree_sitter_available
from .embeddings import embed_chunks, load_embedding_model, load_vector_cache
from .ingest import ingest_history, ingest_repository, read_jsonl
from .retrieval import tokenize
from .sources import RepositorySource
from .vectorstore import NumpyStore, PgVectorStore


RAW_FILE = "repository_files.jsonl"
HISTORY_FILE = "history.jsonl"
CHUNKS_FILE = "chunks.jsonl"
EMBEDDINGS_FILE = "embeddings.npy"
SUMMARIES_FILE = "summaries.json"
MANIFEST_FILE = "manifest.json"

INDEX_FILES = (
    RAW_FILE, HISTORY_FILE, CHUNKS_FILE, EMBEDDINGS_FILE, SUMMARIES_FILE, MANIFEST_FILE,
)

Report = Callable[[str], None]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_manifest(directory: Path) -> dict:
    path = directory / MANIFEST_FILE

    if not path.exists():
        return {}

    return json.loads(path.read_text())


def index_key(directory: Path) -> str:
    """Stable name for an index's rows in a shared vector database."""

    return str(directory.resolve())


def make_store(directory: Path, embeddings: np.ndarray | None, manifest: dict):
    store = manifest.get("vector_store", "numpy")

    if store == "pgvector":
        return PgVectorStore(
            config.DATABASE_URL,
            manifest.get("index_key") or index_key(directory),
            manifest["dimension"],
        )

    if embeddings is None:
        embeddings = np.load(directory / EMBEDDINGS_FILE)

    return NumpyStore(embeddings)


def build_index(
    source: RepositorySource,
    project: str,
    directory: Path,
    previous: Path | None = None,
    include_history: bool = True,
    report: Report | None = None,
    embedding_model=None,
    model_lock=None,
    final_directory: Path | None = None,
) -> dict:
    """
    Ingest, chunk and embed `project` into `directory`. Returns the manifest.
    `model_lock` serializes use of a shared embedding model across threads.
    `final_directory` is where the index will live if it is built elsewhere
    and moved afterwards; it names the index's rows in pgvector.
    """

    def say(message: str) -> None:
        if report:
            report(message)

    directory.mkdir(parents=True, exist_ok=True)
    previous_manifest = read_manifest(previous) if previous else {}

    # A different embedding model or chunker invalidates cached work
    same_embeddings = (
        previous_manifest.get("embedding_model") == config.EMBEDDING_MODEL
        and previous_manifest.get("embedding_context", False) == config.EMBEDDING_CONTEXT
    )

    say("Listing repository files…")

    def progress(position: int, total: int, file_path: str, succeeded: bool):
        if position == total or position % 5 == 0:
            say(f"Downloading changed files ({position}/{total})…")

    ingest = ingest_repository(
        source,
        project,
        directory / RAW_FILE,
        previous / RAW_FILE if previous else None,
        progress,
    )

    if ingest.file_count == 0:
        raise RuntimeError("No supported source files were found in this repository.")

    history_count = 0

    if include_history:
        say("Reading commit and merge request history…")

        try:
            history_count = ingest_history(
                source,
                project,
                directory / HISTORY_FILE,
                config.HISTORY_COMMITS,
                config.HISTORY_MERGE_REQUESTS,
            )
        except Exception as error:  # History is optional; code is not
            say(f"Skipping history: {error}")

    say("Splitting files into chunks…")
    chunk_count, skipped = chunk_dataset(
        directory / RAW_FILE,
        directory / CHUNKS_FILE,
        directory / HISTORY_FILE if history_count else None,
    )

    if chunk_count == 0:
        raise RuntimeError("The repository files produced no chunks.")

    cache = {}

    if previous and same_embeddings:
        cache = load_vector_cache(previous / CHUNKS_FILE, previous / EMBEDDINGS_FILE)

    chunks = read_jsonl(directory / CHUNKS_FILE)
    say(f"Embedding {chunk_count} chunks…")
    model = embedding_model or load_embedding_model()

    if model_lock is None:
        embeddings, reused_vectors = embed_chunks(chunks, model, cache)
    else:
        with model_lock:
            embeddings, reused_vectors = embed_chunks(chunks, model, cache)
    np.save(directory / EMBEDDINGS_FILE, embeddings)

    manifest = {
        "provider": source.provider,
        "project_path": project,
        "web_url": source.web_url(project),
        "head_commit": ingest.head_commit,
        "indexed_at": now(),
        "file_count": ingest.file_count,
        "chunk_count": chunk_count,
        "skipped_files": skipped,
        "history_count": history_count,
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_context": config.EMBEDDING_CONTEXT,
        "dimension": int(embeddings.shape[1]),
        "chunker": config.CHUNKER if tree_sitter_available() else "lines",
        "vector_store": config.VECTOR_STORE,
        "index_key": index_key(final_directory or directory),
        "sync": {
            **ingest.to_dict(),
            "reused_vectors": reused_vectors,
            "incremental": bool(previous_manifest),
        },
    }

    if config.VECTOR_STORE == "pgvector":
        say("Writing vectors to PostgreSQL…")
        PgVectorStore(config.DATABASE_URL, manifest["index_key"], manifest["dimension"]).replace(
            embeddings, [chunk["chunk_id"] for chunk in chunks]
        )

    # Summaries describe directories; keep them unless files changed
    if previous and (previous / SUMMARIES_FILE).exists() and not (
        ingest.changed or ingest.removed
    ):
        shutil.copy(previous / SUMMARIES_FILE, directory / SUMMARIES_FILE)

    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return manifest


@dataclass
class Index:
    directory: Path
    chunks: list[dict]
    store: object
    embedding_model: object
    bm25: BM25Okapi
    manifest: dict = field(default_factory=dict)
    _files: dict | None = None

    def file(self, file_path: str) -> dict | None:
        """The full raw record of a file (content, blob ID, commit)."""

        if self._files is None:
            self._files = {
                record["file_path"]: record
                for record in read_jsonl(self.directory / RAW_FILE)
            }

        return self._files.get(file_path)

    def file_paths(self) -> list[str]:
        self.file("")
        return sorted(self._files)

    @property
    def project_path(self) -> str:
        if self.manifest.get("project_path"):
            return self.manifest["project_path"]

        return self.chunks[0]["project_path"] if self.chunks else ""

    @property
    def provider(self) -> str:
        return self.manifest.get("provider") or self.chunks[0].get("provider", "gitlab")

    def summaries(self) -> dict | None:
        path = self.directory / SUMMARIES_FILE
        return json.loads(path.read_text()) if path.exists() else None


def build_bm25(chunks: list[dict]) -> BM25Okapi:
    return BM25Okapi([
        tokenize(f"{chunk['file_path']}\n{chunk.get('title', '')}\n{chunk['content']}")
        for chunk in chunks
    ])


def load_index(directory: Path, embedding_model=None) -> Index:
    chunks_path = directory / CHUNKS_FILE
    embeddings_path = directory / EMBEDDINGS_FILE

    if not chunks_path.exists() or not embeddings_path.exists():
        raise RuntimeError(
            f"No index in {directory}. Run `python -m repomentor index` first."
        )

    chunks = read_jsonl(chunks_path)
    manifest = read_manifest(directory)

    if not chunks:
        raise RuntimeError(f"No chunks found in {chunks_path}.")

    embeddings = None

    if manifest.get("vector_store", "numpy") == "numpy":
        embeddings = np.load(embeddings_path)

        if len(embeddings) != len(chunks):
            raise RuntimeError("Chunks and embeddings do not match. Rebuild the index.")

    return Index(
        directory=directory,
        chunks=chunks,
        store=make_store(directory, embeddings, manifest),
        embedding_model=embedding_model or load_embedding_model(
            manifest.get("embedding_model")
        ),
        bm25=build_bm25(chunks),
        manifest=manifest,
    )
