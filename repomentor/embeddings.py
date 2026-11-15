"""
Chunk embeddings.

Vectors are cached by chunk content: when an index is rebuilt after a push,
only chunks whose text changed are encoded again.
"""

import hashlib
import json
import threading
from pathlib import Path

import numpy as np

from . import config


_models: dict[str, object] = {}
_models_lock = threading.Lock()


def load_embedding_model(name: str | None = None):
    """Load (once per process) a Sentence Transformers model."""

    name = name or config.EMBEDDING_MODEL

    with _models_lock:
        if name not in _models:
            from sentence_transformers import SentenceTransformer

            _models[name] = SentenceTransformer(name)

    return _models[name]


def content_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def embedding_text(chunk: dict) -> str:
    """
    The text embedded for a chunk. With EMBEDDING_CONTEXT on, code chunks
    start with their file path and the names they define, so a chunk of
    bare function bodies still says where it lives and what it is.
    """

    if not config.EMBEDDING_CONTEXT or chunk.get("kind", "code") != "code":
        return chunk["content"]

    header = f"File: {chunk['file_path']}"

    if chunk.get("symbols"):
        header += f"\nDefines: {', '.join(chunk['symbols'][:12])}"

    return f"{header}\n\n{chunk['content']}"


def load_vector_cache(chunks_path: Path, embeddings_path: Path) -> dict[str, np.ndarray]:
    """Map the content hash of each previously embedded chunk to its vector."""

    if not chunks_path.exists() or not embeddings_path.exists():
        return {}

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    embeddings = np.load(embeddings_path)

    if len(chunks) != len(embeddings):
        return {}

    return {
        content_key(embedding_text(chunk)): vector
        for chunk, vector in zip(chunks, embeddings)
    }


def embed_chunks(
    chunks: list[dict],
    model,
    cache: dict[str, np.ndarray] | None = None,
    show_progress_bar: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Return normalized vectors for `chunks` and how many came from `cache`.
    """

    cache = cache or {}
    texts = [embedding_text(chunk) for chunk in chunks]
    keys = [content_key(text) for text in texts]
    missing = [position for position, key in enumerate(keys) if key not in cache]

    encoded = {}

    if missing:
        vectors = model.encode(
            [texts[position] for position in missing],
            batch_size=32,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
        )
        encoded = dict(zip(missing, vectors))

    if not chunks:
        return np.zeros((0, 0), dtype=np.float32), 0

    rows = [
        encoded[position] if position in encoded else cache[keys[position]]
        for position in range(len(chunks))
    ]

    return np.asarray(rows, dtype=np.float32), len(chunks) - len(missing)
