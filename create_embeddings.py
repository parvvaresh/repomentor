import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_CHUNKS_PATH = Path("data/chunks.jsonl")
DEFAULT_EMBEDDINGS_PATH = Path("data/embeddings.npy")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def create_embeddings(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
    model: SentenceTransformer | None = None,
    show_progress_bar: bool = True,
) -> tuple[int, ...]:
    """
    Embed every chunk and save the normalized vectors.
    Pass an already loaded `model` to avoid loading it again.
    Returns the shape of the saved embedding matrix.
    """

    chunks = []

    with chunks_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    texts = [chunk["content"] for chunk in chunks]

    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
    )

    np.save(embeddings_path, embeddings)

    return embeddings.shape


def main() -> None:
    shape = create_embeddings()

    print(f"Chunks embedded: {shape[0]}")
    print(f"Embedding shape: {shape}")
    print(f"Saved to: {DEFAULT_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
