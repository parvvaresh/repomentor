import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

# Links must point at the configured (possibly self-hosted) GitLab instance
gitlab_url = os.getenv("GITLAB_BASE_URL", "https://gitlab.com").rstrip("/")

chunks_path = Path("data/chunks.jsonl")
embeddings_path = Path("data/embeddings.npy")

chunks = []

with chunks_path.open("r", encoding="utf-8") as file:
    for line in file:
        chunks.append(json.loads(line))

embeddings = np.load(embeddings_path)

model = SentenceTransformer("all-MiniLM-L6-v2")

question = input("Ask about the codebase: ")

question_embedding = model.encode(
    question,
    normalize_embeddings=True,
)

# Because vectors are normalized, dot product gives cosine similarity
similarity_scores = embeddings @ question_embedding

top_k = 5
best_indices = np.argsort(similarity_scores)[::-1][:top_k]

print("\nMost relevant chunks:\n")

for rank, index in enumerate(best_indices, start=1):
    chunk = chunks[index]
    score = similarity_scores[index]

    encoded_path = quote(chunk["file_path"])
    source_url = (
        f"{gitlab_url}/{chunk['project_path']}/-/blob/"
        f"{chunk['last_commit_id']}/{encoded_path}"
        f"#L{chunk['start_line']}-{chunk['end_line']}"
    )

    preview = chunk["content"][:500].replace("\n", " ")

    print(f"{rank}. Similarity: {score:.3f}")
    print(
        f"   Source: {chunk['file_path']}"
        f":{chunk['start_line']}-{chunk['end_line']}"
    )
    print(f"   URL: {source_url}")
    print(f"   Preview: {preview}\n")