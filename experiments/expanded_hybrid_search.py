import json
import re
from pathlib import Path

import numpy as np
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    expanded = []

    for token in tokens:
        expanded.append(token)
        expanded.extend(token.split("_"))

        if len(token) > 3 and token.endswith("s"):
            expanded.append(token[:-1])

    return expanded


def expand_query(question: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen3:1.7b",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Generate likely code identifiers and technical "
                        "search terms for the user's question. "
                        "Do not answer the question. Return only a short "
                        "space-separated list of search terms."
                    ),
                },
                {"role": "user", "content": question},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
        },
        timeout=120,
    )
    response.raise_for_status()

    return response.json()["message"]["content"].strip()


chunks = []

with Path("data/chunks.jsonl").open("r", encoding="utf-8") as file:
    for line in file:
        chunks.append(json.loads(line))

embeddings = np.load("data/embeddings.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    tokenize(f"{chunk['file_path']}\n{chunk['content']}")
    for chunk in chunks
]

bm25 = BM25Okapi(documents)

question = input("Ask about the codebase: ")
additional_terms = expand_query(question)

print(f"\nExpanded terms: {additional_terms}")

# Use both the original and expanded queries for semantic search
query_embeddings = model.encode(
    [question, additional_terms],
    normalize_embeddings=True,
)

semantic_scores = (embeddings @ query_embeddings.T).max(axis=1)
semantic_ranking = np.argsort(semantic_scores)[::-1]

# Use all terms for keyword search
keyword_query = f"{question} {additional_terms}"
keyword_scores = bm25.get_scores(tokenize(keyword_query))
keyword_ranking = np.argsort(keyword_scores)[::-1]

# Chunks without any matching term have an arbitrary order,
# so they must not receive rank-fusion credit
keyword_ranking = keyword_ranking[keyword_scores[keyword_ranking] > 0]

# Combine both rankings
combined_scores = np.zeros(len(chunks))

for ranking in (semantic_ranking, keyword_ranking):
    for rank, index in enumerate(ranking, start=1):
        combined_scores[index] += 1 / (60 + rank)

best_indices = np.argsort(combined_scores)[::-1][:5]

print("\nExpanded hybrid results:\n")

for rank, index in enumerate(best_indices, start=1):
    chunk = chunks[index]

    print(f"{rank}. {chunk['file_path']}")
    print(f"   Lines: {chunk['start_line']}-{chunk['end_line']}")
    print(f"   Semantic score: {semantic_scores[index]:.3f}")
    print(f"   Keyword score: {keyword_scores[index]:.3f}\n")