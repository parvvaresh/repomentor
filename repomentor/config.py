"""
Settings read from the environment (and from .env when present).

Every value has a default that works for a single developer running
RepoMentor locally. Read settings through this module instead of calling
os.getenv elsewhere, so the list of options stays in one place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _text(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _number(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def _flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value in (None, ""):
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


# --- Storage ----------------------------------------------------------------

DATA_DIRECTORY = Path(_text("REPOMENTOR_DATA", "data"))

# "numpy" keeps vectors in data/<repo>/embeddings.npy; "pgvector" also
# stores them in PostgreSQL and searches there (needs DATABASE_URL).
VECTOR_STORE = _text("VECTOR_STORE", "numpy").lower()
DATABASE_URL = _text("DATABASE_URL")

# --- Repository sources -----------------------------------------------------

GITLAB_BASE_URL = _text("GITLAB_BASE_URL", "https://gitlab.com").rstrip("/")
GITLAB_TOKEN = _text("GITLAB_TOKEN")
GITLAB_PROJECT_PATH = _text("GITLAB_PROJECT_PATH")
# Optional token with `api` scope, only used to open merge requests
GITLAB_WRITE_TOKEN = _text("GITLAB_WRITE_TOKEN")
GITLAB_WEBHOOK_SECRET = _text("GITLAB_WEBHOOK_SECRET")

GITHUB_API_URL = _text("GITHUB_API_URL", "https://api.github.com").rstrip("/")
GITHUB_WEB_URL = _text("GITHUB_WEB_URL", "https://github.com").rstrip("/")
GITHUB_TOKEN = _text("GITHUB_TOKEN")
GITHUB_WRITE_TOKEN = _text("GITHUB_WRITE_TOKEN")
GITHUB_WEBHOOK_SECRET = _text("GITHUB_WEBHOOK_SECRET")

# Merged merge/pull requests and commits indexed for "why" questions
HISTORY_COMMITS = int(_number("HISTORY_COMMITS", 300))
HISTORY_MERGE_REQUESTS = int(_number("HISTORY_MERGE_REQUESTS", 100))

# --- Indexing ---------------------------------------------------------------

EMBEDDING_MODEL = _text("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# Prefix code chunks with their file path and defined symbols before embedding
EMBEDDING_CONTEXT = _flag("EMBEDDING_CONTEXT", True)
# "auto" uses Tree-sitter for supported languages when it is installed,
# Markdown headings for Markdown and line windows for everything else.
CHUNKER = _text("CHUNKER", "auto").lower()

# --- Retrieval --------------------------------------------------------------

TOP_K = 5
MAX_TOP_K = 20
# semantic | bm25 | hybrid | expanded-hybrid | auto. Semantic was best on
# the development split (see evaluation/results/), so it is the default.
RETRIEVAL_METHOD = _text("RETRIEVAL_METHOD", "semantic").lower()
# Cross-encoder used to rerank the best candidates; empty disables it
RERANKER_MODEL = _text("RERANKER_MODEL")
RERANK_CANDIDATES = int(_number("RERANK_CANDIDATES", 20))

# --- Language models --------------------------------------------------------

OLLAMA_URL = _text("OLLAMA_URL", "http://localhost:11434").rstrip("/")
MODEL_NAME = _text("OLLAMA_MODEL", "qwen3:1.7b")
CODER_MODEL = _text("OLLAMA_CODER_MODEL") or MODEL_NAME
CODER_CONTEXT_LENGTH = int(_number("OLLAMA_CODER_CONTEXT", 16384))
JUDGE_MODEL = _text("OLLAMA_JUDGE_MODEL") or MODEL_NAME

# Questions whose best semantic match is below this cosine similarity are
# refused without calling the model. 0 disables the check.
MIN_RELEVANCE = _number("MIN_RELEVANCE", 0.2)

# --- Web server -------------------------------------------------------------

# "none": everyone sees every repository (single developer).
# "token": each user signs in with their own GitLab or GitHub token and only
# sees repositories that token can read.
AUTH_MODE = _text("AUTH_MODE", "none").lower()
ACCESS_CACHE_SECONDS = int(_number("ACCESS_CACHE_SECONDS", 300))
ALLOW_MERGE_REQUESTS = _flag("ALLOW_MERGE_REQUESTS")
