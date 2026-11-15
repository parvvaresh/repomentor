# --- Frontend -----------------------------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Application --------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REPOMENTOR_DATA=/data \
    HF_HOME=/data/huggingface \
    OLLAMA_URL=http://ollama:11434

# Git is needed for local checkouts and the evaluation benchmarks
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch keeps the image far smaller than the default CUDA build
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt ./
RUN pip install -r requirements.txt "psycopg[binary]" pgvector

COPY repomentor/ repomentor/
COPY evaluation/ evaluation/
COPY --from=frontend /app/frontend/dist frontend/dist

RUN useradd --create-home repomentor \
    && mkdir -p /data \
    && chown repomentor /data
USER repomentor

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=4)"

CMD ["uvicorn", "repomentor.server:app", "--host", "0.0.0.0", "--port", "8000"]
