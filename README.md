# RepoMentor

**A local, citation-grounded RAG assistant for understanding unfamiliar GitLab codebases.**

RepoMentor helps new developers explore a company’s repositories without reading every file manually. It ingests source code and documentation from GitLab, retrieves the code most relevant to a natural-language question, and uses a local language model to produce an evidence-backed answer with links to exact files, commits, and line ranges.

> **Project status:** Working research prototype with a command-line interface, a React web chat, reproducible retrieval ablations and manually reviewed answer-quality evaluation. The current priority is improving retrieval selection and grounded generation.

![RepoMentor web interface: a chat about a repository with cited facts and their sources](docs/screenshots/web-chat.jpg)

## Evaluation Snapshot

RepoMentor is evaluated in two stages because strong retrieval does not automatically produce a correct answer.

| Evaluation layer | Current result | Benchmark |
| --- | ---: | --- |
| Best retrieval Hit@5 | **93.8%** | Semantic search, 16 questions |
| Best retrieval Recall@5 | **90.6%** | Semantic search, 16 questions |
| Best retrieval MRR@5 | **76.5%** | Semantic search, 16 questions |
| Gold-fact coverage | **27.6%** | Local generation, 10 questions / 29 facts |
| Faithful substantive answers | **50.0%** | Human-reviewed |
| Citation-correct substantive answers | **50.0%** | Human-reviewed |
| Unanswerable-question abstention | **100.0%** | 2 unanswerable questions |

These are development-set results on one demonstration repository, not claims of general performance. They show that retrieval is promising, while the small local generator remains the main bottleneck.

## Why RepoMentor?

Joining a software team often requires learning several things at once:

- where important logic lives;
- how services and components interact;
- which conventions the team follows;
- how authentication, deployment, testing, and data access work;
- and why parts of the system changed over time.

README files help, but they rarely capture the full implementation. Generic chatbots can also produce plausible answers without sufficient evidence. RepoMentor addresses this by retrieving relevant repository content first and grounding every returned fact in a specific source location.

## What It Does

- Connects to private or public GitLab repositories through the GitLab REST API.
- Discovers and downloads supported source-code and documentation files.
- Excludes virtual environments, dependencies, generated files, build output, empty files, and other low-value content.
- Preserves project paths, blob IDs, commit IDs, file sizes, and source content.
- Splits files into overlapping, line-aware chunks for retrieval and citation.
- Generates embeddings locally with Sentence Transformers.
- Combines semantic similarity and BM25 keyword search.
- Expands searches deterministically using identifiers discovered in initially retrieved code.
- Uses Reciprocal Rank Fusion to merge semantic and keyword rankings.
- Reduces duplicate evidence by selecting chunks from different files.
- Uses a local Ollama model to extract facts from one source at a time.
- Appends citations programmatically instead of relying on the model to map citation numbers correctly.
- Links citations to the exact GitLab commit and line range.
- Compares semantic, BM25, hybrid, and expanded-hybrid retrieval through an ablation study.
- Evaluates answer completeness, faithfulness, citation support, and abstention using a human-reviewed gold set.

## Architecture

```mermaid
flowchart TD
    A[GitLab repository] --> B[Repository ingestion]
    B --> C[Filtering and metadata]
    C --> D[Line-aware chunking]
    D --> E[Local embeddings]
    D --> F[BM25 index]
    E --> G[Semantic retrieval]
    F --> H[Keyword retrieval]
    G --> I[Retrieval configuration]
    H --> I
    I --> J[Optional fusion and expansion]
    J --> K[Source-by-source extraction]
    K --> L[Cited answer]
    I --> M[Retrieval evaluation]
    L --> N[Answer evaluation]
```

## RAG Pipeline

### 1. Ingestion

RepoMentor authenticates to GitLab using a read-only token, lists the repository tree, downloads selected files, and stores the raw corpus as JSON Lines (`.jsonl`). Each record contains the file content and traceability metadata.

Example record:

```json
{
  "project_path": "username/demo-company-app",
  "file_path": "backend/app/core/security.py",
  "blob_id": "...",
  "last_commit_id": "...",
  "size": 1234,
  "content": "..."
}
```

### 2. Filtering

The current prototype accepts:

- Python;
- JavaScript and TypeScript;
- React TSX/JSX;
- Markdown;
- YAML;
- TOML;
- SQL;
- Dockerfiles.

It excludes directories such as `node_modules`, `.venv`, `dist`, `build`, `coverage`, and `vendor`. During processing, it also skips empty files, generated TypeScript clients, and oversized release-note content that could dominate retrieval.

### 3. Chunking

Files are split into chunks of up to **60 lines** with a **10-line overlap**. The overlap preserves context when an important function or explanation crosses a chunk boundary.

Each chunk retains:

- project path;
- file path;
- start and end line numbers;
- last commit ID;
- deterministic chunk ID;
- chunk content.

The initial sample run produced:

| Metric | Value |
| --- | ---: |
| Downloaded files | 188 |
| Download failures | 0 |
| Raw corpus size | 610.85 KB |
| Generated chunks | 324 |
| Skipped files | 26 |

### 4. Embeddings

RepoMentor uses `sentence-transformers/all-MiniLM-L6-v2` to represent each chunk as a **384-dimensional normalized vector**. The current corpus therefore produces an embedding matrix with shape `(324, 384)`.

For the prototype, embeddings are stored as a NumPy array. This keeps the retrieval mathematics transparent before introducing a dedicated vector database.

### 5. Retrieval

RepoMentor implements four retrieval configurations for controlled comparison:

- **Semantic retrieval:** compares the question embedding with chunk embeddings using cosine similarity.
- **BM25 retrieval:** finds exact names and technical terms such as `verify_password` or `get_password_hash`.
- **Hybrid retrieval:** combines semantic and BM25 rankings with Reciprocal Rank Fusion (RRF).
- **Expanded hybrid retrieval:** extracts identifiers from initial matches and performs a deterministic second pass.

The ablation study produced an important result: on the current 16-question natural-language benchmark, **semantic retrieval alone performed best**. BM25 was weak, and equal-weight fusion sometimes pushed relevant semantic matches down the ranking. The hybrid methods remain useful experiments, especially for future identifier-heavy questions, but the README does not treat extra complexity as an automatic improvement.

### 6. Grounded Generation

The current generator runs locally through Ollama using `qwen3:1.7b`. To reduce source confusion in a small model:

1. each retrieved source is processed independently;
2. the model extracts only facts directly supported by that source;
3. Python attaches the source citation automatically;
4. the final output links to the exact GitLab file, commit, and line range.

This design reduces unsupported claims, but it does **not** guarantee zero hallucinations. Retrieval and answer quality still require systematic evaluation.

## Example

Question:

```text
How are user passwords handled?
```

Example grounded output:

```text
- User passwords are hashed through get_password_hash before storage.
  [backend/app/crud.py:1-60]

- Password hashing is configured with Argon2Hasher and BcryptHasher.
  [backend/app/core/security.py:1-36]
```

Each citation is followed by a GitLab URL pinned to the relevant commit and line range.

## Evaluation Results

### Retrieval ablation

Sixteen manually labeled onboarding questions were evaluated at multiple values of K. At `K=5`:

| Method | Hit@5 | Recall@5 | MRR@5 |
| --- | ---: | ---: | ---: |
| Semantic | **0.938** | **0.906** | **0.765** |
| BM25 | 0.500 | 0.406 | 0.226 |
| Hybrid | 0.875 | 0.781 | 0.729 |
| Expanded hybrid | 0.875 | 0.844 | 0.669 |

At `K=10`, semantic retrieval reached `1.000` Hit@10 and Recall@10. Expanded hybrid reached `1.000` Hit@10 and `0.969` Recall@10. Error analysis showed that identifier expansion occasionally helped multi-file password questions, but also caused query drift toward related backend files.

### Answer quality

Ten questions were evaluated: eight answerable questions and two deliberately unanswerable questions. The gold set contains 29 required facts and known incorrect claims. Generated answers were manually reviewed rather than judged by the same model that produced them.

| Metric | Result | Interpretation |
| --- | ---: | --- |
| Gold-fact coverage | 8/29 (**27.6%**) | Required facts stated accurately and with support |
| Complete-answer rate | 0/8 (**0.0%**) | Answerable questions containing every required fact |
| Answerable response rate | 6/8 (**75.0%**) | Questions answered instead of incorrectly refused |
| Unanswerable abstention accuracy | 2/2 (**100.0%**) | Unsupported questions correctly refused |
| Faithful substantive-answer rate | 3/6 (**50.0%**) | Non-refusal answers with no unsupported claims |
| Citation-correct substantive-answer rate | 3/6 (**50.0%**) | Non-refusal answers whose citations supported their claims |
| Known-error trigger rate | 2/22 (**9.1%**) | Predefined incorrect claims that appeared |
| Mean expected-source recall | **56.2%** | Expected evidence files cited in final answers |

The evaluation exposed cases where the system cited a real file but described it incorrectly. Citations improve traceability; they do not prove that an interpretation is correct.

## Technology Stack

| Area | Technology |
| --- | --- |
| Language | Python |
| Repository source | GitLab REST API |
| HTTP client | Requests |
| Configuration | python-dotenv |
| Embeddings | Sentence Transformers |
| Embedding model | `all-MiniLM-L6-v2` |
| Vector operations | NumPy |
| Keyword retrieval | BM25 (`rank-bm25`) |
| Rank fusion | Reciprocal Rank Fusion |
| Local LLM runtime | Ollama |
| Current local LLM | `qwen3:1.7b` |
| Raw/intermediate format | JSON Lines and NumPy arrays |
| Web API | FastAPI + SQLite (chat history) |
| Web interface | React + TypeScript (Vite) |

## Project Structure

```text
repomentor/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── test_connection.py
├── list_repository_files.py
├── fetch_one_file.py
├── ingest_repository.py
├── inspect_dataset.py
├── chunk_dataset.py
├── create_embeddings.py
├── semantic_search.py
├── hybrid_search.py
├── expanded_hybrid_search.py
├── rag_answer.py
├── evaluate_retrieval.py
├── evaluate_ablation.py
├── analyze_retrieval_errors.py
├── evaluate_answers.py
├── review_answers.py
├── score_answers.py
├── server.py                  # FastAPI backend for the web interface
├── agents/                    # Pluggable chat agents (mentor, coder, …)
├── frontend/                  # React chat interface (Vite + TypeScript)
├── evaluation/
│   ├── retrieval_questions.example.json
│   ├── retrieval_questions.json
│   ├── answer_questions.json
│   ├── ablation_results.txt
│   ├── retrieval_error_analysis.txt
│   ├── generated_answers.json
│   └── answer_evaluation_results.json
└── data/                       # Generated locally; ignored by Git
    ├── repository_files.jsonl
    ├── chunks.jsonl
    ├── embeddings.npy
    ├── repomentor.db           # Web repositories and chat history
    └── repos/<id>/             # One index per repository added on the web
```

Some scripts document intermediate experiments used to develop the final retrieval approach. A future refactor will move reusable logic into a Python package and separate production commands from experiments.

## Getting Started

### Prerequisites

- Python 3.10 or newer;
- Git;
- a GitLab account and repository;
- a GitLab personal access token with read-only API access;
- Ollama installed and running locally.

### 1. Clone the repository

```bash
git clone https://github.com/amir-hri/repomentor.git
cd repomentor
```

### 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install and prepare Ollama

Install Ollama from [ollama.com](https://ollama.com), then download the current local model:

```bash
ollama pull qwen3:1.7b
```

Verify that it runs:

```bash
ollama run qwen3:1.7b
```

Exit the interactive prompt with `/bye`. Keep the Ollama application or service running while using RepoMentor.

### 5. Configure GitLab access

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GITLAB_TOKEN=your_read_only_gitlab_token
GITLAB_BASE_URL=https://gitlab.com
GITLAB_PROJECT_PATH=your-username/your-project
# Optional: any chat model pulled into Ollama (default qwen3:1.7b)
OLLAMA_MODEL=qwen3:1.7b
# Optional: a separate model for the coder agent (default OLLAMA_MODEL)
OLLAMA_CODER_MODEL=qwen2.5-coder:7b
```

Never commit `.env` or paste its token into logs, screenshots, issues, or documentation.

### 6. Test the GitLab connection

```bash
python test_connection.py
```

### 7. Ingest the repository

```bash
python ingest_repository.py
```

This creates:

```text
data/repository_files.jsonl
```

### 8. Inspect the corpus

```bash
python inspect_dataset.py
```

Review file counts, file types, empty files, total size, and the largest files before creating an index.

### 9. Create chunks

```bash
python chunk_dataset.py
```

This creates:

```text
data/chunks.jsonl
```

### 10. Create embeddings

```bash
python create_embeddings.py
```

The first run downloads the embedding model. The generated vectors are stored in:

```text
data/embeddings.npy
```

### 11. Ask a question

```bash
python rag_answer.py
```

This starts an interactive session. The index and embedding model are loaded once, so you can ask several questions in a row. Press Enter or type `exit` to quit.

```text
Ask RepoMentor: How does user authentication work?
```

You can also pass a question directly and adjust retrieval:

```bash
# Answer a single question and exit
python rag_answer.py "How are user passwords handled?"

# Retrieve 8 source files instead of 5
python rag_answer.py "How does deployment work?" --top-k 8

# Show retrieved sources only, without calling Ollama
python rag_answer.py "Where are database models defined?" --retrieval-only
```

### 12. Evaluate retrieval

Create `evaluation/retrieval_questions.json` from the example file and list, for each question, the files that should be retrieved:

```bash
cp evaluation/retrieval_questions.example.json evaluation/retrieval_questions.json
python evaluate_retrieval.py --top-k 5
```

```json
[
  {
    "question": "How are user passwords handled?",
    "expected_files": ["backend/app/core/security.py", "backend/app/crud.py"]
  }
]
```

The script runs the same retrieval pipeline as `rag_answer.py` (no LLM calls) and reports file-level **Hit@K**, **Recall@K**, and **MRR@K**, plus the expected files each question missed. Use it to compare chunking, embedding, and ranking changes objectively.

Compare all retrieval configurations:

```bash
python evaluate_ablation.py
python analyze_retrieval_errors.py
```

The ablation script compares semantic, BM25, hybrid, and expanded-hybrid retrieval. Error analysis shows which questions improve or regress, rather than relying only on averages.

### 13. Evaluate generated answers

Answer evaluation uses a gold set containing required facts, expected sources, known incorrect claims, and unanswerable questions:

```bash
python evaluate_answers.py
python review_answers.py
python score_answers.py
```

- `evaluate_answers.py` generates and stores answers for the benchmark.
- `review_answers.py` collects human labels for fact coverage, unsupported claims, and citation support.
- `score_answers.py` aggregates the reviewed labels into reproducible metrics.

The local model is not used as its own factual judge. Human review is slower, but it avoids reporting self-evaluation as objective ground truth.

### 14. Use the web interface (optional)

The web interface lets you add several GitLab repositories, pick one, ask questions in a chat, and keeps every conversation in `data/repomentor.db`.

Install the frontend once (requires Node.js 18+):

```bash
cd frontend
npm install
cd ..
```

**Development** — run the API and the Vite dev server in two terminals:

```bash
uvicorn server:app --reload          # API on http://127.0.0.1:8000
cd frontend && npm run dev           # UI on http://localhost:5173
```

**Single server** — build the UI once and let FastAPI serve it:

```bash
cd frontend && npm run build && cd ..
uvicorn server:app                   # open http://127.0.0.1:8000
```

![Each source can be expanded to show the exact lines the answer was based on](docs/screenshots/web-source-code.jpg)

What the interface offers:

- **Repository picker:** add several GitLab projects; each one is indexed in the background and can be re-indexed or removed.
- **Chat history:** every conversation is saved per repository in SQLite and can be reopened, renamed or deleted.
- **Cited answers:** each fact has a numbered citation; hovering it highlights the source, and each source opens the exact commit and lines in GitLab or expands to show the code.
- **Agents:** pick an agent per message from the composer. **Mentor** gives cited, fact-by-fact answers; **Coder** writes or changes code that follows the repository's own patterns and links every source it builds on.
- **Retrieval options:** choose how many source files to retrieve, or switch to retrieval-only mode to inspect sources without calling Ollama.

In the sidebar, choose **+ Add** and enter a project path such as `group/project` (a pasted GitLab URL also works). The server downloads, chunks and embeds it in the background using the GitLab settings from `.env`. An index already built with the command-line scripts (`data/chunks.jsonl`) appears automatically. Use the refresh button to re-index a repository after it changes.

#### Agents

Each message is handled by an agent from the `agents/` package. The web interface lists every registered agent, and the chat history records which agent answered each message.

| Agent | What it does |
| --- | --- |
| `mentor` | The original RepoMentor pipeline: hybrid retrieval, then facts extracted one source at a time with citations attached in Python. |
| `coder` | Retrieves the most relevant sources, gives them to the model with numbered labels and asks for a plan plus code that reuses the repository's functions and style. It sees the earlier turns of the chat, so follow-ups such as "now add tests" work. |

![Choosing the Coder agent in the composer shows its description and example tasks](docs/screenshots/web-agents.jpg)

![A Coder answer: code blocks with a copy button, followed by the repository sources it was given as context](docs/screenshots/web-coder.jpg)

The answer above was produced by `qwen2.5:1.5b`. A model this small often ignores the requested format and writes code that does not fully fit the repository, so a dedicated code model (see `OLLAMA_CODER_MODEL` below) is recommended for real use.

The coder benefits from a larger code model than the default. Optional settings in `.env`:

```env
OLLAMA_CODER_MODEL=qwen2.5-coder:7b   # defaults to OLLAMA_MODEL
OLLAMA_CODER_CONTEXT=16384            # context window passed to Ollama
```

The coder only proposes code in the chat; it never changes the repository.

**Adding an agent.** Create a module in `agents/` with a subclass of `Agent` and register it in `agents/__init__.py`:

```python
# agents/reviewer.py
import rag_answer
from .base import Agent, AgentContext


class ReviewerAgent(Agent):
    id = "reviewer"                       # stored with messages; never rename
    name = "Reviewer"
    description = "Reviews code for bugs and risky patterns."
    examples = ["Review the authentication flow for security issues."]
    working_label = "Reviewing code…"

    def run(self, question: str, context: AgentContext) -> dict:
        indices, identifiers = rag_answer.retrieve(
            question, context.index, context.top_k,
        )
        chunks = [context.index["chunks"][i] for i in indices]
        answer = rag_answer.call_ollama([...])   # your prompt here

        return {
            "answer": answer,              # Markdown; [n] cites sources[n-1]
            "sources": [rag_answer.describe_source(c) for c in chunks],
        }
```

```python
# agents/__init__.py
from .reviewer import ReviewerAgent
register(ReviewerAgent())
```

An agent without a dedicated view in `frontend/src/components/AssistantMessage.tsx` is rendered as Markdown with its sources, so a new agent works in the interface without frontend changes.

## Configuration and Security

RepoMentor is designed around least-privilege access:

- use a GitLab token with read-only API scope;
- store credentials only in `.env`;
- keep `.env`, `.venv`, `data/`, caches, and compiled Python files out of Git;
- do not commit downloaded private source code or generated embeddings;
- rotate the GitLab token immediately if it is ever exposed;
- review staged files with `git status` and `git diff --cached` before every public push.

Recommended `.gitignore` entries:

```gitignore
.env
.venv/
data/
__pycache__/
*.pyc
.DS_Store
```

## Design Decisions

### Why a local LLM?

Local generation keeps retrieved source code on the developer’s machine and avoids per-request API charges. The tradeoff is lower answer quality and slower inference on limited hardware compared with larger hosted models.

### Why NumPy instead of a vector database?

The sample corpus contains only a few hundred chunks, so exhaustive cosine similarity is fast and easy to inspect. A vector database becomes valuable when scaling to many repositories, millions of chunks, concurrent users, metadata filtering, and incremental updates.

### Why hybrid retrieval?

Hybrid retrieval was implemented to test a reasonable code-search hypothesis: semantic search handles meaning, while BM25 handles exact identifiers such as `get_password_hash`. The ablation results did not support equal-weight fusion on the current natural-language benchmark; semantic search performed better. The hybrid implementation is retained as an experiment and a baseline for future identifier-heavy evaluation, not presented as the final answer by default.

### Why attach citations in Python?

Small models can cite the wrong source number even when the correct evidence was retrieved. RepoMentor processes one source at a time and attaches citations programmatically, reducing citation-mapping errors.

## Current Limitations

- Chunking is line-based rather than AST-aware.
- The embedding model is general-purpose rather than code-specific.
- NumPy search scans every chunk and is not intended for large deployments.
- Repository updates require manually rerunning ingestion and indexing.
- The current generator is a small local model with limited reasoning ability.
- Fact extraction can still misinterpret source code.
- Access control currently depends on the configured GitLab token rather than individual application users.
- The command-line scripts index one configured repository per run; the web interface supports several.
- Follow-up questions in a chat are answered independently; earlier messages are not used as context.
- Evaluation uses small, manually labeled development sets from one repository.
- Answer review is intentionally human-in-the-loop and is not yet suitable for large benchmarks.
- The current expanded-hybrid production path is not the strongest method on the present retrieval benchmark.

## Roadmap

- [ ] Refactor ingestion, chunking, retrieval, and generation into reusable modules.
- [x] Add retrieval evaluation with Hit@K, Recall@K, and expected-file ground truth.
- [x] Add retrieval ablation and per-question error analysis.
- [x] Add answer-level completeness, citation, faithfulness, and abstention evaluation.
- [ ] Select retrieval dynamically or make semantic retrieval the default after held-out validation.
- [ ] Improve evidence extraction and refusal logic using the answer-level error analysis.
- [ ] Expand the benchmark and separate development and held-out test questions.
- [ ] Add code-aware chunking with Tree-sitter.
- [ ] Add a reranking model for top retrieval candidates.
- [ ] Replace the NumPy prototype with PostgreSQL/pgvector or another vector store.
- [ ] Add incremental GitLab synchronization using commits and webhooks.
- [x] Support multiple repositories (web interface).
- [ ] Respect per-user GitLab permissions in a multi-user deployment.
- [x] Add a FastAPI backend and a web chat interface with history.
- [ ] Add Docker-based local deployment.
- [ ] Add automated tests and continuous integration.

## Evaluation Methodology

Retrieval and generation are measured separately so failures can be traced to the correct stage.

Retrieval metrics:

- **Hit@K:** whether at least one expected file appears in the top K results;
- **Recall@K:** the fraction of expected files present in the top K results;
- **Mean Reciprocal Rank:** how early the first relevant file appears;
- **source diversity:** whether retrieval returns complementary implementation evidence instead of repeated chunks from one file.

Generation metrics include gold-fact coverage, complete-answer rate, answerable response rate, unanswerable abstention accuracy, faithfulness, citation correctness, expected-source recall, and known-error triggers. The current numbers are development baselines intended to guide engineering decisions, not leaderboard-style claims.

## Demonstration Corpus

Development currently uses the public [Full Stack FastAPI Template](https://github.com/fastapi/full-stack-fastapi-template) as a demonstration codebase. The template is MIT-licensed. Its source code and generated embeddings are not included in this repository; users ingest their own authorized GitLab repository locally.

## Responsible Use

RepoMentor is a developer-assistance tool, not an authority on repository behavior. Users should open and verify cited code before making security, architecture, deployment, or production decisions. A citation shows which source informed an answer; it does not by itself prove that the interpretation is correct or complete.

## Contributing

The project is currently an individual portfolio project and is evolving rapidly. Bug reports, evaluation cases, retrieval improvements, and documentation suggestions are welcome through GitHub issues.

---

Built as a hands-on exploration of retrieval-augmented generation, code search, local language models, and evidence-grounded developer tooling.
