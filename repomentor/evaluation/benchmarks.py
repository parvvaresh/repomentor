"""
Retrieval benchmarks.

A benchmark pins a public repository at an exact commit and lists labeled
questions:

    {
      "name": "flask",
      "repository": {"url": "https://github.com/pallets/flask.git",
                     "commit": "49943b5..."},
      "questions": [
        {"id": "sessions", "question": "...", "expected_files": ["..."],
         "split": "dev" | "test", "type": "natural" | "identifier"}
      ]
    }

Use the "dev" split to tune retrieval and the "test" split only to
confirm a decision, so the reported numbers are not tuned to the
questions they are measured on.

`prepare` clones the repository at the pinned commit and builds an index
with the local Git source, so results are reproducible offline.
"""

import json
import subprocess
from pathlib import Path

from .. import config


BENCHMARK_DIRECTORY = Path("evaluation/benchmarks")
SPLITS = ("dev", "test", "all")


def benchmark_paths() -> list[Path]:
    return sorted(BENCHMARK_DIRECTORY.glob("*.json"))


def load_benchmark(path: Path) -> dict:
    benchmark = json.loads(path.read_text())

    for position, case in enumerate(benchmark["questions"], start=1):
        if not case.get("question") or not case.get("expected_files"):
            raise ValueError(
                f"{path}: question {position} needs 'question' and 'expected_files'."
            )

        case.setdefault("split", "dev")
        case.setdefault("type", "natural")

    return benchmark


def select(benchmark: dict, split: str = "all", kind: str | None = None) -> list[dict]:
    return [
        case for case in benchmark["questions"]
        if (split == "all" or case["split"] == split)
        and (kind is None or case["type"] == kind)
    ]


def benchmark_directory(benchmark: dict, root: Path | None = None) -> Path:
    return (root or config.DATA_DIRECTORY / "benchmarks") / benchmark["name"]


def checkout(benchmark: dict, root: Path | None = None) -> Path:
    """Clone (once) and check out the pinned commit. Returns the checkout."""

    directory = benchmark_directory(benchmark, root) / "checkout"
    repository = benchmark["repository"]

    if not (directory / ".git").exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none", repository["url"], str(directory)],
            check=True,
        )

    subprocess.run(
        ["git", "-C", str(directory), "checkout", "--quiet", "--detach", repository["commit"]],
        check=True,
    )
    return directory


def index_directory(benchmark: dict, root: Path | None = None, variant: str = "") -> Path:
    name = f"index-{variant}" if variant else "index"
    return benchmark_directory(benchmark, root) / name


def prepare(benchmark: dict, root: Path | None = None, variant: str = "", report=print) -> Path:
    """Build (or reuse) the index for a benchmark. Returns its directory."""

    from ..index import MANIFEST_FILE, build_index, read_manifest
    from ..sources import LocalGitSource

    directory = index_directory(benchmark, root, variant)
    manifest = read_manifest(directory) if (directory / MANIFEST_FILE).exists() else {}

    if (
        manifest.get("head_commit") == benchmark["repository"]["commit"]
        and manifest.get("embedding_model") == config.EMBEDDING_MODEL
        and manifest.get("chunker_setting", config.CHUNKER) == config.CHUNKER
        and manifest.get("embedding_context", False) == config.EMBEDDING_CONTEXT
    ):
        return directory

    project = str(checkout(benchmark, root))
    build_index(
        LocalGitSource(),
        project,
        directory,
        include_history=False,
        report=report,
    )

    # Remember the chunker setting the index was built with
    manifest = read_manifest(directory)
    manifest["chunker_setting"] = config.CHUNKER
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))

    missing = sorted({
        path for case in benchmark["questions"] for path in case["expected_files"]
        if not (Path(project) / path).exists()
    })

    if missing:
        raise ValueError(f"Expected files missing at the pinned commit: {missing}")

    return directory
