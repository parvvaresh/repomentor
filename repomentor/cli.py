"""
Command line: python -m repomentor <command>

    index       download, chunk and embed a repository (incremental)
    ask         ask a question (interactive when no question is given)
    summarize   build directory summaries for the tour agent
    inspect     statistics about an index
    serve       run the web interface
    eval        evaluation commands (see `python -m repomentor eval -h`)
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

from . import config


def add_data_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        type=Path,
        default=config.DATA_DIRECTORY,
        help=f"Index directory (default: {config.DATA_DIRECTORY}).",
    )


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

def command_index(arguments) -> None:
    import shutil

    from .index import INDEX_FILES, MANIFEST_FILE, RAW_FILE, build_index
    from .sources import get_source

    project = arguments.project

    if arguments.provider == "gitlab":
        project = project or config.GITLAB_PROJECT_PATH

    if arguments.provider == "local":
        project = str(Path(project or ".").resolve())

    if not project:
        sys.exit("Give --project (or set GITLAB_PROJECT_PATH in .env).")

    directory = arguments.data
    directory.mkdir(parents=True, exist_ok=True)
    previous = None if arguments.full or not (directory / RAW_FILE).exists() else directory

    # Build next to the old index and swap at the end, so a failed sync
    # keeps the old one. Other files in the directory (such as the web
    # database in data/) are left alone.
    build = directory.with_name(directory.name + ".building")
    shutil.rmtree(build, ignore_errors=True)

    manifest = build_index(
        get_source(arguments.provider),
        project,
        build,
        previous=previous,
        include_history=not arguments.no_history,
        report=print,
        final_directory=directory,
    )

    for name in INDEX_FILES:
        target = directory / name

        if target.exists():
            target.unlink()

        if (build / name).exists():
            (build / name).rename(target)

    shutil.rmtree(build, ignore_errors=True)

    sync = manifest["sync"]
    print("\nIndex ready.")
    print(f"Repository: {manifest['provider']}:{manifest['project_path']} @ {manifest['head_commit'][:10]}")
    print(f"Files: {manifest['file_count']} ({sync['downloaded']} downloaded, {sync['reused']} unchanged)")
    print(f"Chunks: {manifest['chunk_count']} ({sync['reused_vectors']} embeddings reused)")
    print(f"History records: {manifest['history_count']}")
    print(f"Saved to: {directory / MANIFEST_FILE}")


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------

def print_result(agent, result: dict) -> None:
    sources = result.get("sources", [])

    if result.get("related_identifiers"):
        print("\nRelated code identifiers:")
        print(", ".join(result["related_identifiers"]))

    print("\nAnswer:\n")
    print(agent.format_text(result))

    for change in result.get("changes", []):
        print(f"\n{change['path']} ({change['status']}, {change['check']['message']})")
        print(change["diff"])

    if sources:
        print("\nSources:\n")

        for source in sources:
            print(source["citation"])
            print(source["url"])
            print()


def command_ask(arguments) -> None:
    from .agents import AgentContext, get_agent
    from .index import load_index
    from .sources import get_source

    index = load_index(arguments.data)
    agent = get_agent(arguments.agent)

    if agent is None:
        sys.exit(f"Unknown agent '{arguments.agent}'.")

    if arguments.method:
        config.RETRIEVAL_METHOD = arguments.method

    def ask(question: str) -> None:
        context = AgentContext(
            index=index,
            project_path=index.project_path,
            top_k=arguments.top_k,
            retrieval_only=arguments.retrieval_only,
            source=get_source(index.provider),
            source_project=index.project_path,
        )
        result = None

        for event in agent.stream(question, context):
            if event["type"] == "status":
                print(f"… {event['label']}", file=sys.stderr)
            elif event["type"] == "delta":
                print(event["text"], end="", flush=True)
            elif event["type"] == "result":
                result = event["result"]

        print_result(agent, result)

    if arguments.question:
        ask(arguments.question.strip())
        return

    # Interactive session: the index is loaded once for all questions
    print("Type a question, or press Enter / type 'exit' to quit.")

    while True:
        try:
            question = input("\nAsk RepoMentor: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question or question.lower() in {"exit", "quit"}:
            break

        ask(question)


# ---------------------------------------------------------------------------
# summarize, inspect, serve
# ---------------------------------------------------------------------------

def command_summarize(arguments) -> None:
    from .index import load_index
    from .summaries import build_summaries

    index = load_index(arguments.data)

    for event in build_summaries(index):
        if event["type"] == "status":
            print(event["label"])
        else:
            summaries = event["summaries"]
            print(f"\n{summaries['overview']}\n")

            for number, step in enumerate(summaries["learning_path"], start=1):
                print(f"{number}. {step['path']} - {step['why']}")


def command_inspect(arguments) -> None:
    from .index import CHUNKS_FILE, RAW_FILE, read_manifest
    from .ingest import read_jsonl

    records = read_jsonl(arguments.data / RAW_FILE)
    chunks = read_jsonl(arguments.data / CHUNKS_FILE)
    manifest = read_manifest(arguments.data)

    if manifest:
        print(json.dumps({k: v for k, v in manifest.items() if k != "sync"}, indent=2))

    sizes = sorted(records, key=lambda record: -record["size"])
    print(f"\nFiles: {len(records)}  ({sum(r['size'] for r in records) / 1024:.1f} KB)")
    print("By extension:", dict(Counter(
        PurePosixPath(r["file_path"]).suffix or PurePosixPath(r["file_path"]).name for r in records
    ).most_common()))
    print("Largest files:")

    for record in sizes[:10]:
        print(f"  {record['size'] / 1024:7.1f} KB  {record['file_path']}")

    print(f"\nChunks: {len(chunks)}")
    print("By kind:", dict(Counter(chunk.get("kind", "code") for chunk in chunks)))
    print("By chunker:", dict(Counter(chunk.get("chunker", "lines") for chunk in chunks)))
    lengths = [chunk["end_line"] - chunk["start_line"] + 1 for chunk in chunks if chunk.get("kind", "code") == "code"]

    if lengths:
        print(f"Lines per code chunk: mean {sum(lengths) / len(lengths):.1f}, max {max(lengths)}")


def command_serve(arguments) -> None:
    import uvicorn

    uvicorn.run(
        "repomentor.server:app",
        host=arguments.host,
        port=arguments.port,
        reload=arguments.reload,
    )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    from .agents import DEFAULT_AGENT, list_agents
    from .evaluation import add_eval_commands
    from .retrieval import METHODS
    from .sources import PROVIDERS

    parser = argparse.ArgumentParser(prog="repomentor", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="Download, chunk and embed a repository.")
    index.add_argument("--provider", choices=PROVIDERS, default="gitlab")
    index.add_argument("--project", help="group/project, owner/repo, or a local checkout path.")
    index.add_argument("--full", action="store_true", help="Rebuild from scratch instead of syncing.")
    index.add_argument("--no-history", action="store_true", help="Skip commits and merge requests.")
    add_data_argument(index)
    index.set_defaults(handler=command_index)

    ask = commands.add_parser("ask", help="Ask a question about the indexed repository.")
    ask.add_argument("question", nargs="?", help="Omit to start an interactive session.")
    ask.add_argument("--top-k", type=int, default=config.TOP_K)
    ask.add_argument("--retrieval-only", action="store_true",
                     help="Show retrieved sources without calling the language model.")
    ask.add_argument("--method", choices=METHODS, help="Retrieval method (default: RETRIEVAL_METHOD).")
    ask.add_argument("--agent", choices=[agent.id for agent in list_agents()], default=DEFAULT_AGENT)
    add_data_argument(ask)
    ask.set_defaults(handler=command_ask)

    summarize = commands.add_parser("summarize", help="Build directory summaries (tour agent).")
    add_data_argument(summarize)
    summarize.set_defaults(handler=command_summarize)

    inspect = commands.add_parser("inspect", help="Show statistics about an index.")
    add_data_argument(inspect)
    inspect.set_defaults(handler=command_inspect)

    serve = commands.add_parser("serve", help="Run the web interface.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(handler=command_serve)

    add_eval_commands(commands)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if getattr(arguments, "top_k", 1) < 1:
        parser.error("--top-k must be at least 1.")

    arguments.handler(arguments)
