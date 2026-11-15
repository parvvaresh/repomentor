"""
Hierarchical (RAPTOR-style) summaries of a repository.

Broad questions such as "How is the application deployed?" or "Where do I
start?" are not answered by any single chunk. This module builds a tree of
summaries bottom-up:

1. each file gets a cheap digest without a model: its symbols and the
   first lines of its docstring or heading;
2. each directory is summarized by the model from the digests of its files
   and the summaries of its subdirectories, deepest directories first;
3. the repository gets an overview and a suggested learning path.

The result is saved as summaries.json next to the index and reused until
files change.
"""

import json
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import PurePosixPath

from . import config, llm, safety
from .index import SUMMARIES_FILE, now


MAX_DEPTH = 4
MAX_DIRECTORIES = 60
MAX_FILES_PER_DIRECTORY = 40
MAX_LEARNING_STEPS = 8

DIRECTORY_PROMPT = """
You summarize one directory of the repository {project} for a developer who
is new to it. Using only the file digests and subdirectory summaries given,
write 2-3 sentences: what this directory is responsible for and the most
important files in it. Do not guess beyond what the digests show.
{policy}"""

OVERVIEW_PROMPT = """
You write the overview of the repository {project} for a new developer,
using only the directory summaries given. Write:
1. One paragraph on what the system is and its main parts.
2. A line starting "Learning path:" followed by up to {steps} numbered steps,
   each formatted exactly "N. <path> - <why to read it at this point>", where
   <path> is one of the listed directories or files. Order them so each step
   builds on the previous ones.
{policy}"""

STEP_PATTERN = re.compile(r"^\s*\d+[.)]\s*`?([\w./-]+)`?\s*[-–—:]\s*(.+)$")


def file_digest(record: dict, symbols: list[str]) -> str:
    """A one-line description of a file built without a model."""

    lines = [line.strip() for line in record["content"].splitlines() if line.strip()]
    description = ""

    for line in lines[:8]:
        if line.startswith(("#", '"""', "'''", "//", "/*", "*")):
            text = line.strip("#/*\"' ").strip()

            if text and not text.startswith("!"):
                description = text
                break

    digest = record["file_path"]

    if description:
        digest += f": {description[:120]}"

    if symbols:
        digest += f" [defines {', '.join(symbols[:8])}]"

    return digest


def directory_of(file_path: str) -> str:
    parts = PurePosixPath(file_path).parts[:-1][:MAX_DEPTH]
    return "/".join(parts) or "."


def parent_of(directory: str) -> str | None:
    if directory == ".":
        return None

    parent = str(PurePosixPath(directory).parent)
    return "." if parent in ("", ".") else parent


def plan_directories(file_paths: list[str]) -> dict[str, dict]:
    """{directory: {"files": [...], "children": [...]}} for every directory."""

    tree = defaultdict(lambda: {"files": [], "children": set()})

    for path in file_paths:
        directory = directory_of(path)
        tree[directory]["files"].append(path)

        child = directory

        while (parent := parent_of(child)) is not None:
            tree[parent]["children"].add(child)
            child = parent

    # Keep the directories with the most files when there are too many
    if len(tree) > MAX_DIRECTORIES:
        keep = sorted(tree, key=lambda d: (d.count("/"), -len(tree[d]["files"])))
        keep = set(keep[:MAX_DIRECTORIES]) | {"."}

        for directory in list(tree):
            if directory not in keep:
                parent = parent_of(directory)

                while parent not in keep:
                    parent = parent_of(parent)

                tree[parent]["files"].extend(tree[directory]["files"])
                del tree[directory]

        for node in tree.values():
            node["children"] = {child for child in node["children"] if child in tree}

    return {
        directory: {"files": sorted(node["files"]), "children": sorted(node["children"])}
        for directory, node in tree.items()
    }


def parse_learning_path(text: str, known_paths: set[str]) -> list[dict]:
    steps = []

    for line in text.splitlines():
        match = STEP_PATTERN.match(line)

        if not match:
            continue

        path = match.group(1).strip("/")

        if path in known_paths:
            steps.append({"path": path, "why": match.group(2).strip()})

        if len(steps) == MAX_LEARNING_STEPS:
            break

    return steps


def build_summaries(index, model: str | None = None) -> Iterator[dict]:
    """
    Yield {"type": "status", ...} events while summarizing, then
    {"type": "summaries", "summaries": {...}}. Saves summaries.json.
    """

    project = index.project_path
    symbols = defaultdict(list)

    for chunk in index.chunks:
        if chunk.get("kind", "code") == "code":
            symbols[chunk["file_path"]].extend(chunk.get("symbols", []))

    file_paths = index.file_paths()
    directories = plan_directories(file_paths)
    digests = {
        path: file_digest(index.file(path), sorted(set(symbols[path])))
        for path in file_paths
    }

    # Deepest first, so children are summarized before their parents
    order = sorted(directories, key=lambda d: (-(d.count("/") + (d != ".")), d))
    summaries: dict[str, str] = {}

    for position, directory in enumerate(order, start=1):
        node = directories[directory]
        yield {
            "type": "status",
            "label": f"Summarizing {directory} ({position}/{len(order)})…",
        }

        tag = safety.new_tag()
        file_lines = "\n".join(
            f"- {digests[path]}" for path in node["files"][:MAX_FILES_PER_DIRECTORY]
        )
        child_lines = "\n".join(
            f"- {child}/: {summaries.get(child, '')}" for child in node["children"]
        )
        content = ""

        if file_lines:
            content += safety.wrap(file_lines, tag, kind="file digests") + "\n\n"

        if child_lines:
            content += safety.wrap(child_lines, tag, kind="subdirectory summaries")

        summaries[directory] = llm.chat(
            [
                {
                    "role": "system",
                    "content": DIRECTORY_PROMPT.format(project=project, policy=safety.policy(tag)),
                },
                {"role": "user", "content": f"Directory: {directory}/\n\n{content}"},
            ],
            model=model,
        )

    yield {"type": "status", "label": "Writing the overview and learning path…"}
    tag = safety.new_tag()
    top_level = "\n".join(
        f"- {directory}/: {summaries[directory]}"
        for directory in sorted(summaries)
        if directory.count("/") <= 1
    )
    readme = index.file("README.md")
    readme_text = readme["content"][:3000] if readme else ""

    overview_text = llm.chat(
        [
            {
                "role": "system",
                "content": OVERVIEW_PROMPT.format(
                    project=project, steps=MAX_LEARNING_STEPS, policy=safety.policy(tag)
                ),
            },
            {
                "role": "user",
                "content": safety.wrap(top_level, tag, kind="directory summaries")
                + ("\n\n" + safety.wrap(readme_text, tag, file="README.md") if readme_text else ""),
            },
        ],
        model=model,
    )

    overview, _, path_text = overview_text.partition("Learning path:")
    known = set(file_paths) | set(directories)

    result = {
        "generated_at": now(),
        "model": model or config.MODEL_NAME,
        "head_commit": index.manifest.get("head_commit"),
        "overview": overview.strip(),
        "learning_path": parse_learning_path(path_text or overview_text, known),
        "directories": {
            directory: {
                "summary": summaries[directory],
                "files": directories[directory]["files"],
                "children": directories[directory]["children"],
            }
            for directory in sorted(directories)
        },
    }

    (index.directory / SUMMARIES_FILE).write_text(json.dumps(result, indent=2))
    yield {"type": "summaries", "summaries": result}
