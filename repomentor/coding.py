"""
Turning a model's code answer into concrete file changes.

The coder agent asks the model for SEARCH/REPLACE blocks for existing files
and full content for new files. This module parses those blocks, applies
them to the indexed version of each file, produces unified diffs and runs
a syntax check on the result, so the user sees a reviewable patch instead
of loose snippets, and can open a merge request from it.
"""

import ast
import difflib
import json
import re
import tomllib
from pathlib import PurePosixPath


FILE_HEADER = re.compile(
    r"^\s*(?:#+\s*)?(?:\*\*|`)*(?:File:\s*)?(?:\*\*|`)*"
    r"(?P<path>[\w./-]+\.\w+|[\w./-]*Dockerfile)"
    r"(?:\*\*|`)*\s*(?P<new>\((?:new|new file)\))?\s*:?\s*$",
    re.IGNORECASE,
)
FENCE = re.compile(r"^\s*```")
SEARCH = re.compile(r"^\s*<{5,}\s*SEARCH\s*$")
DIVIDER = re.compile(r"^\s*={5,}\s*$")
REPLACE = re.compile(r"^\s*>{5,}\s*REPLACE\s*$")

TREE_SITTER_LANGUAGES = {".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "tsx"}


def parse_blocks(answer: str) -> list[dict]:
    """
    Find code blocks preceded by a file path line. Returns
    [{"path", "new": bool, "code": str}] in answer order.
    """

    lines = answer.splitlines()
    blocks = []
    path = None
    new = False
    position = 0

    while position < len(lines):
        line = lines[position]
        header = FILE_HEADER.match(line)

        if header and not FENCE.match(line):
            path = header["path"].strip("/")
            new = bool(header["new"])
            position += 1
            continue

        if FENCE.match(line):
            code = []
            position += 1

            while position < len(lines) and not FENCE.match(lines[position]):
                code.append(lines[position])
                position += 1

            position += 1  # closing fence

            if path:
                blocks.append({"path": path, "new": new, "code": "\n".join(code)})

            path = None
            new = False
            continue

        if line.strip():
            # Prose between a path and a code block detaches them
            path = None

        position += 1

    return blocks


def parse_edits(code: str) -> list[tuple[str, str]] | None:
    """SEARCH/REPLACE pairs in a block, or None if it has none."""

    edits = []
    lines = code.splitlines()
    position = 0

    while position < len(lines):
        if not SEARCH.match(lines[position]):
            position += 1
            continue

        search, replace = [], []
        position += 1

        while position < len(lines) and not DIVIDER.match(lines[position]):
            search.append(lines[position])
            position += 1

        position += 1

        while position < len(lines) and not REPLACE.match(lines[position]):
            replace.append(lines[position])
            position += 1

        position += 1
        edits.append(("\n".join(search), "\n".join(replace)))

    return edits or None


def apply_edit(content: str, search: str, replace: str) -> str | None:
    """Replace `search` once; tolerate trailing-whitespace differences."""

    if search and search in content:
        return content.replace(search, replace, 1)

    if not search.strip():
        # An empty SEARCH means "append"
        return content.rstrip("\n") + "\n\n" + replace + "\n"

    content_lines = content.splitlines()
    search_lines = [line.rstrip() for line in search.splitlines()]

    # Ignore blank lines the model added or dropped at the ends
    while search_lines and not search_lines[0]:
        search_lines.pop(0)

    while search_lines and not search_lines[-1]:
        search_lines.pop()

    size = len(search_lines)

    for start in range(len(content_lines) - size + 1):
        window = [line.rstrip() for line in content_lines[start:start + size]]

        if window == search_lines:
            updated = content_lines[:start] + replace.splitlines() + content_lines[start + size:]
            return "\n".join(updated) + ("\n" if content.endswith("\n") else "")

    return None


def check_syntax(path: str, content: str) -> dict:
    """{"ok": bool | None, "message": str}; ok is None when not checked."""

    suffix = PurePosixPath(path).suffix.lower()

    try:
        if suffix == ".py":
            ast.parse(content, filename=path)
            return {"ok": True, "message": "Python syntax is valid."}

        if suffix == ".json":
            json.loads(content)
            return {"ok": True, "message": "JSON is valid."}

        if suffix == ".toml":
            tomllib.loads(content)
            return {"ok": True, "message": "TOML is valid."}

        if suffix in (".yml", ".yaml"):
            try:
                import yaml
            except ImportError:
                return {"ok": None, "message": "Install PyYAML to check YAML."}

            list(yaml.safe_load_all(content))
            return {"ok": True, "message": "YAML is valid."}

        if suffix in TREE_SITTER_LANGUAGES:
            from .chunking import get_parser

            parser = get_parser(TREE_SITTER_LANGUAGES[suffix])

            if parser is None:
                return {"ok": None, "message": "Install the Tree-sitter grammars to check this file."}

            tree = parser.parse(content.encode())

            if tree.root_node.has_error:
                row = first_error_row(tree.root_node)
                return {"ok": False, "message": f"Syntax error near line {row + 1}."}

            return {"ok": True, "message": "Syntax is valid."}

    except SyntaxError as error:
        return {"ok": False, "message": f"Line {error.lineno}: {error.msg}"}
    except Exception as error:  # json, toml and yaml errors
        return {"ok": False, "message": str(error).splitlines()[0]}

    return {"ok": None, "message": "No syntax check for this file type."}


def first_error_row(node) -> int:
    if node.type == "ERROR" or node.is_missing:
        return node.start_point[0]

    for child in node.children:
        if child.has_error:
            return first_error_row(child)

    return node.start_point[0]


def unified_diff(path: str, before: str, after: str, new_file: bool) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="/dev/null" if new_file else f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def build_changes(answer: str, read_file) -> list[dict]:
    """
    Apply the answer's blocks. `read_file(path)` returns the current content
    of a repository file or None. Returns one entry per changed file:
    {"path", "status": "modified" | "created" | "failed", "diff",
     "content", "check", "error"}.
    """

    files: dict[str, dict] = {}

    for block in parse_blocks(answer):
        path = block["path"]
        entry = files.get(path)

        if entry is None:
            original = read_file(path)
            entry = files[path] = {
                "path": path,
                "original": original,
                "content": original,
                "errors": [],
            }

        edits = parse_edits(block["code"])

        if edits is None:
            # A whole file (new, or rewritten in full)
            entry["content"] = block["code"].rstrip("\n") + "\n"
            continue

        if entry["content"] is None:
            entry["errors"].append(f"{path} is not in the index, so it cannot be edited.")
            continue

        for search, replace in edits:
            updated = apply_edit(entry["content"], search, replace)

            if updated is None:
                entry["errors"].append(
                    "A SEARCH block did not match the indexed file:\n" + search[:300]
                )
            else:
                entry["content"] = updated

    changes = []

    for path, entry in files.items():
        original, content = entry["original"], entry["content"]
        new_file = original is None

        if content is None or (content == original and not entry["errors"]):
            continue

        changes.append({
            "path": path,
            "status": "failed" if entry["errors"] and content == original else (
                "created" if new_file else "modified"
            ),
            "diff": unified_diff(path, original or "", content or "", new_file),
            "content": content,
            "check": check_syntax(path, content) if content else {"ok": None, "message": ""},
            "error": "\n".join(entry["errors"]) or None,
        })

    return changes
