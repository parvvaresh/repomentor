"""
Split the raw corpus into chunks for retrieval and citation.

Three strategies are available:

- "tree-sitter": code is split at syntax boundaries (functions, classes,
  methods), so a chunk rarely starts or ends halfway through a function.
  Small neighbouring definitions are packed together; definitions that are
  too long are split into their children, and only as a last resort into
  line windows. Needs `tree-sitter` and the tree-sitter-python,
  -javascript and -typescript grammar packages.
- "markdown": documents are split at headings.
- "lines": the original 60-line windows with a 10-line overlap.

CHUNKER=auto (the default) picks the best strategy available per file.
History records (commits, merge requests) become one chunk each.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath

from . import config


CHUNK_SIZE = 60
OVERLAP = 10
# Code-aware chunks may be a little longer than line windows, because
# cutting a function in two hurts more than a slightly longer chunk.
MAX_CODE_LINES = 80
MIN_CODE_LINES = 8

LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}

# Node types that name something worth listing as a chunk's symbol
DEFINITION_TYPES = {
    "function_definition", "class_definition", "decorated_definition",
    "function_declaration", "class_declaration", "method_definition",
    "interface_declaration", "type_alias_declaration", "enum_declaration",
    "lexical_declaration", "variable_declaration", "export_statement",
    "abstract_class_declaration",
}

# Children of a large definition worth splitting at
BODY_FIELDS = ("body", "declaration", "definition")

_parsers: dict[str, object] = {}


def should_skip(file_path: str, content: str) -> bool:
    if not content.strip():
        return True

    name = PurePosixPath(file_path).name.lower()

    if name in {"release-notes.md", "changelog.md", "package-lock.json"}:
        return True

    return file_path.endswith((".gen.ts", ".min.js", ".lock"))


def create_chunk_id(*parts) -> str:
    identity = ":".join(str(part) for part in parts)
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Line windows
# ---------------------------------------------------------------------------

def line_ranges(line_count: int, size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """(start, end) pairs, 0-based and end-exclusive."""

    step = size - overlap

    for start in range(0, line_count, step):
        end = min(start + size, line_count)
        yield start, end

        if end == line_count:
            break


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def markdown_ranges(lines: list[str]):
    """Split at headings outside code fences; long sections use line windows."""

    starts = [0]
    in_fence = False

    for number, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        elif not in_fence and line.startswith("#") and number > 0:
            starts.append(number)

    starts.append(len(lines))
    sections = list(zip(starts, starts[1:]))

    # Merge tiny sections into the next one
    merged = []

    for start, end in sections:
        if merged and (merged[-1][1] - merged[-1][0]) < MIN_CODE_LINES:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    for start, end in merged:
        if end - start <= MAX_CODE_LINES:
            yield start, end, []
        else:
            for window_start, window_end in line_ranges(end - start):
                yield start + window_start, start + window_end, []


# ---------------------------------------------------------------------------
# Tree-sitter
# ---------------------------------------------------------------------------

GRAMMARS = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
}


def get_parser(language: str):
    """A Tree-sitter parser, or None when the grammar is not installed."""

    if language not in _parsers:
        try:
            import importlib

            from tree_sitter import Language, Parser

            module_name, function = GRAMMARS[language]
            module = importlib.import_module(module_name)
            _parsers[language] = Parser(Language(getattr(module, function)()))
        except Exception:
            _parsers[language] = None

    return _parsers[language]


def tree_sitter_available() -> bool:
    return get_parser("python") is not None


def node_name(node) -> str | None:
    if node.type == "decorated_definition":
        definition = node.child_by_field_name("definition")
        return node_name(definition) if definition else None

    if node.type == "export_statement":
        for child in node.named_children:
            name = node_name(child)

            if name:
                return name

        return None

    if node.type in {"lexical_declaration", "variable_declaration"}:
        for child in node.named_children:
            name = child.child_by_field_name("name")

            if name is not None:
                return name.text.decode(errors="replace")

        return None

    name = node.child_by_field_name("name")
    return name.text.decode(errors="replace") if name is not None else None


def split_children(node):
    """Children worth splitting a too-long node at (e.g. a class's methods)."""

    target = node

    for field in BODY_FIELDS:
        child = target.child_by_field_name(field)

        if child is not None:
            target = child

            # decorated_definition -> class_definition -> block
            for inner_field in BODY_FIELDS:
                inner = target.child_by_field_name(inner_field)

                if inner is not None:
                    target = inner
                    break

            break

    return [child for child in target.named_children if child.type != "comment"]


def syntax_segments(nodes, parent: str | None = None):
    """
    Yield (start_row, end_row, symbol) for each node. A definition longer
    than MAX_CODE_LINES is replaced by segments for its children (a class
    by its methods); its header stays with the first child segment.
    """

    for node in nodes:
        start, end = node.start_point[0], node.end_point[0]
        name = node_name(node) if node.type in DEFINITION_TYPES else None
        symbol = f"{parent}.{name}" if parent and name else (name or parent)
        inner = split_children(node) if end - start + 1 > MAX_CODE_LINES else []

        if not inner:
            yield start, end, symbol
            continue

        first = True

        for segment_start, segment_end, segment_symbol in syntax_segments(inner, symbol):
            if first:
                segment_start = start
                first = False

            yield segment_start, segment_end, segment_symbol


def code_ranges(content: str, language: str):
    """
    Yield (start, end, symbols) with 0-based, end-exclusive line numbers,
    or None if the file cannot be parsed.
    """

    parser = get_parser(language)

    if parser is None:
        return None

    tree = parser.parse(content.encode())

    if tree.root_node.has_error and tree.root_node.type == "ERROR":
        return None

    line_count = len(content.splitlines())
    segments = sorted(
        syntax_segments(tree.root_node.named_children),
        key=lambda segment: (segment[0], segment[1]),
    )

    if not segments:
        return None

    ranges = []
    chunk_start = 0
    chunk_end = None
    symbols: list[str] = []

    def flush():
        if chunk_end is not None and chunk_end > chunk_start:
            ranges.append((chunk_start, chunk_end, list(symbols)))

    for start, end, symbol in segments:
        end = end + 1  # end-exclusive

        if chunk_end is not None and end - chunk_start > MAX_CODE_LINES and (
            chunk_end - chunk_start >= MIN_CODE_LINES
        ):
            flush()
            # Lines between definitions (comments, blank lines, decorators)
            # belong to the next definition
            chunk_start = chunk_end
            symbols = []

        chunk_end = max(end, chunk_end or 0)

        if symbol:
            symbols.append(symbol)

    chunk_end = max(chunk_end or 0, line_count)
    flush()

    # A single statement longer than the limit (a huge dict, a long JSX
    # tree) still has to be cut into windows
    result = []

    for start, end, chunk_symbols in ranges:
        if end - start <= MAX_CODE_LINES * 1.5:
            result.append((start, end, chunk_symbols))
            continue

        for window_start, window_end in line_ranges(end - start):
            result.append((start + window_start, start + window_end, chunk_symbols))

    return result


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def choose_ranges(file_path: str, content: str, chunker: str):
    lines = content.splitlines()
    suffix = PurePosixPath(file_path).suffix.lower()

    if chunker in ("auto", "tree-sitter") and suffix in LANGUAGES:
        ranges = code_ranges(content, LANGUAGES[suffix])

        if ranges is not None:
            return "tree-sitter", ranges

    if chunker in ("auto", "markdown") and suffix == ".md":
        return "markdown", list(markdown_ranges(lines))

    return "lines", [(start, end, []) for start, end in line_ranges(len(lines))]


def chunk_file(record: dict, chunker: str = "auto") -> list[dict]:
    file_path = record["file_path"]
    content = record["content"]

    if should_skip(file_path, content):
        return []

    lines = content.splitlines()
    method, ranges = choose_ranges(file_path, content, chunker)
    chunks = []

    for start, end, symbols in ranges:
        text = "\n".join(lines[start:end]).strip()

        if not text:
            continue

        chunks.append({
            "chunk_id": create_chunk_id(
                record["project_path"], file_path,
                record["last_commit_id"], start + 1, end,
            ),
            "kind": "code",
            "provider": record.get("provider", "gitlab"),
            "project_path": record["project_path"],
            "web_url": record.get("web_url"),
            "file_path": file_path,
            "blob_id": record.get("blob_id"),
            "last_commit_id": record["last_commit_id"],
            "start_line": start + 1,
            "end_line": end,
            "symbols": sorted(set(symbols)),
            "chunker": method,
            "content": text,
        })

    return chunks


def chunk_history(record: dict) -> dict | None:
    body = (record.get("body") or "").strip()
    title = (record.get("title") or "").strip()

    if not title and not body:
        return None

    label = "Merge request" if record["kind"] == "merge_request" else "Commit"
    reference = (
        f"!{record['id']}" if record["kind"] == "merge_request"
        else record["id"][:10]
    )
    text = f"{label} {reference}: {title}\nAuthor: {record['author']}\nDate: {record['date']}"

    if body and body != title:
        text += f"\n\n{body[:4000]}"

    lines = text.count("\n") + 1

    return {
        "chunk_id": create_chunk_id(record["project_path"], record["kind"], record["id"]),
        "kind": record["kind"],
        "provider": record.get("provider", "gitlab"),
        "project_path": record["project_path"],
        "web_url": record.get("web_url"),
        "file_path": f"{record['kind']}s/{reference}",
        "title": title,
        "author": record.get("author", ""),
        "date": record.get("date", ""),
        "url": record.get("url", ""),
        "last_commit_id": record["id"] if record["kind"] == "commit" else "",
        "start_line": 1,
        "end_line": lines,
        "symbols": [],
        "chunker": "history",
        "content": text,
    }


def chunk_dataset(
    input_path: Path,
    output_path: Path,
    history_path: Path | None = None,
    chunker: str | None = None,
) -> tuple[int, int]:
    """
    Chunk every file of the raw corpus (plus history records, if given).
    Returns the number of chunks created and files skipped.
    """

    chunker = chunker or config.CHUNKER
    chunk_count = 0
    skipped_files = 0

    with (
        input_path.open("r", encoding="utf-8") as input_file,
        output_path.open("w", encoding="utf-8") as output_file,
    ):
        for line in input_file:
            if not line.strip():
                continue

            chunks = chunk_file(json.loads(line), chunker)

            if not chunks:
                skipped_files += 1

            for chunk in chunks:
                output_file.write(json.dumps(chunk) + "\n")
                chunk_count += 1

        if history_path and history_path.exists():
            with history_path.open("r", encoding="utf-8") as history_file:
                for line in history_file:
                    if not line.strip():
                        continue

                    chunk = chunk_history(json.loads(line))

                    if chunk:
                        output_file.write(json.dumps(chunk) + "\n")
                        chunk_count += 1

    return chunk_count, skipped_files
