import numpy as np
import pytest

from repomentor import chunking
from repomentor.chunking import chunk_file, chunk_history, line_ranges, markdown_ranges
from repomentor.embeddings import embed_chunks
from repomentor.index import build_index, load_index, read_manifest
from repomentor.sources import LocalGitSource

from .conftest import git


def record(path: str, content: str) -> dict:
    return {
        "project_path": "group/project",
        "file_path": path,
        "last_commit_id": "abc",
        "content": content,
    }


def test_line_ranges_overlap():
    assert list(line_ranges(130)) == [(0, 60), (50, 110), (100, 130)]


def test_markdown_splits_at_headings():
    lines = (
        ["# Title"] + ["text"] * 10
        + ["## Setup"] + ["```", "# not a heading", "```"] + ["more"] * 10
    )
    ranges = list(markdown_ranges(lines))

    assert [start for start, _, _ in ranges] == [0, 11]


@pytest.mark.skipif(not chunking.tree_sitter_available(), reason="tree-sitter not installed")
def test_tree_sitter_keeps_functions_whole():
    functions = "\n\n".join(
        f"def function_{n}(value):\n" + "\n".join(f"    value += {i}" for i in range(20)) + "\n    return value"
        for n in range(6)
    )
    chunks = chunk_file(record("app/module.py", functions))

    assert all(chunk["chunker"] == "tree-sitter" for chunk in chunks)
    assert len(chunks) > 1

    for chunk in chunks:
        # Every chunk starts at a definition, never in the middle of one
        assert chunk["content"].startswith("def function_")
        assert chunk["content"].count("def ") == chunk["content"].count("return value")

    assert {"function_0", "function_5"} <= {s for c in chunks for s in c["symbols"]}


@pytest.mark.skipif(not chunking.tree_sitter_available(), reason="tree-sitter not installed")
def test_tree_sitter_splits_large_classes_into_methods():
    methods = "\n\n".join(
        f"    def method_{n}(self):\n" + "\n".join(f"        x = {i}" for i in range(25))
        for n in range(5)
    )
    chunks = chunk_file(record("app/service.py", f"class Service:\n{methods}\n"))

    assert len(chunks) > 1
    assert chunks[0]["content"].startswith("class Service:")
    assert any(symbol.startswith("Service.method_") for c in chunks for s in [c["symbols"]] for symbol in s)


def test_line_chunker_is_used_when_requested():
    chunks = chunk_file(record("app/a.py", "x = 1\n" * 130), chunker="lines")
    assert [chunk["chunker"] for chunk in chunks] == ["lines"] * 3


def test_generated_and_empty_files_are_skipped():
    assert chunk_file(record("client/sdk.gen.ts", "export const a = 1")) == []
    assert chunk_file(record("empty.py", "   \n")) == []


def test_history_chunk():
    chunk = chunk_history({
        "kind": "merge_request", "id": "12", "title": "Switch to Argon2",
        "body": "bcrypt was too slow on our hardware", "author": "Ada",
        "date": "2024-01-01", "url": "https://example.com/mr/12",
        "project_path": "g/p",
    })

    assert chunk["file_path"] == "merge_requests/!12"
    assert "bcrypt was too slow" in chunk["content"]
    assert chunk["url"] == "https://example.com/mr/12"


def test_embedding_text_includes_path_and_symbols(monkeypatch):
    from repomentor import config
    from repomentor.embeddings import embedding_text

    chunk = {"file_path": "app/a.py", "symbols": ["run"], "content": "def run(): pass"}
    monkeypatch.setattr(config, "EMBEDDING_CONTEXT", True)
    assert embedding_text(chunk) == "File: app/a.py\nDefines: run\n\ndef run(): pass"
    assert embedding_text({**chunk, "kind": "commit"}) == "def run(): pass"

    monkeypatch.setattr(config, "EMBEDDING_CONTEXT", False)
    assert embedding_text(chunk) == "def run(): pass"


def test_embedding_cache_reuses_vectors(fake_model, monkeypatch):
    from repomentor import config

    monkeypatch.setattr(config, "EMBEDDING_CONTEXT", False)
    chunks = [{"content": "alpha"}, {"content": "beta"}]
    vectors, reused = embed_chunks(chunks, fake_model)
    assert reused == 0

    cache = {}
    from repomentor.embeddings import content_key

    cache[content_key("alpha")] = vectors[0]
    _, reused = embed_chunks(chunks + [{"content": "gamma"}], fake_model, cache)
    assert reused == 1


def test_build_index_and_incremental_sync(tmp_path, local_repo, fake_model):
    source = LocalGitSource()
    first = tmp_path / "first"
    manifest = build_index(source, str(local_repo), first, embedding_model=fake_model)

    assert manifest["file_count"] == 4
    assert manifest["history_count"] == 2
    assert not manifest["sync"]["incremental"]

    index = load_index(first, fake_model)
    kinds = {chunk["kind"] for chunk in index.chunks}
    assert kinds == {"code", "commit"}
    assert len(index.chunks) == len(index.store.embeddings)
    assert index.file("app/security.py")["content"].startswith('"""Password')

    (local_repo / "app" / "users.py").write_text("def nothing():\n    pass\n")
    git(local_repo, "commit", "-qam", "Simplify users")

    second = tmp_path / "second"
    manifest = build_index(source, str(local_repo), second, previous=first, embedding_model=fake_model)
    sync = manifest["sync"]

    assert sync["incremental"]
    assert sync["downloaded"] == 1 and sync["reused"] == 3
    assert sync["reused_vectors"] > 0
    assert read_manifest(second)["head_commit"] != read_manifest(first)["head_commit"]
    assert np.isfinite(np.load(second / "embeddings.npy")).all()
