from datetime import datetime, timezone

import pytest

from repomentor import llm, summaries
from repomentor.agents import AgentContext, get_agent, list_agents
from repomentor.agents.experts import rank_people
from repomentor.agents.reviewer import find_reference, parse_pasted_diff
from repomentor.agents.tour import tour_markdown, wants_tour
from repomentor.coding import apply_edit, build_changes, check_syntax, parse_blocks
from repomentor.sources import LocalGitSource


def context(index, repo=None, **overrides):
    values = dict(
        index=index,
        project_path="demo",
        top_k=3,
        source=LocalGitSource() if repo else None,
        source_project=str(repo) if repo else None,
    )
    values.update(overrides)
    return AgentContext(**values)


def final(events):
    return [event for event in events if event["type"] == "result"][0]["result"]


def test_registry_lists_all_agents():
    assert [agent.id for agent in list_agents()] == ["mentor", "coder", "reviewer", "experts", "tour"]
    assert get_agent(None).id == "mentor"


# --- Coder ------------------------------------------------------------------------

CODER_ANSWER = """Plan:
- Rename the helper [1]

**app/security.py**
```python
<<<<<<< SEARCH
def get_password_hash(password: str) -> str:
=======
def hash_password(password: str) -> str:
>>>>>>> REPLACE
```

**app/helpers.py** (new file)
```python
def broken(:
    pass
```
"""


def test_build_changes_applies_edits_and_checks_syntax():
    original = {"app/security.py": "import x\n\ndef get_password_hash(password: str) -> str:\n    return x\n"}
    changes = {c["path"]: c for c in build_changes(CODER_ANSWER, original.get)}

    modified = changes["app/security.py"]
    assert modified["status"] == "modified"
    assert "+def hash_password" in modified["diff"]
    assert modified["check"]["ok"] is True

    created = changes["app/helpers.py"]
    assert created["status"] == "created"
    assert created["check"]["ok"] is False


def test_unmatched_search_block_is_reported():
    answer = "**a.py**\n```python\n<<<<<<< SEARCH\nnot there\n=======\nx\n>>>>>>> REPLACE\n```"
    [change] = build_changes(answer, lambda path: "print(1)\n")

    assert change["status"] == "failed"
    assert "did not match" in change["error"]


def test_apply_edit_tolerates_trailing_whitespace():
    assert apply_edit("a = 1   \nb = 2\n", "a = 1\nb = 2", "a = 3\nb = 4") == "a = 3\nb = 4\n"


def test_parse_blocks_needs_a_path():
    assert parse_blocks("Some text\n```python\nx = 1\n```") == []


def test_check_syntax_formats():
    assert check_syntax("a.json", "{")["ok"] is False
    assert check_syntax("a.toml", "x = 1")["ok"] is True
    assert check_syntax("README.md", "# hi")["ok"] is None


def test_coder_streams_and_produces_changes(built_index, monkeypatch):
    def fake_stream(messages, **_):
        # The repository text is delimited as untrusted data
        assert "Security rules" in messages[0]["content"]
        yield from (CODER_ANSWER[:40], CODER_ANSWER[40:])

    monkeypatch.setattr(llm, "chat_stream", fake_stream)
    events = list(get_agent("coder").stream("Rename get_password_hash", context(built_index)))

    assert [e["type"] for e in events].count("delta") == 2
    result = final(events)
    assert {change["path"] for change in result["changes"]} == {"app/security.py", "app/helpers.py"}
    assert result["base_commit"] == built_index.manifest["head_commit"]


# --- Reviewer ---------------------------------------------------------------------

PASTED_DIFF = """diff --git a/app/users.py b/app/users.py
--- a/app/users.py
+++ b/app/users.py
@@ -1,3 +1,4 @@
+import hashlib
 def create_user(email, password):
-    return get_password_hash(password)
+    return hashlib.md5(password.encode()).hexdigest()
"""


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("Review !12 please", 12),
        ("review #7", 7),
        ("https://gitlab.com/g/p/-/merge_requests/42", 42),
        ("https://github.com/o/r/pull/9", 9),
        ("look at MR 5", 5),
        ("hello", None),
    ],
)
def test_find_reference(text, number):
    assert find_reference(text) == number


def test_parse_pasted_diff():
    [change] = parse_pasted_diff(f"```diff\n{PASTED_DIFF}```")
    assert change["path"] == "app/users.py"
    assert "+import hashlib" in change["diff"]


def test_reviewer_reviews_pasted_diff(built_index, monkeypatch):
    seen = {}

    def fake_stream(messages, **_):
        seen["prompt"] = messages[1]["content"]
        yield "## Summary\nUses MD5 [1]."

    monkeypatch.setattr(llm, "chat_stream", fake_stream)
    result = final(get_agent("reviewer").stream(PASTED_DIFF, context(built_index)))

    assert result["files"] == ["app/users.py"]
    assert result["merge_request"] is None
    assert "hashlib.md5" in seen["prompt"]
    assert result["sources"]


def test_reviewer_needs_a_change(built_index):
    with pytest.raises(ValueError):
        list(get_agent("reviewer").stream("please review", context(built_index)))


# --- Who to ask -------------------------------------------------------------------

def test_rank_people_weights_recent_work_and_skips_bots():
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)

    def commit(sha, name, date):
        return {"sha": sha, "title": sha, "url": "", "author_name": name,
                "author_email": f"{name}@x", "date": date}

    people = rank_people({
        "a.py": [commit("1", "old", "2020-01-01T00:00:00Z")] * 3
        + [commit("2", "new", "2024-12-01T00:00:00Z"), commit("3", "dependabot[bot]", "2024-12-31")],
    }, now)

    assert [person["name"] for person in people] == ["new", "old"]
    assert people[1]["commits"] == 1  # the same commit counted once


def test_experts_agent_reads_file_history(built_index, local_repo):
    result = final(get_agent("experts").stream("Who knows the user service?", context(built_index, local_repo)))
    names = [person["name"] for person in result["people"]]

    assert "Grace Hopper" in names
    assert result["files_checked"] >= 1


# --- Tour ---------------------------------------------------------------------------

def test_wants_tour():
    assert wants_tour("Give me a tour")
    assert wants_tour("")
    assert not wants_tour("How is the application deployed?")


def test_plan_directories_builds_a_tree():
    tree = summaries.plan_directories(["a/b/c.py", "a/d.py", "e.md"])
    assert tree["."]["children"] == ["a"]
    assert tree["a"]["children"] == ["a/b"]
    assert tree["a/b"]["files"] == ["a/b/c.py"]
    assert tree["."]["files"] == ["e.md"]


def test_parse_learning_path_keeps_known_paths():
    steps = summaries.parse_learning_path(
        "1. `app` - start here\n2. invented/ - nope\n3. README.md: overview",
        {"app", "README.md"},
    )
    assert [step["path"] for step in steps] == ["app", "README.md"]


def test_tour_builds_summaries_once(built_index, monkeypatch):
    calls = []

    def fake_chat(messages, **_):
        calls.append(messages)

        if "Learning path" in messages[0]["content"]:
            return "A demo app.\nLearning path:\n1. app - the code\n2. README.md - docs"

        return "Holds the code."

    monkeypatch.setattr(llm, "chat", fake_chat)
    tour = get_agent("tour")
    result = final(tour.stream("Give me a tour", context(built_index)))

    assert result["mode"] == "tour"
    assert [step["path"] for step in result["learning_path"]] == ["app", "README.md"]
    assert "Architecture map" in tour_markdown(built_index.summaries())

    count = len(calls)
    final(tour.stream("Give me a tour", context(built_index)))
    assert len(calls) == count  # cached in summaries.json

    monkeypatch.setattr(llm, "chat_stream", lambda *a, **k: iter(["Deployed with docker [1]."]))
    answer = final(tour.stream("How is the application deployed?", context(built_index)))
    assert answer["mode"] == "answer"
    assert answer["sources"][0]["kind"] == "summary"
