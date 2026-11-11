"""
The contract every RepoMentor agent implements.

An agent receives a question plus the indexed repository and returns a
JSON-serializable result. The web server stores that result with the
chat message, and the frontend picks a renderer from `result["agent"]`.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass
class AgentContext:
    """Everything an agent may use to answer one request."""

    # Loaded search index of the chat's repository (see rag_answer.load_index)
    index: dict
    project_path: str
    top_k: int
    retrieval_only: bool = False
    # Serializes use of the shared embedding model across threads
    retrieve_lock: Any = None
    # Earlier messages of the chat, oldest first:
    # [{"role": "user" | "assistant", "content": str, "agent": str | None}]
    history: list[dict] = field(default_factory=list)


class Agent:
    """
    Base class for agents. Subclasses set the class attributes and
    implement `run`. Register an instance with `agents.register`.
    """

    # Stable identifier, stored with every message; never rename it
    id: ClassVar[str]
    name: ClassVar[str]
    description: ClassVar[str]
    # Example prompts shown on an empty chat
    examples: ClassVar[list[str]] = []
    # Text the frontend shows while the agent is working
    working_label: ClassVar[str] = "Working…"

    def run(self, question: str, context: AgentContext) -> dict:
        """
        Return a JSON-serializable dict. It must not contain the key
        "error" (reserved for failures) and should include "sources"
        (see rag_answer.describe_source) when it cites the repository.
        The server adds "agent" automatically.
        """

        raise NotImplementedError

    def format_text(self, result: dict) -> str:
        """Plain-text version of a result, used for copying and history."""

        return str(result.get("answer", ""))

    def describe(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "examples": list(self.examples),
            "working_label": self.working_label,
        }
