"""The original RepoMentor behavior: cited, fact-by-fact answers."""

from .. import answer
from .base import Agent, AgentContext


class MentorAgent(Agent):
    id = "mentor"
    name = "Mentor"
    description = (
        "Answers questions about the codebase. Every fact links to the "
        "file and lines it came from."
    )
    examples = [
        "How does user authentication work?",
        "How are user passwords handled?",
        "Where are the database models defined?",
        "Why was the login flow changed recently?",
    ]
    working_label = "Reading sources and extracting facts…"

    def stream(self, question: str, context: AgentContext):
        yield from answer.stream_answer(
            question,
            context.index,
            context.top_k,
            context.retrieval_only,
            retrieve_lock=context.retrieve_lock,
        )

    def format_text(self, result: dict) -> str:
        return answer.format_answer(result)
