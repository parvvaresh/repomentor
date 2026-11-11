"""The original RepoMentor behavior: cited, fact-by-fact answers."""

import rag_answer

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
        "How is the application deployed?",
    ]
    working_label = "Reading sources and extracting facts…"

    def run(self, question: str, context: AgentContext) -> dict:
        return rag_answer.generate_answer(
            question,
            context.index,
            context.top_k,
            context.retrieval_only,
            retrieve_lock=context.retrieve_lock,
        )

    def format_text(self, result: dict) -> str:
        sources = result["sources"]

        if result["retrieval_only"]:
            return "\n".join(source["citation"] for source in sources)

        if not result["facts"]:
            return (
                "I couldn't find enough evidence in the "
                "retrieved repository sources."
            )

        return "\n".join(
            f"- {fact['text']} {sources[fact['source']]['citation']}"
            for fact in result["facts"]
        )
