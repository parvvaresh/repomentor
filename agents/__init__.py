"""
Agent registry.

To add an agent:
1. Create a module in this package with a subclass of `Agent`.
2. Import it below and pass an instance to `register`.

The web interface lists every registered agent automatically.
"""

from .base import Agent, AgentContext
from .coder import CoderAgent
from .mentor import MentorAgent


DEFAULT_AGENT = MentorAgent.id

_registry: dict[str, Agent] = {}


def register(agent: Agent) -> None:
    if agent.id in _registry:
        raise ValueError(f"Agent '{agent.id}' is already registered.")

    _registry[agent.id] = agent


def get_agent(agent_id: str | None) -> Agent | None:
    return _registry.get(agent_id or DEFAULT_AGENT)


def list_agents() -> list[Agent]:
    return list(_registry.values())


register(MentorAgent())
register(CoderAgent())


__all__ = [
    "Agent",
    "AgentContext",
    "DEFAULT_AGENT",
    "get_agent",
    "list_agents",
    "register",
]
