"""ChatService: the HTTP layer's handle on today's conversational agent. A thin pass-through - the
agent, its pipeline and its retriever are unchanged (pinned by tests/test_chat_contract.py)."""
from __future__ import annotations

from ..agent import AgentReply, HMSRouterAgent
from ..accumulator import AccumulatedState


class ChatService:
    def __init__(self, agent: HMSRouterAgent):
        self.agent = agent

    def handle(self, text: str, session_id: str | None) -> AgentReply:
        return self.agent.handle(text, session_id)

    def feedback(self, kind: str, **kwargs) -> dict:
        return self.agent.feedback.submit(kind, **kwargs)

    def snapshot(self, session_id: str) -> AccumulatedState:
        return self.agent.accumulator.snapshot(session_id)

    def clear(self, session_id: str) -> None:
        self.agent.accumulator.clear(session_id)
