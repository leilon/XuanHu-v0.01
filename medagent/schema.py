from dataclasses import dataclass, field
from typing import Any


@dataclass
class UserContext:
    user_id: str
    age: int | None = None
    sex: str | None = None
    chronic_history: list[str] = field(default_factory=list)
    allergies: list[str] = field(default_factory=list)
    profile_facts: dict[str, Any] = field(default_factory=dict)
    returning_user: bool = False


@dataclass
class AgentMessage:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorState:
    user_context: UserContext
    messages: list[AgentMessage] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    intent: str = "general_intake"
    evidence: list[dict[str, Any]] = field(default_factory=list)
    risk_level: str = "unknown"
    final_response: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
