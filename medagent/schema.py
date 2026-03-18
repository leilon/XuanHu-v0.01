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
class QuestionCandidate:
    slot: str
    question: str
    priority: float = 0.5
    rationale: str = ""


@dataclass
class VisitRecord:
    visit_id: str
    user_id: str
    chief_complaint: str = ""
    history_of_present_illness: str = ""
    past_history: list[str] = field(default_factory=list)
    allergy_history: list[str] = field(default_factory=list)
    current_medications: list[str] = field(default_factory=list)
    epidemiology_history: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    recommended_tests: list[str] = field(default_factory=list)
    preliminary_assessment: str = ""
    triage_label: str = ""
    human_readable_summary: str = ""
    source_documents: list[dict[str, Any]] = field(default_factory=list)


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
    turn_index: int = 1
    question_queue: list[QuestionCandidate] = field(default_factory=list)
    asked_slots: list[str] = field(default_factory=list)
    filled_slots: dict[str, Any] = field(default_factory=dict)
    red_flags: list[str] = field(default_factory=list)
    stop_reason: str = ""
    visit_record: VisitRecord | None = None
