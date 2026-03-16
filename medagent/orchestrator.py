from medagent.agents.education import EducationAgent
from medagent.agents.intake import IntakeAgent
from medagent.agents.medication import MedicationAgent
from medagent.agents.report import ReportAgent
from medagent.agents.triage import TriageAgent
from medagent.config import AppConfig
from medagent.schema import AgentMessage, OrchestratorState, UserContext
from medagent.services.adapter_bank import AdapterBank
from medagent.services.clinical_pathway import extract_profile_updates
from medagent.services.intent_router import IntentRouter
from medagent.services.memory import MemoryStore
from medagent.services.memory_fusion import MemoryFusionEngine
from medagent.services.rag import RAGService
from medagent.services.safety import SafetyGuard


class Orchestrator:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.memory = MemoryStore()
        self.rag = RAGService()
        self.safety = SafetyGuard(self.config)
        self.intent_router = IntentRouter()
        self.intake = IntakeAgent()
        self.triage = TriageAgent()
        self.medication = MedicationAgent()
        self.report = ReportAgent()
        self.education = EducationAgent()
        self.adapter_bank = AdapterBank(self.config.adapter_bank_dir)
        self.memory_fusion = MemoryFusionEngine(self.memory, self.adapter_bank)

    def _section_title(self, task: str) -> str:
        return self.config.branding.section_label(task)

    def _plan(self, intent: str) -> list[str]:
        if intent == "patient_education":
            return ["education", "rag_summary"]
        if intent == "education_report":
            return ["education", "rag_summary"]
        if intent == "report_followup":
            return ["report", "intake", "triage", "medication", "rag_summary"]
        if intent == "medication_followup":
            return ["intake", "medication", "triage", "rag_summary"]
        return ["intake", "triage", "medication", "rag_summary"]

    def _build_user_context(self, user_id: str, age: int | None, sex: str | None) -> UserContext:
        profile = self.memory.build_clinical_snapshot(user_id)
        final_age = age if age is not None else profile.get("age")
        final_sex = sex if sex is not None else profile.get("sex")
        return UserContext(
            user_id=user_id,
            age=int(final_age) if isinstance(final_age, str) and final_age.isdigit() else final_age,
            sex=str(final_sex) if final_sex is not None else None,
            chronic_history=self.memory.get_profile_list(user_id, "chronic_history"),
            allergies=self.memory.get_profile_list(user_id, "allergies"),
            profile_facts=profile,
            returning_user=bool(profile or self.memory.get_recent(user_id)),
        )

    def run(self, user_id: str, user_text: str, age: int | None = None, sex: str | None = None) -> str:
        ctx = self._build_user_context(user_id=user_id, age=age, sex=sex)
        state = OrchestratorState(
            user_context=ctx,
            messages=[AgentMessage(role="user", content=user_text)],
        )
        route = self.intent_router.route(user_text)
        state.intent = route.intent
        state.artifacts["route"] = {"intent": route.intent, "reason": route.reason}
        state.risk_level = self.safety.detect_risk(user_text)
        state.tasks = self._plan(route.intent)

        self.memory.append_turn(user_id, user_text)
        if ctx.age is not None:
            self.memory.upsert_profile_fact(user_id, "age", str(ctx.age))
        if ctx.sex is not None:
            self.memory.upsert_profile_fact(user_id, "sex", ctx.sex)

        sections: list[str] = []
        for task in state.tasks:
            if task == "intake":
                sections.append(f"[{self._section_title(task)}]\n{self.intake.run(state)}")
            elif task == "triage":
                sections.append(f"[{self._section_title(task)}]\n{self.triage.run(state)}")
            elif task == "report":
                sections.append(f"[{self._section_title(task)}]\n{self.report.run(state)}")
            elif task == "education":
                docs = self.rag.retrieve(user_text, top_k=3)
                state.artifacts["education_docs"] = [
                    {
                        "source": d.source,
                        "chunk": d.chunk,
                        "source_type": d.source_type,
                    }
                    for d in docs
                ]
                sections.append(f"[{self._section_title(task)}]\n{self.education.run(state)}")
            elif task == "medication":
                sections.append(f"[{self._section_title(task)}]\n{self.medication.run(state)}")
            elif task == "rag_summary":
                docs = self.rag.retrieve(user_text, top_k=2)
                state.evidence = [{"source": d.source, "chunk": d.chunk} for d in docs]
                evidence_text = "; ".join(f"{d.source}: {d.chunk}" for d in docs)
                sections.append(f"[{self._section_title(task)}]\n{evidence_text}")

        draft = "\n".join(sections)
        if self.config.require_citations and state.evidence:
            refs = "；".join(item["source"] for item in state.evidence)
            draft += f"\n[引用] {refs}"

        profile_updates = extract_profile_updates(state)
        for allergy in profile_updates.get("allergies", []):
            self.memory.append_profile_item(user_id, "allergies", allergy)
        for condition in profile_updates.get("chronic_history", []):
            self.memory.append_profile_item(user_id, "chronic_history", condition)
        for med in profile_updates.get("current_meds", []):
            self.memory.append_profile_item(user_id, "current_meds", med)
        if profile_updates.get("pregnancy_status"):
            self.memory.upsert_profile_fact(user_id, "pregnancy_status", profile_updates["pregnancy_status"])

        triage_label = state.artifacts.get("triage", {}).get("label", "未完成分诊")
        self.memory.upsert_profile_fact(user_id, "recent_assessment", triage_label)
        self.memory.append_episode(user_id, topic=state.tasks[0], content=draft[:500])

        if self.config.enable_memory_fusion:
            draft = self.memory_fusion.generate(user_id=user_id, query=user_text, draft=draft)
        state.final_response = self.safety.enforce(draft, state.risk_level)
        return state.final_response
