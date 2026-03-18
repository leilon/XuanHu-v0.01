from medagent.agents.education import EducationAgent
from medagent.agents.intake import IntakeAgent
from medagent.agents.medication import MedicationAgent
from medagent.agents.report import ReportAgent
from medagent.agents.triage import TriageAgent
from medagent.config import AppConfig
from medagent.schema import AgentMessage, OrchestratorState, UserContext
from medagent.services.adapter_bank import AdapterBank
from medagent.services.clinical_pathway import extract_profile_updates
from medagent.services.huatuo_runtime import HuatuoRuntime
from medagent.services.intent_router import IntentRouter
from medagent.services.memory import MemoryStore
from medagent.services.memory_fusion import MemoryFusionEngine
from medagent.services.prompt_registry import PromptRegistry
from medagent.services.rag import RAGBundle, RAGService, RetrievedMemory
from medagent.services.safety import SafetyGuard


class Orchestrator:
    PROMPT_NAME_MAP = {
        "intake": "bianque_intake",
        "triage": "huatuo_triage",
        "report": "canggong_report",
        "education": "lishizhen_education",
        "medication": "shennong_medication",
        "orchestrator": "qibo_orchestrator",
        "memory": "simiao_memory",
    }

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
        self.prompt_registry = PromptRegistry()
        self.llm_runtime = HuatuoRuntime(model_ref=self.config.default_profile.name)

    def _section_title(self, task: str) -> str:
        return self.config.branding.section_label(task)

    def _plan(self, intent: str) -> list[str]:
        if intent == "patient_education":
            return ["education", "rag_summary"]
        if intent == "education_report":
            return ["report", "education", "rag_summary"]
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

    def _load_prompt(self, task: str) -> str:
        prompt_name = self.PROMPT_NAME_MAP.get(task)
        if not prompt_name:
            return ""
        try:
            return self.prompt_registry.load(prompt_name)
        except FileNotFoundError:
            return ""

    def _serialize_rag_bundle(self, bundle: RAGBundle) -> dict[str, list[dict[str, str | float]]]:
        return {
            "knowledge_docs": [
                {
                    "source": doc.source,
                    "title": doc.title,
                    "chunk": doc.chunk,
                    "source_type": doc.source_type,
                    "url": doc.url,
                    "score": doc.score,
                }
                for doc in bundle.knowledge_docs
            ],
            "memory_hits": [
                {
                    "source": item.source,
                    "content": item.content,
                    "memory_type": item.memory_type,
                    "score": item.score,
                }
                for item in bundle.memory_hits
            ],
        }

    def _format_knowledge_docs(self, bundle: RAGBundle) -> str:
        if not bundle.knowledge_docs:
            return "暂无可引用的医学资料。"
        lines = ["资料端："]
        for doc in bundle.knowledge_docs:
            source_label = doc.title or doc.source
            lines.append(f"- {source_label}: {doc.chunk}")
        return "\n".join(lines)

    def _format_memory_hits(self, memory_hits: list[RetrievedMemory]) -> str:
        if not memory_hits:
            return "用户端：暂无可用长期记忆。"
        lines = ["用户端："]
        for item in memory_hits:
            lines.append(f"- {item.memory_type} | {item.content}")
        return "\n".join(lines)

    def _format_rag_summary(self, bundle: RAGBundle) -> str:
        return f"{self._format_memory_hits(bundle.memory_hits)}\n{self._format_knowledge_docs(bundle)}"

    def run(
        self,
        user_id: str,
        user_text: str,
        age: int | None = None,
        sex: str | None = None,
        image_path: str | None = None,
    ) -> str:
        ctx = self._build_user_context(user_id=user_id, age=age, sex=sex)
        state = OrchestratorState(
            user_context=ctx,
            messages=[AgentMessage(role="user", content=user_text)],
        )
        route = self.intent_router.route(user_text)
        state.intent = route.intent
        state.artifacts["route"] = {"intent": route.intent, "reason": route.reason}
        state.artifacts["prompt_profiles"] = {
            "orchestrator": self._load_prompt("orchestrator"),
            "memory": self._load_prompt("memory"),
        }
        state.artifacts["llm_runtime"] = self.llm_runtime
        state.artifacts["image_path"] = image_path
        state.risk_level = self.safety.detect_risk(user_text)
        state.tasks = self._plan(route.intent)

        self.memory.append_turn(user_id, user_text)
        if ctx.age is not None:
            self.memory.upsert_profile_fact(user_id, "age", str(ctx.age))
        if ctx.sex is not None:
            self.memory.upsert_profile_fact(user_id, "sex", ctx.sex)

        bundle = self.rag.build_bundle(user_id=user_id, query=user_text, memory_store=self.memory, knowledge_top_k=3, memory_top_k=3)
        state.artifacts["rag"] = self._serialize_rag_bundle(bundle)
        state.artifacts["knowledge_docs"] = state.artifacts["rag"]["knowledge_docs"]
        state.artifacts["education_docs"] = state.artifacts["rag"]["knowledge_docs"]
        state.artifacts["memory_hits"] = state.artifacts["rag"]["memory_hits"]

        sections: list[str] = []
        for task in state.tasks:
            state.artifacts["active_prompt"] = self._load_prompt(task)
            state.artifacts["prompt_profiles"][task] = state.artifacts["active_prompt"]
            if task == "intake":
                sections.append(f"[{self._section_title(task)}]\n{self.intake.run(state)}")
            elif task == "triage":
                sections.append(f"[{self._section_title(task)}]\n{self.triage.run(state)}")
            elif task == "report":
                sections.append(f"[{self._section_title(task)}]\n{self.report.run(state)}")
            elif task == "education":
                sections.append(f"[{self._section_title(task)}]\n{self.education.run(state)}")
            elif task == "medication":
                sections.append(f"[{self._section_title(task)}]\n{self.medication.run(state)}")
            elif task == "rag_summary":
                state.evidence = [{"source": doc.source, "chunk": doc.chunk} for doc in bundle.knowledge_docs]
                sections.append(f"[{self._section_title(task)}]\n{self._format_rag_summary(bundle)}")

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
        first_task = state.tasks[0] if state.tasks else "general"
        self.memory.append_episode(user_id, topic=first_task, content=draft[:500])

        if self.config.enable_memory_fusion:
            draft = self.memory_fusion.generate(user_id=user_id, query=user_text, draft=draft)
        state.final_response = self.safety.enforce(draft, state.risk_level)
        return state.final_response
