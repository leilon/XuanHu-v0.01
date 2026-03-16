from medagent.agents.intake import IntakeAgent
from medagent.agents.medication import MedicationAgent
from medagent.agents.report import ReportAgent
from medagent.agents.triage import TriageAgent
from medagent.config import AppConfig
from medagent.schema import AgentMessage, OrchestratorState, UserContext
from medagent.services.adapter_bank import AdapterBank
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
        self.intake = IntakeAgent()
        self.triage = TriageAgent()
        self.medication = MedicationAgent()
        self.report = ReportAgent()
        self.adapter_bank = AdapterBank(self.config.adapter_bank_dir)
        self.memory_fusion = MemoryFusionEngine(self.memory, self.adapter_bank)

    def _plan(self, user_text: str) -> list[str]:
        tasks = ["intake", "triage", "medication", "rag_summary"]
        if any(k in user_text for k in ("报告", "化验", "影像", "白细胞", "CT", "MRI")):
            tasks.insert(2, "report")
        return tasks

    def run(self, user_id: str, user_text: str, age: int | None = None, sex: str | None = None) -> str:
        ctx = UserContext(user_id=user_id, age=age, sex=sex)
        state = OrchestratorState(
            user_context=ctx,
            messages=[AgentMessage(role="user", content=user_text)],
        )
        state.risk_level = self.safety.detect_risk(user_text)
        state.tasks = self._plan(user_text)

        self.memory.append_turn(user_id, user_text)
        if age is not None:
            self.memory.upsert_profile_fact(user_id, "age", str(age))
        if sex is not None:
            self.memory.upsert_profile_fact(user_id, "sex", sex)

        sections: list[str] = []
        for task in state.tasks:
            if task == "intake":
                sections.append(f"[病情采集] {self.intake.run(state)}")
            elif task == "triage":
                sections.append(f"[分诊建议] {self.triage.run(state)}")
            elif task == "report":
                sections.append(f"[报告分析] {self.report.run(state)}")
            elif task == "medication":
                sections.append(f"[用药建议] {self.medication.run(state)}")
            elif task == "rag_summary":
                docs = self.rag.retrieve(user_text, top_k=2)
                state.evidence = [{"source": d.source, "chunk": d.chunk} for d in docs]
                evidence_text = "; ".join(f"{d.source}: {d.chunk}" for d in docs)
                sections.append(f"[知识依据] {evidence_text}")

        draft = "\n".join(sections)
        if self.config.require_citations and state.evidence:
            refs = "，".join(item["source"] for item in state.evidence)
            draft += f"\n[引用] {refs}"

        self.memory.append_episode(user_id, topic=state.tasks[0], content=draft[:280])

        if self.config.enable_memory_fusion:
            draft = self.memory_fusion.generate(user_id=user_id, query=user_text, draft=draft)
        state.final_response = self.safety.enforce(draft, state.risk_level)
        return state.final_response

