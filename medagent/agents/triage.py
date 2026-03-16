from medagent.agents.base import BaseAgent
from medagent.schema import OrchestratorState
from medagent.services.clinical_pathway import build_triage_decision


class TriageAgent(BaseAgent):
    name = "triage"

    def run(self, state: OrchestratorState) -> str:
        decision = build_triage_decision(state)
        state.artifacts["triage"] = decision
        reasons = "；".join(decision["reasons"]) if decision["reasons"] else "待补充更多信息后再次分层"
        return (
            f"分诊结论：{decision['label']}\n"
            f"建议科室：{decision['department']}\n"
            f"判断依据：{reasons}"
        )
