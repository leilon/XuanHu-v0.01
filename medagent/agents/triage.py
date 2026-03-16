from medagent.agents.base import BaseAgent
from medagent.schema import OrchestratorState


class TriageAgent(BaseAgent):
    name = "triage"

    def run(self, state: OrchestratorState) -> str:
        risk = state.risk_level
        if risk == "high":
            return "分诊结果：高风险，建议立即线下急诊。"
        if risk == "medium":
            return "分诊结果：中风险，建议24小时内就医评估。"
        return "分诊结果：低风险，可先居家观察并按需复诊。"

