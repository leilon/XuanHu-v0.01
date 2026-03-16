from medagent.agents.base import BaseAgent
from medagent.schema import OrchestratorState
from medagent.services.tools import DrugInteractionTool


class MedicationAgent(BaseAgent):
    name = "medication"

    def __init__(self) -> None:
        self.drug_tool = DrugInteractionTool()

    def run(self, state: OrchestratorState) -> str:
        interaction = self.drug_tool.check("布洛芬", "阿司匹林")
        return (
            "可选建议：对乙酰氨基酚或布洛芬用于退热止痛（需结合年龄、体重、基础疾病评估）；"
            f"相互作用提示：{interaction}"
        )

