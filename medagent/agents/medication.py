from medagent.agents.base import BaseAgent
from medagent.schema import OrchestratorState
from medagent.services.tools import DrugInteractionTool


class MedicationAgent(BaseAgent):
    task_name = "medication"
    display_name = "ShenNong-Medication"

    def __init__(self) -> None:
        self.drug_tool = DrugInteractionTool()

    def run(self, state: OrchestratorState) -> str:
        allergies = set(state.user_context.allergies)
        profile = state.user_context.profile_facts or {}
        if isinstance(profile.get("allergies"), list):
            allergies.update(str(item) for item in profile["allergies"])

        warning_parts = [
            "用药建议必须建立在首程问诊和分诊完成后，不能在缺少病史时直接给药。",
        ]
        if "布洛芬" in allergies:
            warning_parts.append("长期记忆提示用户对布洛芬过敏，应避免推荐含布洛芬药物。")
        if state.user_context.sex and state.user_context.sex.lower() in {"female", "f", "女", "女性"}:
            warning_parts.append("若为育龄女性且后续涉及退热镇痛药、抗感染药或影像检查，应先确认是否妊娠。")

        interaction = self.drug_tool.check("布洛芬", "阿司匹林")
        warning_parts.append(f"示例相互作用提示：{interaction}")
        return " ".join(warning_parts)
