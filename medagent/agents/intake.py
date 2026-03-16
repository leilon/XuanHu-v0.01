from medagent.agents.base import BaseAgent
from medagent.schema import OrchestratorState


class IntakeAgent(BaseAgent):
    name = "intake"

    def run(self, state: OrchestratorState) -> str:
        user_text = state.messages[-1].content if state.messages else ""
        missing = []
        if "几天" not in user_text and "天" not in user_text:
            missing.append("症状持续时间")
        if "体温" not in user_text and "度" not in user_text:
            missing.append("体温范围")
        if "过敏" not in user_text:
            missing.append("药物过敏史")
        if not missing:
            return "病情采集较完整，可进入分诊与建议阶段。"
        return f"建议继续追问：{', '.join(missing)}。"

