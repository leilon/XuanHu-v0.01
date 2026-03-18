from medagent.agents.base import BaseAgent
from medagent.schema import OrchestratorState
from medagent.services.clinical_pathway import build_triage_decision


class TriageAgent(BaseAgent):
    task_name = "triage"
    display_name = "HuaTuo-Triage"

    def _build_model_prompt(self, state: OrchestratorState, decision: dict) -> str:
        intake = state.artifacts.get("intake", {})
        return "\n".join(
            [
                f"用户原始提问：{self._user_query(state)}",
                f"系统风险等级：{state.risk_level}",
                f"主诉聚类：{'、'.join(decision.get('complaints', [])) or '未明确'}",
                f"建议分诊级别：{decision['label']}",
                f"建议科室：{decision['department']}",
                "判断依据候选：",
                *[f"- {item}" for item in decision["reasons"]],
                "首程问诊要点：",
                *[f"- {item}" for item in intake.get("targeted_questions", [])[:4]],
                "请严格按照系统提示词输出，给出分诊结论、去向和1到3条判断依据。",
            ]
        )

    def _fallback(self, decision: dict) -> str:
        reasons = "；".join(decision["reasons"]) if decision["reasons"] else "待补充更多信息后再次分层"
        return (
            f"分诊结论：{decision['label']}\n"
            f"建议科室：{decision['department']}\n"
            f"判断依据：{reasons}"
        )

    def run(self, state: OrchestratorState) -> str:
        decision = build_triage_decision(state)
        state.artifacts["triage"] = decision
        generated = self._generate(state, self._build_model_prompt(state, decision), max_new_tokens=220)
        if generated:
            return generated
        return self._fallback(decision)
