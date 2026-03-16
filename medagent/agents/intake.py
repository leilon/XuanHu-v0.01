from medagent.agents.base import BaseAgent
from medagent.schema import OrchestratorState
from medagent.services.clinical_pathway import build_first_visit_prompt, build_intake_plan


class IntakeAgent(BaseAgent):
    task_name = "intake"
    display_name = "BianQue-Intake"

    def run(self, state: OrchestratorState) -> str:
        plan = build_intake_plan(state)
        plan["base_prompt"] = build_first_visit_prompt(state, plan)
        state.artifacts["intake"] = plan

        sections = [
            f"主诉聚类：{'、'.join(plan['chief_complaints'])}",
            "首程优先追问：",
            "；".join(plan["targeted_questions"]),
            "一般病史补充：",
            "；".join(plan["general_questions"]),
        ]
        if plan["special_notes"]:
            sections.extend(
                [
                    "特殊人群注意：",
                    "；".join(plan["special_notes"]),
                ]
            )
        if plan["memory_considerations"]:
            sections.extend(
                [
                    "长期记忆带来的重点：",
                    "；".join(plan["memory_considerations"]),
                ]
            )
        sections.extend(
            [
                "建议初步检查：",
                "；".join(plan["recommended_tests"]),
            ]
        )
        return "\n".join(sections)
