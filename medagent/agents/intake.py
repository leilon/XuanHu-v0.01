from medagent.agents.base import BaseAgent
from medagent.schema import OrchestratorState, QuestionCandidate
from medagent.services.clinical_pathway import build_first_visit_prompt, build_intake_plan


class IntakeAgent(BaseAgent):
    task_name = "intake"
    display_name = "BianQue-Intake"

    def _build_model_prompt(self, state: OrchestratorState, plan: dict) -> str:
        ctx = state.user_context
        memory_lines = plan["memory_considerations"] or ["暂无可用长期记忆"]
        special_lines = plan["special_notes"] or ["暂无特殊人群提醒"]
        queue_lines = [
            candidate.question
            for candidate in sorted(state.question_queue, key=lambda item: item.priority, reverse=True)[:4]
        ]
        return "\n".join(
            [
                f"用户原始提问：{self._user_query(state)}",
                f"当前轮次：第 {state.turn_index} 轮",
                f"年龄：{ctx.age if ctx.age is not None else '未提供'}",
                f"性别：{ctx.sex or '未提供'}",
                f"主诉聚类：{'、'.join(plan['chief_complaints'])}",
                f"已知槽位：{state.filled_slots or '暂无'}",
                f"历史红旗：{state.red_flags or '暂无'}",
                "候选追问：",
                *[f"- {item}" for item in (queue_lines or plan["targeted_questions"][:6])],
                "一般病史候选：",
                *[f"- {item}" for item in plan["general_questions"][:5]],
                "特殊人群提醒：",
                *[f"- {item}" for item in special_lines[:3]],
                "长期记忆提醒：",
                *[f"- {item}" for item in memory_lines[:3]],
                "建议检查候选：",
                *[f"- {item}" for item in plan["recommended_tests"][:4]],
                "请严格按照系统提示词输出，保持简洁，本轮只问1到2个问题，不要编号，不要写多余提示语。",
            ]
        )

    def _update_question_queue(self, state: OrchestratorState, plan: dict) -> None:
        existing = {item.question for item in state.question_queue}
        already_asked = set(state.asked_slots)
        queue = list(state.question_queue)
        candidates: list[QuestionCandidate] = []
        for idx, question in enumerate(plan["targeted_questions"][:6]):
            if question in existing or question in already_asked:
                continue
            priority = max(0.95 - idx * 0.08, 0.45)
            candidates.append(
                QuestionCandidate(
                    slot=f"targeted_{idx}",
                    question=question,
                    priority=priority,
                    rationale="targeted",
                )
            )
        for idx, question in enumerate(plan["general_questions"][:4]):
            if question in existing or question in already_asked:
                continue
            priority = max(0.55 - idx * 0.05, 0.25)
            candidates.append(
                QuestionCandidate(
                    slot=f"general_{idx}",
                    question=question,
                    priority=priority,
                    rationale="general",
                )
            )
        queue.extend(candidates)
        state.question_queue = sorted(queue, key=lambda item: item.priority, reverse=True)[:10]

    def _fallback(self, plan: dict) -> str:
        lines = ["本轮追问："]
        for question in plan["targeted_questions"][:2]:
            lines.append(f"- {question}")
        if plan["memory_considerations"]:
            lines.append(f"记忆提醒：{'；'.join(plan['memory_considerations'][:2])}")
        lines.append(f"建议检查：{'；'.join(plan['recommended_tests'][:3])}")
        return "\n".join(lines)

    def run(self, state: OrchestratorState) -> str:
        plan = build_intake_plan(state)
        plan["base_prompt"] = build_first_visit_prompt(state, plan)
        state.artifacts["intake"] = plan
        self._update_question_queue(state, plan)
        generated = self._generate(state, self._build_model_prompt(state, plan), max_new_tokens=320)
        if generated:
            return generated
        return self._fallback(plan)
