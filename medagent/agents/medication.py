from medagent.agents.base import BaseAgent
from medagent.schema import OrchestratorState
from medagent.services.tools import DrugInteractionTool


class MedicationAgent(BaseAgent):
    task_name = "medication"
    display_name = "ShenNong-Medication"

    def __init__(self) -> None:
        self.drug_tool = DrugInteractionTool()

    def _build_model_prompt(
        self,
        state: OrchestratorState,
        *,
        allergies: list[str],
        memory_text: str,
        interaction: str,
    ) -> str:
        triage = state.artifacts.get("triage", {})
        docs = state.artifacts.get("knowledge_docs", [])
        lines = [
            f"用户问题：{self._user_query(state)}",
            f"当前分诊：{triage.get('label', '未完成分诊')}",
            f"已知过敏：{'、'.join(allergies) if allergies else '暂无'}",
            f"长期记忆：{memory_text or '暂无'}",
            f"工具提示：{interaction}",
        ]
        if docs:
            lines.append("知识依据：")
            for idx, doc in enumerate(docs[:2], start=1):
                source = doc.get("title") or doc.get("source") or "未知来源"
                lines.append(f"{idx}. [{source}] {doc.get('chunk', '')}")
        lines.append("请只给保守、安全、患者可执行的用药建议；如果信息不足，先补问，不给处方级方案。")
        return "\n".join(lines)

    def _fallback(self, state: OrchestratorState, *, allergies: list[str], memory_text: str, interaction: str) -> str:
        warning_parts = [
            "用药建议必须建立在病史、分诊和禁忌信息相对完整的前提下，不能在信息不足时直接给处方级方案。",
        ]
        if "布洛芬" in allergies:
            warning_parts.append("长期记忆提示用户对布洛芬过敏，应避免推荐含布洛芬药物。")
        if state.user_context.sex and state.user_context.sex.lower() in {"female", "f", "女", "女性"}:
            warning_parts.append("如果是育龄女性且后续涉及退热镇痛药、抗感染药或影像检查，应先确认是否妊娠。")
        warning_parts.append(f"工具提示：{interaction}")
        if memory_text:
            warning_parts.append(f"长期记忆：{memory_text}")
        return " ".join(warning_parts)

    def run(self, state: OrchestratorState) -> str:
        allergies = set(state.user_context.allergies)
        profile = state.user_context.profile_facts or {}
        if isinstance(profile.get("allergies"), list):
            allergies.update(str(item) for item in profile["allergies"])

        memory_hits = state.artifacts.get("memory_hits", [])
        memory_text = ""
        if memory_hits:
            first_hit = memory_hits[0]
            memory_text = first_hit.get("content", "") if isinstance(first_hit, dict) else str(first_hit)

        interaction = self.drug_tool.check("布洛芬", "阿司匹林")
        generated = self._generate(
            state,
            self._build_model_prompt(
                state,
                allergies=sorted(allergies),
                memory_text=memory_text,
                interaction=interaction,
            ),
            max_new_tokens=256,
        )
        if generated:
            return generated
        return self._fallback(
            state,
            allergies=sorted(allergies),
            memory_text=memory_text,
            interaction=interaction,
        )
