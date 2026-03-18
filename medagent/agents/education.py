from medagent.agents.base import BaseAgent
from medagent.schema import OrchestratorState


class EducationAgent(BaseAgent):
    task_name = "education"
    display_name = "LiShiZhen-Education"

    def _build_model_prompt(self, state: OrchestratorState, docs: list[dict]) -> str:
        lines = [
            f"用户问题：{self._user_query(state)}",
            "命中的知识依据：",
        ]
        for idx, doc in enumerate(docs[:3], start=1):
            source = doc.get("title") or doc.get("source") or "未知来源"
            lines.append(f"{idx}. [{source}] {doc.get('chunk', '')}")
        lines.append("请基于这些材料做患者友好的医学解释，不要直接复制原文。")
        return "\n".join(lines)

    def _fallback(self, state: OrchestratorState, docs: list[dict]) -> str:
        query = self._user_query(state)
        if not docs:
            return (
                "这个问题更像医学解释类问题。"
                "当前还没有命中合适的知识依据，我可以先给你做通俗解释，之后再补更可靠的参考资料。"
            )

        first = docs[0]
        summary = first.get("chunk", "")
        source_type = first.get("source_type", "medical_reference")
        if source_type == "drug_label":
            prefix = "这是偏用药说明类的问题。"
        elif source_type == "medical_test":
            prefix = "这是偏检查指标解释类的问题。"
        else:
            prefix = "这是偏疾病或医学知识解释类的问题。"

        return (
            f"{prefix}\n"
            f"围绕你的问题“{query}”，先给一个通俗解释：{summary}\n"
            "如果你愿意，我可以继续补充：常见误区、哪些情况需要就医，以及和你当前症状最相关的风险点。"
        )

    def run(self, state: OrchestratorState) -> str:
        docs = state.artifacts.get("knowledge_docs", []) or state.artifacts.get("education_docs", [])
        generated = self._generate(state, self._build_model_prompt(state, docs), max_new_tokens=320)
        if generated:
            return generated
        return self._fallback(state, docs)
