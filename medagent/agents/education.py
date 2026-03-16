from medagent.agents.base import BaseAgent
from medagent.schema import OrchestratorState


class EducationAgent(BaseAgent):
    task_name = "education"
    display_name = "LiShiZhen-Education"

    def run(self, state: OrchestratorState) -> str:
        docs = state.artifacts.get("education_docs", [])
        query = state.messages[-1].content if state.messages else ""
        if docs:
            first = docs[0]
            summary = first.get("chunk", "")
            source_type = first.get("source_type", "medical_reference")
            prefix = "这是一个偏医学科普/知识解释的问题。"
            if source_type == "medical_test":
                prefix = "这更像是在问检查或化验相关的解释。"
            elif source_type == "drug_label":
                prefix = "这更像是在问药物知识和注意事项。"
            return (
                f"{prefix}\n"
                f"围绕你的问题“{query}”，先给一个通俗解释：{summary}\n"
                "如果你想进一步了解适用人群、常见误区或需要就医的情况，我可以继续展开。"
            )
        return (
            "这更像一个医学知识解释问题，而不是首程问诊。"
            "如果你愿意，我可以先用通俗语言解释概念，再补充哪些情况下需要线下就诊。"
        )
