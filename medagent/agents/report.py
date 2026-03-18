from medagent.agents.base import BaseAgent
from medagent.schema import OrchestratorState
from medagent.services.tools import ReportParsingTool


class ReportAgent(BaseAgent):
    task_name = "report"
    display_name = "CangGong-Report"

    def __init__(self) -> None:
        self.report_tool = ReportParsingTool()

    def _build_model_prompt(self, state: OrchestratorState, parsed: dict) -> str:
        docs = state.artifacts.get("knowledge_docs", [])
        lines = [
            f"用户描述或报告文本：{self._user_query(state)}",
            f"工具抽取摘要：{parsed.get('summary', '暂无')}",
        ]
        if docs:
            lines.append("知识依据：")
            for idx, doc in enumerate(docs[:2], start=1):
                source = doc.get("title") or doc.get("source") or "未知来源"
                lines.append(f"{idx}. [{source}] {doc.get('chunk', '')}")
        if state.artifacts.get("image_path"):
            lines.append("本轮附带了报告图片，请结合图片内容与文本一起解释。")
        lines.append("请输出三部分：关键异常、通俗解释、下一步建议。")
        return "\n".join(lines)

    def _fallback(self, state: OrchestratorState, parsed: dict) -> str:
        docs = state.artifacts.get("knowledge_docs", [])
        evidence_hint = ""
        if docs:
            evidence_hint = f" 资料端可参考：{docs[0].get('chunk', '')}"
        return f"报告解读：{parsed['summary']}。{evidence_hint}".strip()

    def run(self, state: OrchestratorState) -> str:
        latest = state.messages[-1].content if state.messages else ""
        parsed = self.report_tool.parse_report_text(latest)
        state.artifacts["report_parsed"] = parsed
        generated = self._generate(
            state,
            self._build_model_prompt(state, parsed),
            image_path=state.artifacts.get("image_path"),
            max_new_tokens=320,
        )
        if generated:
            return generated
        return self._fallback(state, parsed)
