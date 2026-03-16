from medagent.agents.base import BaseAgent
from medagent.schema import OrchestratorState
from medagent.services.tools import ReportParsingTool


class ReportAgent(BaseAgent):
    task_name = "report"
    display_name = "CangGong-Report"

    def __init__(self) -> None:
        self.report_tool = ReportParsingTool()

    def run(self, state: OrchestratorState) -> str:
        latest = state.messages[-1].content if state.messages else ""
        parsed = self.report_tool.parse_report_text(latest)
        return f"报告解读：{parsed['summary']}"
