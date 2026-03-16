class DrugInteractionTool:
    def check(self, drug_a: str, drug_b: str) -> str:
        if {drug_a, drug_b} == {"布洛芬", "阿司匹林"}:
            return "存在胃肠道风险叠加，建议咨询医生。"
        return "未发现高风险相互作用（示例库）。"


class ReportParsingTool:
    def parse_report_text(self, report_text: str) -> dict[str, str]:
        # Placeholder parser for multimodal extension.
        if "白细胞" in report_text and "升高" in report_text:
            return {"summary": "提示可能存在炎症/感染信号，建议结合症状与医生评估。"}
        return {"summary": "报告未解析到明确异常关键词。"}

