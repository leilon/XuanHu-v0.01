class DrugInteractionTool:
    def check(self, drug_a: str, drug_b: str) -> str:
        if {drug_a, drug_b} == {"布洛芬", "阿司匹林"}:
            return "两者同用会增加胃肠道不良反应和出血风险，需由医生判断是否联合使用。"
        return "示例库中未发现明确高风险相互作用，但正式用药仍需结合病史核对。"


class ReportParsingTool:
    def parse_report_text(self, report_text: str) -> dict[str, str]:
        if "白细胞" in report_text and "升高" in report_text:
            return {"summary": "提示可能存在感染或炎症反应，需要结合发热、咳嗽、腹痛等症状综合判断。"}
        if "白细胞" in report_text and "降低" in report_text:
            return {"summary": "提示白细胞偏低，需结合感染史、用药史和既往血常规变化综合评估。"}
        return {"summary": "暂未从文本中识别到明确异常关键词，建议结合原始报告图片和参考范围再次核对。"}
