from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouteDecision:
    intent: str
    reason: str


class IntentRouter:
    def __init__(self) -> None:
        self.first_person_tokens = (
            "我",
            "自己",
            "最近",
            "这两天",
            "这几天",
            "现在",
            "刚刚",
            "今天",
            "昨晚",
            "孩子",
            "家人",
            "老人",
        )
        self.symptom_tokens = (
            "疼",
            "痛",
            "闷",
            "堵",
            "喘",
            "咳",
            "发热",
            "发烧",
            "恶心",
            "呕吐",
            "腹泻",
            "头晕",
            "头痛",
            "出血",
            "皮疹",
            "不舒服",
            "难受",
        )
        self.report_tokens = ("报告", "化验", "影像", "胸片", "CT", "MRI", "彩超", "检查单", "检验单")
        self.medication_tokens = ("药", "用药", "剂量", "禁忌", "相互作用", "退烧药", "消炎药")
        self.education_tokens = (
            "是什么",
            "什么意思",
            "原理",
            "区别",
            "科普",
            "怎么理解",
            "为什么",
            "会传染吗",
            "正常吗",
            "偏高说明什么",
        )

    def route(self, query: str) -> RouteDecision:
        text = query.strip()
        lowered = text.lower()
        has_first_person = any(token in text for token in self.first_person_tokens)
        has_symptom = any(token in text for token in self.symptom_tokens)
        has_report = any(token.lower() in lowered for token in self.report_tokens)
        has_medication = any(token in text for token in self.medication_tokens)
        has_education = any(token in text for token in self.education_tokens)

        if has_report and has_first_person:
            return RouteDecision("report_followup", "用户带着自身报告/检查结果来问，优先报告解读并补首程问诊。")
        if has_report:
            return RouteDecision("education_report", "更像泛化的检查或报告知识解释。")
        if has_first_person and has_symptom:
            return RouteDecision("general_intake", "用户在描述自己或家人的当前症状，优先走首程问诊。")
        if has_medication and has_first_person:
            return RouteDecision("medication_followup", "用户在问自身用药，需要结合病史和安全边界。")
        if has_education or (has_medication and not has_first_person and not has_symptom):
            return RouteDecision("patient_education", "更像医学知识/科普解释问题。")
        if has_symptom:
            return RouteDecision("general_intake", "存在症状线索，优先按问诊处理。")
        return RouteDecision("patient_education", "默认按医学知识解释处理，必要时再升级到问诊。")
