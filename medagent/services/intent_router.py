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
            "今天",
            "昨晚",
            "孩子",
            "宝宝",
            "家人",
            "老人",
            "我妈",
            "我爸",
        )
        self.symptom_tokens = (
            "疼",
            "痛",
            "痒",
            "烧",
            "咳",
            "吐",
            "恶心",
            "腹泻",
            "拉肚子",
            "头晕",
            "头痛",
            "胸闷",
            "胸痛",
            "气短",
            "喘",
            "出血",
            "发热",
            "发烧",
            "不舒服",
            "难受",
            "月经",
            "白带",
            "尿频",
            "尿痛",
        )
        self.report_tokens = (
            "报告",
            "化验",
            "化验单",
            "检查单",
            "检验单",
            "影像",
            "胸片",
            "ct",
            "mri",
            "彩超",
            "b超",
            "结果",
        )
        self.medication_tokens = (
            "药",
            "用药",
            "剂量",
            "禁忌",
            "相互作用",
            "退烧药",
            "消炎药",
            "抗生素",
            "能不能吃",
            "一起吃",
        )
        self.education_tokens = (
            "是什么",
            "什么意思",
            "怎么理解",
            "原理",
            "区别",
            "科普",
            "为什么",
            "会传染吗",
            "正常吗",
            "偏高说明什么",
            "偏低说明什么",
        )

    def route(self, query: str) -> RouteDecision:
        text = query.strip()
        lowered = text.lower()

        has_first_person = any(token in text for token in self.first_person_tokens)
        has_symptom = any(token in text for token in self.symptom_tokens)
        has_report = any(token in lowered for token in self.report_tokens)
        has_medication = any(token in text for token in self.medication_tokens)
        has_education = any(token in text for token in self.education_tokens)

        if has_report and (has_first_person or has_symptom):
            return RouteDecision(
                "report_followup",
                "用户带着自己的检查结果来问，并且伴随症状线索，优先走报告解读并补问关键病史。",
            )
        if has_report:
            return RouteDecision(
                "education_report",
                "更像泛化的报告或指标解释问题，优先走报告解释和知识补充。",
            )
        if has_medication and (has_first_person or has_symptom):
            return RouteDecision(
                "medication_followup",
                "用户在问自己的用药是否安全，需要结合既往史和当前症状处理。",
            )
        if has_first_person and has_symptom:
            return RouteDecision(
                "general_intake",
                "用户在描述自己或家人的当前不适，优先走首程问诊。",
            )
        if has_education or (has_medication and not has_first_person and not has_symptom):
            return RouteDecision(
                "patient_education",
                "更像疾病知识、指标解释或泛化问药问题，优先走科普解释。",
            )
        if has_symptom:
            return RouteDecision(
                "general_intake",
                "存在症状线索，先按首程问诊处理更稳妥。",
            )
        return RouteDecision(
            "patient_education",
            "默认按医学知识解释处理，如后续出现个人症状线索再升级为问诊。",
        )
