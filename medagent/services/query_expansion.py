from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class QueryExpansion:
    original_query: str
    rewritten_query: str
    hyde_document: str
    intent: str
    symptom_tags: list[str]


class QueryExpander:
    def __init__(self) -> None:
        self.intent_keywords = {
            "report_qa": ("报告", "化验", "影像", "胸片", "ct", "mri", "彩超"),
            "medication_qa": ("药", "用药", "剂量", "禁忌", "相互作用"),
            "triage": ("急诊", "门诊", "科室", "严重", "危险"),
        }
        self.symptom_keywords = (
            "发热",
            "咳嗽",
            "胸痛",
            "胸闷",
            "胸口堵",
            "堵得慌",
            "腹痛",
            "头痛",
            "呕吐",
            "腹泻",
            "呼吸困难",
            "喘",
            "皮疹",
            "尿频",
            "阴道流血",
        )
        self.translation_map = {
            "发热": "fever",
            "发烧": "fever",
            "咳嗽": "cough",
            "咳痰": "productive cough",
            "胸痛": "chest pain",
            "胸闷": "chest tightness",
            "胸口堵": "chest tightness",
            "堵得慌": "chest tightness",
            "喘": "shortness of breath",
            "呼吸困难": "shortness of breath",
            "腹痛": "abdominal pain",
            "肚子痛": "abdominal pain",
            "头痛": "headache",
            "头晕": "dizziness",
            "呕吐": "vomiting",
            "恶心": "nausea",
            "腹泻": "diarrhea",
            "尿频": "frequent urination",
            "尿痛": "painful urination",
            "阴道流血": "vaginal bleeding",
            "皮疹": "rash",
            "肝功能": "liver function test",
            "血常规": "complete blood count",
            "尿常规": "urinalysis",
            "肺炎": "pneumonia",
            "支原体肺炎": "mycoplasma pneumonia",
            "化验单": "lab test report",
            "检验单": "lab test report",
            "报告": "medical report",
        }

    def _infer_intent(self, query: str) -> str:
        lowered = query.lower()
        for intent, keywords in self.intent_keywords.items():
            if any(keyword in lowered for keyword in keywords):
                return intent
        return "general_intake"

    def _extract_symptoms(self, query: str) -> list[str]:
        found = [keyword for keyword in self.symptom_keywords if keyword in query]
        if found:
            return found
        fallback = re.findall(r"[\u4e00-\u9fff]{2,6}", query)
        return fallback[:4]

    def expand(self, query: str) -> QueryExpansion:
        cleaned = re.sub(r"\s+", " ", query).strip()
        intent = self._infer_intent(cleaned)
        symptoms = self._extract_symptoms(cleaned)
        symptom_text = "、".join(symptoms) if symptoms else "待进一步追问的主诉"
        english_terms = [
            english for chinese, english in self.translation_map.items() if chinese in cleaned
        ]
        english_text = ", ".join(dict.fromkeys(english_terms)) if english_terms else "medical history and symptoms"
        rewritten = (
            f"任务={intent}；主诉={symptom_text}；"
            f"英文检索提示={english_text}；"
            "检索目标=危险信号、常见鉴别方向、首批检查建议、患者友好解释"
        )
        hyde = (
            "这是一段用于检索的假设性临床摘要。"
            f"用户当前主诉可能与{symptom_text}相关，"
            f"英文提示包括：{english_text}。"
            "需要查找与诊断学相关的首程问诊要点、危险信号、初步检查、"
            "是否需要急诊或尽快门诊，以及面向患者的解释方式。"
        )
        return QueryExpansion(
            original_query=cleaned,
            rewritten_query=rewritten,
            hyde_document=hyde,
            intent=intent,
            symptom_tags=symptoms,
        )
