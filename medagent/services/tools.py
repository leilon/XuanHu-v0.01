from __future__ import annotations

from dataclasses import dataclass
import itertools
import re
from typing import Any


@dataclass
class InteractionHit:
    drugs: tuple[str, str]
    severity: str
    summary: str
    recommendation: str


@dataclass
class ReportFinding:
    analyte: str
    direction: str
    value: float | None
    unit: str
    interpretation: str
    severity: str
    department: str


class DrugInteractionTool:
    DRUG_ALIASES = {
        "布洛芬": ("布洛芬", "ibuprofen", "芬必得"),
        "阿司匹林": ("阿司匹林", "aspirin"),
        "氯吡格雷": ("氯吡格雷", "clopidogrel", "波立维"),
        "华法林": ("华法林", "warfarin"),
        "对乙酰氨基酚": ("对乙酰氨基酚", "acetaminophen", "扑热息痛"),
        "阿莫西林": ("阿莫西林", "amoxicillin"),
        "二甲双胍": ("二甲双胍", "metformin"),
        "左氧氟沙星": ("左氧氟沙星", "levofloxacin"),
    }

    INTERACTION_RULES = {
        frozenset(("布洛芬", "阿司匹林")): InteractionHit(
            drugs=("布洛芬", "阿司匹林"),
            severity="high",
            summary="两者同用会增加胃肠道出血和胃黏膜损伤风险，还可能削弱阿司匹林部分抗血小板作用。",
            recommendation="若必须联用，应由医生评估心血管获益与出血风险，避免自行长期同服。",
        ),
        frozenset(("阿司匹林", "氯吡格雷")): InteractionHit(
            drugs=("阿司匹林", "氯吡格雷"),
            severity="high",
            summary="双联抗血小板常见于心血管疾病，但会明显增加出血风险。",
            recommendation="若病人近期有黑便、便血、牙龈出血或即将手术，应尽快线下复核处方。",
        ),
        frozenset(("华法林", "阿司匹林")): InteractionHit(
            drugs=("华法林", "阿司匹林"),
            severity="critical",
            summary="抗凝药与抗血小板药联用会显著增加严重出血风险。",
            recommendation="除非心内/血液科明确要求，不建议自行联用；如已联用且有出血迹象，应立即就医。",
        ),
        frozenset(("华法林", "布洛芬")): InteractionHit(
            drugs=("华法林", "布洛芬"),
            severity="critical",
            summary="NSAIDs 可增加华法林相关胃肠道出血和 INR 波动风险。",
            recommendation="镇痛退热优先考虑对乙酰氨基酚，并由医生评估 INR 监测需求。",
        ),
    }

    CONTRAINDICATION_HINTS = {
        "布洛芬": (
            ("胃溃疡", "消化道出血", "黑便"), "存在消化道出血/溃疡风险，布洛芬需谨慎甚至避免使用。"
        ),
        "布洛芬#renal": (
            ("肾功能异常", "肾病", "肌酐高"), "肾功能异常时 NSAIDs 可能进一步加重肾损伤。"
        ),
        "对乙酰氨基酚": (("肝病", "肝硬化", "转氨酶高"), "存在肝功能异常时，对乙酰氨基酚总剂量需要严格控制。"),
        "阿莫西林": (("青霉素过敏", "阿莫西林过敏"), "若对青霉素/阿莫西林过敏，应避免再次使用该类药物。"),
        "左氧氟沙星": (("怀孕", "妊娠", "儿童"), "妊娠期和儿童一般不优先自行使用喹诺酮类药物。"),
    }

    OTC_HINTS = (
        ("发热", "头痛", "咽痛", "肌肉酸痛"),
        ("咳嗽", "黄痰", "咽痛"),
        ("腹泻", "腹痛"),
    )

    def _normalize_drug(self, name: str) -> str:
        lowered = name.strip().lower()
        for canonical, aliases in self.DRUG_ALIASES.items():
            if any(alias.lower() == lowered for alias in aliases):
                return canonical
        return name.strip()

    def extract_drugs(self, text: str) -> list[str]:
        lowered = text.lower()
        found: list[str] = []
        for canonical, aliases in self.DRUG_ALIASES.items():
            if any(alias.lower() in lowered for alias in aliases):
                found.append(canonical)
        return list(dict.fromkeys(found))

    def check_pair(self, drug_a: str, drug_b: str) -> InteractionHit | None:
        normalized = frozenset((self._normalize_drug(drug_a), self._normalize_drug(drug_b)))
        return self.INTERACTION_RULES.get(normalized)

    def check(self, drug_a: str, drug_b: str) -> str:
        hit = self.check_pair(drug_a, drug_b)
        if hit:
            return f"{hit.summary} {hit.recommendation}"
        return "当前示例相互作用库中未命中高风险组合，但正式用药仍需结合处方、肝肾功能和出血风险评估。"

    def _profile_text(self, profile: dict[str, Any], allergies: set[str]) -> str:
        tokens: list[str] = []
        for value in profile.values():
            if isinstance(value, list):
                tokens.extend(str(item) for item in value)
            elif value:
                tokens.append(str(value))
        tokens.extend(allergies)
        return " ".join(tokens)

    def screen_context(
        self,
        query: str,
        profile: dict[str, Any],
        allergies: set[str],
        sex: str | None = None,
        age: int | None = None,
    ) -> dict[str, Any]:
        mentioned_drugs = self.extract_drugs(query)
        current_meds = {
            self._normalize_drug(str(item))
            for item in profile.get("current_meds", [])
            if str(item).strip()
        }
        all_drugs = list(dict.fromkeys(mentioned_drugs + list(current_meds)))
        hits = [
            hit
            for pair in itertools.combinations(sorted(set(all_drugs)), 2)
            if (hit := self.check_pair(pair[0], pair[1])) is not None
        ]

        profile_text = self._profile_text(profile, allergies) + " " + query
        warnings: list[str] = []
        for key, (tokens, message) in self.CONTRAINDICATION_HINTS.items():
            drug_name = key.split("#", 1)[0]
            if drug_name not in all_drugs and not any(token in query for token in tokens):
                continue
            if any(token in profile_text for token in tokens):
                warnings.append(message)

        if sex and str(sex).lower() in {"female", "f", "女", "女性"} and 12 <= (age or 30) <= 55:
            if any(drug in all_drugs for drug in ("布洛芬", "左氧氟沙星", "阿莫西林")):
                warnings.append("育龄女性如涉及止痛药、抗感染药或影像检查，应先确认是否妊娠。")

        allergies_text = " ".join(allergies)
        if "布洛芬" in all_drugs and "布洛芬" in allergies_text:
            warnings.append("长期记忆提示用户对布洛芬过敏，应避免再次推荐该药。")
        if "阿莫西林" in all_drugs and ("青霉素" in allergies_text or "阿莫西林" in allergies_text):
            warnings.append("长期记忆提示用户存在青霉素/阿莫西林相关过敏史，应回避该类药物。")

        followup_questions: list[str] = []
        if any(token in query for token in self.OTC_HINTS[0]) and "发热" in query or "发烧" in query:
            followup_questions.extend(["最高体温是多少", "有没有肝病或消化道出血史", "是否已经服过退烧药"])
        if "咳嗽" in query or "痰" in query:
            followup_questions.extend(["是干咳还是有痰", "有没有气促、胸闷或胸痛", "是否伴高热"])
        if "腹痛" in query or "腹泻" in query:
            followup_questions.extend(["大便次数和性状如何", "有没有脱水、黑便或便血", "是否进食不洁食物"])

        otc_options: list[str] = []
        if "发热" in query or "发烧" in query:
            otc_options.append("若仅为短期退热镇痛且无明显肝病，可考虑对乙酰氨基酚，但仍需核对剂量和过敏史。")
            if "胃溃疡" not in profile_text and "肾病" not in profile_text and "布洛芬" not in allergies_text:
                otc_options.append("若无消化道出血、肾病或 NSAIDs 过敏史，布洛芬也可作为候选，但需先确认是否妊娠。")
        if "咽痛" in query or "鼻塞" in query:
            otc_options.append("呼吸道症状初期更重要的是补液、休息和症状观察，不建议在病史不清时自行叠加多种感冒药。")

        return {
            "mentioned_drugs": all_drugs,
            "interaction_hits": hits,
            "warnings": list(dict.fromkeys(warnings)),
            "followup_questions": list(dict.fromkeys(followup_questions)),
            "otc_options": otc_options,
        }


class ReportParsingTool:
    ANALYTE_RULES = [
        {
            "name": "白细胞",
            "aliases": ("白细胞", "WBC"),
            "unit": "x10^9/L",
            "low": 3.5,
            "high": 9.5,
            "critical_high": 15.0,
            "critical_low": 2.0,
            "high_msg": "提示感染、炎症或应激反应可能性增高。",
            "low_msg": "提示白细胞偏低，需结合感染史、用药史和免疫状态评估。",
            "department": "呼吸内科/感染科",
        },
        {
            "name": "中性粒细胞",
            "aliases": ("中性粒细胞", "NEUT"),
            "unit": "%",
            "low": 40.0,
            "high": 75.0,
            "critical_high": 85.0,
            "critical_low": 20.0,
            "high_msg": "常见于细菌感染、应激或炎症反应。",
            "low_msg": "偏低时需结合病毒感染、骨髓抑制或药物因素分析。",
            "department": "感染科/血液科",
        },
        {
            "name": "血红蛋白",
            "aliases": ("血红蛋白", "Hb", "HGB"),
            "unit": "g/L",
            "low": 115.0,
            "high": 150.0,
            "critical_high": 180.0,
            "critical_low": 70.0,
            "high_msg": "偏高时需结合脱水、慢性缺氧等情况判断。",
            "low_msg": "提示贫血可能，需要结合出血史、月经史和营养状况评估。",
            "department": "血液科/全科",
        },
        {
            "name": "血小板",
            "aliases": ("血小板", "PLT"),
            "unit": "x10^9/L",
            "low": 125.0,
            "high": 350.0,
            "critical_high": 600.0,
            "critical_low": 50.0,
            "high_msg": "升高可见于炎症反应或骨髓增殖性疾病。",
            "low_msg": "偏低时需注意出血风险，结合皮下出血、牙龈出血和感染评估。",
            "department": "血液科",
        },
        {
            "name": "C反应蛋白",
            "aliases": ("C反应蛋白", "CRP"),
            "unit": "mg/L",
            "low": 0.0,
            "high": 8.0,
            "critical_high": 100.0,
            "critical_low": None,
            "high_msg": "明显升高提示体内炎症或感染活动增强。",
            "low_msg": "",
            "department": "感染科/呼吸内科",
        },
        {
            "name": "丙氨酸氨基转移酶",
            "aliases": ("ALT", "谷丙转氨酶", "丙氨酸氨基转移酶"),
            "unit": "U/L",
            "low": 0.0,
            "high": 40.0,
            "critical_high": 200.0,
            "critical_low": None,
            "high_msg": "提示肝细胞损伤可能，需结合药物、酒精、脂肪肝或病毒性肝炎评估。",
            "low_msg": "",
            "department": "消化内科/肝病科",
        },
        {
            "name": "天门冬氨酸氨基转移酶",
            "aliases": ("AST", "谷草转氨酶", "天门冬氨酸氨基转移酶"),
            "unit": "U/L",
            "low": 0.0,
            "high": 40.0,
            "critical_high": 200.0,
            "critical_low": None,
            "high_msg": "升高提示肝脏、心肌或肌肉损伤可能，需要结合 ALT 和症状判断。",
            "low_msg": "",
            "department": "消化内科/肝病科",
        },
        {
            "name": "肌酐",
            "aliases": ("肌酐", "Cr", "CREA"),
            "unit": "umol/L",
            "low": 44.0,
            "high": 133.0,
            "critical_high": 200.0,
            "critical_low": None,
            "high_msg": "升高提示肾功能受损可能，需要结合尿量、既往肾病和近期用药分析。",
            "low_msg": "",
            "department": "肾内科",
        },
        {
            "name": "血糖",
            "aliases": ("血糖", "葡萄糖", "GLU"),
            "unit": "mmol/L",
            "low": 3.9,
            "high": 6.1,
            "critical_high": 16.7,
            "critical_low": 2.8,
            "high_msg": "升高提示高血糖，需要区分空腹、餐后或糖尿病相关情况。",
            "low_msg": "偏低需警惕低血糖，结合出汗、手抖、意识改变等症状判断。",
            "department": "内分泌科",
        },
        {
            "name": "β-hCG",
            "aliases": ("β-hCG", "HCG", "绒毛膜促性腺激素"),
            "unit": "IU/L",
            "low": 0.0,
            "high": 5.0,
            "critical_high": None,
            "critical_low": None,
            "high_msg": "若为育龄女性且 HCG 升高，需要结合停经、腹痛或阴道流血评估是否妊娠相关。",
            "low_msg": "",
            "department": "妇科/产科",
        },
    ]

    def _extract_value(self, text: str, aliases: tuple[str, ...]) -> tuple[float | None, str]:
        for alias in aliases:
            pattern = rf"{re.escape(alias)}[^0-9\-<>]*(?P<value>[<>]?\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z/%\^\-0-9μumolmgLIU]*)"
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            raw = match.group("value")
            clean = raw.lstrip("<>")
            try:
                return float(clean), match.group("unit") or ""
            except ValueError:
                continue
        return None, ""

    def _explicit_direction(self, text: str, aliases: tuple[str, ...]) -> str | None:
        for alias in aliases:
            up = re.search(rf"{re.escape(alias)}.*?(升高|偏高|↑)", text, flags=re.IGNORECASE)
            if up:
                return "high"
            down = re.search(rf"{re.escape(alias)}.*?(降低|偏低|↓)", text, flags=re.IGNORECASE)
            if down:
                return "low"
        return None

    def parse_report_text(self, report_text: str) -> dict[str, Any]:
        text = " ".join(report_text.split())
        findings: list[ReportFinding] = []

        for rule in self.ANALYTE_RULES:
            value, unit = self._extract_value(text, rule["aliases"])
            direction = self._explicit_direction(text, rule["aliases"])
            severity = "routine"
            interpretation = ""

            if value is not None:
                low = rule["low"]
                high = rule["high"]
                critical_low = rule["critical_low"]
                critical_high = rule["critical_high"]
                if critical_high is not None and value >= critical_high:
                    direction = "high"
                    severity = "urgent"
                    interpretation = rule["high_msg"]
                elif critical_low is not None and value <= critical_low:
                    direction = "low"
                    severity = "urgent"
                    interpretation = rule["low_msg"]
                elif value > high:
                    direction = "high"
                    severity = "followup"
                    interpretation = rule["high_msg"]
                elif value < low:
                    direction = "low"
                    severity = "followup"
                    interpretation = rule["low_msg"]
            elif direction == "high":
                interpretation = rule["high_msg"]
                severity = "followup"
            elif direction == "low":
                interpretation = rule["low_msg"]
                severity = "followup"

            if direction:
                findings.append(
                    ReportFinding(
                        analyte=rule["name"],
                        direction=direction,
                        value=value,
                        unit=unit or rule["unit"],
                        interpretation=interpretation or "该指标异常，建议结合原始报告和临床症状进一步判断。",
                        severity=severity,
                        department=rule["department"],
                    )
                )

        if not findings:
            return {
                "summary": "目前没有从文本里稳定提取到可解释的关键异常指标。更适合补充原始化验单、参考范围或直接上传报告图片后再分析。",
                "findings": [],
                "risk_level": "uncertain",
                "department": "相关专科门诊",
                "recommended_actions": [
                    "补充原始报告中的指标名称、数值和参考范围",
                    "如伴发热、胸痛、腹痛、呼吸困难等症状，应结合症状优先分诊",
                ],
            }

        urgent = any(item.severity == "urgent" for item in findings)
        main_departments = list(dict.fromkeys(item.department for item in findings))
        risk_level = "urgent" if urgent else "followup"

        finding_summaries = []
        for item in findings[:4]:
            direction = "升高" if item.direction == "high" else "降低"
            value_text = f"{item.value:g}{item.unit}" if item.value is not None else direction
            finding_summaries.append(f"{item.analyte}{direction}（{value_text}）")

        summary = (
            f"报告分析：主要关注 {'、'.join(finding_summaries)}。"
            f"{findings[0].interpretation}"
        )
        actions = [
            "结合本次主诉、起病时间和伴随症状判断这些异常是否与当前不适一致",
            "若原始报告还有参考范围或其他异常指标，建议一并补充",
        ]
        if urgent:
            actions.append("若同时伴明显气促、持续高热、胸痛、黑便、意识改变等，建议尽快急诊或当天线下就医。")
        else:
            actions.append("若症状持续或逐渐加重，建议尽快到对应专科门诊完善复查。")

        return {
            "summary": summary,
            "findings": findings,
            "risk_level": risk_level,
            "department": "；".join(main_departments),
            "recommended_actions": actions,
        }
