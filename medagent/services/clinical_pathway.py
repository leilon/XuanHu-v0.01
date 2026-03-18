from __future__ import annotations

from typing import Any

from medagent.schema import OrchestratorState


FEMALE_TOKENS = {"female", "f", "女", "woman", "女性"}
MALE_TOKENS = {"male", "m", "男", "man", "男性"}

COMPLAINT_RULES = [
    ("胸痛/胸闷", ("胸痛", "胸闷", "心口痛", "压榨痛", "胸口疼", "胸口压着疼", "胸口堵", "堵得慌", "胸口发紧")),
    ("发热/呼吸道症状", ("发热", "发烧", "咳嗽", "咳痰", "咽痛", "鼻塞", "流涕", "气促", "喘", "喘不过气", "呼吸困难")),
    ("腹痛/消化道症状", ("腹痛", "肚子痛", "腹泻", "呕吐", "恶心", "黑便", "便血", "发黑发亮", "大便发黑")),
    ("头痛/神经系统症状", ("头痛", "头晕", "抽搐", "意识不清", "言语不清", "说话不清", "肢体无力")),
    ("泌尿生殖系统症状", ("尿频", "尿急", "尿痛", "血尿", "阴道流血", "白带", "下腹痛")),
    ("皮疹/过敏", ("皮疹", "荨麻疹", "瘙痒", "过敏", "红疹", "疹子", "嘴唇发胀", "喉咙发紧")),
    ("报告解读", ("报告", "化验", "检验", "白细胞", "胸片", "彩超", "CT", "MRI", "影像")),
]

EMERGENCY_KEYWORDS = (
    "胸痛",
    "胸口压着疼",
    "胸口发紧",
    "呼吸困难",
    "喘不上气",
    "喘不过气",
    "大汗",
    "出汗",
    "意识不清",
    "抽搐",
    "便血",
    "黑便",
    "发黑发亮",
    "呕血",
    "嘴唇发胀",
    "喉咙发紧",
    "单侧无力",
    "言语不清",
    "说话不清",
)

ADMISSION_HINTS = (
    "高热不退",
    "持续高热",
    "反复呕吐",
    "脱水",
    "不能进食",
    "明显气促",
    "持续胸痛",
    "黑便",
    "便血",
    "发黑发亮",
    "反复黑便",
    "发虚",
)

HIGH_RISK_COMORBIDITIES = (
    "糖尿病",
    "高血压",
    "冠心病",
    "心衰",
    "慢阻肺",
    "哮喘",
    "肿瘤",
    "肾病",
    "肝硬化",
    "免疫抑制",
    "妊娠",
)

COMMON_ALLERGIES = ("青霉素", "头孢", "布洛芬", "阿莫西林", "阿司匹林")
COMMON_LONG_TERM_MEDS = ("二甲双胍", "胰岛素", "阿司匹林", "氯吡格雷", "缬沙坦", "氨氯地平")


def _normalize_text(text: str) -> str:
    return text.strip().lower()


def _normalize_sex(sex: str | None) -> str:
    if not sex:
        return "unknown"
    text = _normalize_text(sex)
    if text in FEMALE_TOKENS:
        return "female"
    if text in MALE_TOKENS:
        return "male"
    return "unknown"


def _is_reproductive_female(sex: str | None, age: int | None) -> bool:
    if _normalize_sex(sex) != "female":
        return False
    if age is None:
        return True
    return 12 <= age <= 55


def _infer_chief_complaints(text: str) -> list[str]:
    hits: list[str] = []
    for label, keywords in COMPLAINT_RULES:
        if any(keyword in text for keyword in keywords):
            hits.append(label)
    return hits or ["未明确主诉"]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def _profile_list(profile: dict[str, Any], key: str) -> list[str]:
    value = profile.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        return [item.strip() for item in text.split(",") if item.strip()]
    return [text]


def build_first_visit_prompt(state: OrchestratorState, intake_plan: dict[str, Any]) -> str:
    ctx = state.user_context
    memory_facts = intake_plan.get("memory_considerations", [])
    memory_text = "；".join(memory_facts) if memory_facts else "无既往长期记忆可用"
    return (
        "你是互联网医疗场景下的首程问诊 Agent。请严格按真实临床首程问诊思路工作：\n"
        "1. 先围绕用户本次最困扰的主诉，按时间线追问起病时间、演变过程、严重程度和伴随症状。\n"
        "2. 再围绕主诉可能涉及的系统做定向追问，不要一上来给结论或开药。\n"
        "3. 必问一般病史：既往基础病、手术史、过敏史、当前用药、家族史、饮食作息、流行病学接触史。\n"
        "4. 仅在与症状、用药或检查相关时追问冶游史/性接触史；仅在育龄女性且与当前决策相关时追问末次月经和妊娠可能，男性不要问怀孕。\n"
        "5. 如果用户是老用户，优先结合长期记忆中的慢病、过敏史、既往同类发作和既往检查结果，不要重复问已经明确的信息，但要确认是否有变化。\n"
        "6. 最后给出门诊/急诊/是否需住院评估的分层建议，以及初步检查建议，不要把推测说成确定诊断。\n\n"
        f"当前主诉聚类：{ '、'.join(intake_plan['chief_complaints']) }\n"
        f"优先追问：{ '；'.join(intake_plan['targeted_questions']) }\n"
        f"一般病史：{ '；'.join(intake_plan['general_questions']) }\n"
        f"长期记忆：{memory_text}\n"
        f"建议初步检查：{ '；'.join(intake_plan['recommended_tests']) }\n"
        "输出应先给追问项，再给初步分诊，再给检查建议。"
    )


def build_intake_plan(state: OrchestratorState) -> dict[str, Any]:
    query = state.messages[-1].content if state.messages else ""
    ctx = state.user_context
    profile = ctx.profile_facts or {}
    complaints = _infer_chief_complaints(query)

    targeted: list[str] = [
        "先确认本次最困扰的症状、起病时间、持续时长、是否逐渐加重",
        "确认症状严重程度，是否影响进食、睡眠、活动或工作",
    ]
    general = [
        "既往基础疾病、手术史、住院史",
        "当前正在使用的药物、近期是否自行用药",
        "药物过敏史和食物过敏史",
        "近期饮食、睡眠、饮酒吸烟情况",
        "流行病学接触史、旅行史、同住者是否有类似症状",
    ]
    tests: list[str] = []
    memory_notes: list[str] = []
    special_notes: list[str] = []

    if "胸痛/胸闷" in complaints:
        targeted.extend(
            [
                "胸痛部位、性质、是否向左肩背部放射、与活动是否相关",
                "是否伴大汗、呼吸困难、心悸、濒死感或晕厥",
                "既往是否有高血压、冠心病、血脂异常或吸烟史",
            ]
        )
        tests.extend(["心电图", "肌钙蛋白/心肌酶", "血压和血氧监测", "必要时胸部影像"])

    if "发热/呼吸道症状" in complaints:
        targeted.extend(
            [
                "最高体温、发热持续天数、是否寒战",
                "咳嗽是干咳还是有痰，痰色如何，是否伴胸闷气促",
                "是否有咽痛、鼻塞、流涕、肌肉酸痛、接触发热患者",
            ]
        )
        tests.extend(["血常规", "CRP/降钙素原", "流感/新冠抗原或核酸", "必要时胸片或肺部CT"])

    if "腹痛/消化道症状" in complaints:
        targeted.extend(
            [
                "腹痛部位、性质、是否放射、与进食或排便是否相关",
                "是否伴恶心、呕吐、腹泻、便血、黑便、发热",
                "近期饮食是否异常，是否进食不洁食物或大量饮酒",
            ]
        )
        tests.extend(["血常规", "尿常规", "肝肾功能和电解质", "腹部超声或腹部CT"])

    if "头痛/神经系统症状" in complaints:
        targeted.extend(
            [
                "头痛/神经症状是否突然出现，是否为最严重一次",
                "是否伴发热、呕吐、视物模糊、单侧无力、说话困难",
                "是否有外伤史、既往偏头痛或高血压病史",
            ]
        )
        tests.extend(["血压测量", "神经系统查体", "必要时头颅CT/MRI"])

    if "泌尿生殖系统症状" in complaints:
        targeted.extend(
            [
                "是否有尿频、尿急、尿痛、血尿、分泌物异常或下腹坠痛",
                "症状与月经、性行为、饮水量是否相关",
            ]
        )
        general.append("如症状相关，再追问性接触史/冶游史")
        tests.extend(["尿常规", "尿培养", "盆腔超声或泌尿系超声"])

    if "皮疹/过敏" in complaints:
        targeted.extend(
            [
                "皮疹出现时间、分布、是否瘙痒、是否伴呼吸道症状",
                "近期是否接触新食物、新药物、化妆品或环境刺激物",
            ]
        )
        tests.extend(["过敏原评估", "必要时血常规或皮肤科检查"])

    if "报告解读" in complaints:
        targeted.extend(
            [
                "确认报告类型、检查日期、异常指标名称和参考范围",
                "结合症状追问该指标相关的不适和既往类似检查结果",
            ]
        )
        tests.extend(["原始报告图片/OCR核对", "必要时重复相关实验室检查"])

    if _is_reproductive_female(ctx.sex, ctx.age):
        special_notes.append("如涉及用药、影像检查、下腹痛、恶心呕吐或阴道流血，需追问末次月经和妊娠可能")
        if any(label in complaints for label in ("腹痛/消化道症状", "泌尿生殖系统症状", "发热/呼吸道症状")):
            targeted.append("如为育龄女性，请补问末次月经、是否可能妊娠、是否有异常阴道流血")
            tests.append("必要时尿/血HCG")

    allergies = _profile_list(profile, "allergies")
    chronic = _profile_list(profile, "chronic_history")
    current_meds = _profile_list(profile, "current_meds")
    prior_assessment = str(profile.get("recent_assessment", "")).strip()
    if ctx.returning_user or profile:
        if allergies:
            memory_notes.append(f"长期记忆提示既往过敏：{'、'.join(allergies)}，本轮先确认是否有新增过敏")
        if chronic:
            memory_notes.append(f"长期记忆提示基础病：{'、'.join(chronic)}，需判断本次是否为原病加重或并发症")
        if current_meds:
            memory_notes.append(f"长期记忆提示当前/长期用药：{'、'.join(current_meds)}，需确认最近是否调整剂量")
        if prior_assessment:
            memory_notes.append(f"可参考上次就诊结论：{prior_assessment}")
        memory_notes.append("如果这是老问题复发，追问与上次相比是加重、减轻还是表现不同")

    return {
        "chief_complaints": complaints,
        "targeted_questions": _dedupe(targeted),
        "general_questions": _dedupe(general),
        "special_notes": _dedupe(special_notes),
        "memory_considerations": _dedupe(memory_notes),
        "recommended_tests": _dedupe(tests) or ["待补充主诉后决定检查"],
    }


def _query_has_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _profile_has_high_risk(profile: dict[str, Any]) -> bool:
    chronic = " ".join(_profile_list(profile, "chronic_history"))
    return any(keyword in chronic for keyword in HIGH_RISK_COMORBIDITIES)


def _recommend_department(complaints: list[str], sex: str | None) -> str:
    if "胸痛/胸闷" in complaints:
        return "急诊/心内科"
    if "头痛/神经系统症状" in complaints:
        return "急诊/神经内科"
    if "腹痛/消化道症状" in complaints:
        return "消化内科/急诊外科"
    if "泌尿生殖系统症状" in complaints:
        if _normalize_sex(sex) == "female":
            return "妇科/泌尿外科"
        return "泌尿外科"
    if "发热/呼吸道症状" in complaints:
        return "发热门诊/呼吸内科"
    if "皮疹/过敏" in complaints:
        return "皮肤科/过敏反应门诊"
    return "全科/普通门诊"


def extract_profile_updates(state: OrchestratorState) -> dict[str, Any]:
    query = state.messages[-1].content if state.messages else ""
    updates: dict[str, Any] = {"allergies": [], "chronic_history": [], "current_meds": []}

    for item in HIGH_RISK_COMORBIDITIES:
        if item in query:
            updates["chronic_history"].append(item)

    for item in COMMON_ALLERGIES:
        if f"对{item}过敏" in query or (item in query and "过敏" in query):
            updates["allergies"].append(item)

    for item in COMMON_LONG_TERM_MEDS:
        if item in query and any(token in query for token in ("长期吃", "一直吃", "常年吃", "正在吃")):
            updates["current_meds"].append(item)

    if "怀孕" in query or "孕" in query:
        updates["pregnancy_status"] = "用户本轮提到可能存在妊娠相关情况"

    return updates


def build_triage_decision(state: OrchestratorState) -> dict[str, Any]:
    query = state.messages[-1].content if state.messages else ""
    full_user_history = "；".join(item.content for item in state.messages if item.role == "user")
    complaints = state.artifacts.get("intake", {}).get("chief_complaints") or _infer_chief_complaints(full_user_history or query)
    profile = state.user_context.profile_facts or {}
    reasons: list[str] = []
    level = "routine_outpatient"
    label = "普通门诊评估"

    emergency = _query_has_any(full_user_history or query, EMERGENCY_KEYWORDS) or bool(state.red_flags)
    consider_admission = _query_has_any(full_user_history or query, ADMISSION_HINTS) or (
        state.risk_level == "high" and _profile_has_high_risk(profile)
    )

    if emergency:
        level = "emergency"
        label = "立即急诊/必要时呼叫120"
        reasons.append("当前描述包含典型急危重危险信号")
    elif consider_admission:
        level = "consider_admission"
        label = "建议急诊评估，并由线下医生判断是否留观或收治"
        reasons.append("存在持续高热、明显气促、反复呕吐、便血等需要线下密切观察的信号")
    elif state.risk_level == "medium" or _profile_has_high_risk(profile):
        level = "urgent_outpatient"
        label = "建议24小时内专科/发热门诊就诊"
        reasons.append("目前更像需要尽快线下评估，但未达到明确急救指征")
    else:
        level = "routine_outpatient"
        label = "普通门诊或居家观察后按需复诊"
        reasons.append("当前更像低到中低风险问题，可先完成首程问诊再决定是否线下就诊")

    if _is_reproductive_female(state.user_context.sex, state.user_context.age) and any(
        item in full_user_history for item in ("腹痛", "阴道流血", "停经", "恶心", "呕吐")
    ):
        reasons.append("育龄女性伴腹痛/出血/恶心时需排除妊娠相关问题")
        if level == "routine_outpatient":
            level = "urgent_outpatient"
            label = "建议尽快妇科/急诊评估"

    return {
        "level": level,
        "label": label,
        "department": _recommend_department(complaints, state.user_context.sex),
        "reasons": reasons,
        "complaints": complaints,
    }
