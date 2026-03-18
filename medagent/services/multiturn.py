from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import re
from uuid import uuid4

from medagent.schema import OrchestratorState, QuestionCandidate, VisitRecord


TEMPERATURE_RE = re.compile(r"(\d{2}(?:\.\d+)?)\s*度")
DURATION_RE = re.compile(r"(\d+)\s*(小时|天|周|个月|年)")

RED_FLAG_PATTERNS = {
    "胸痛": "胸痛",
    "胸口压着疼": "胸痛",
    "胸口发紧": "胸闷明显",
    "胸闷": "胸闷明显",
    "呼吸困难": "呼吸困难",
    "喘不上气": "喘不上气",
    "喘不过气": "喘不上气",
    "抽搐": "抽搐",
    "意识不清": "意识不清",
    "昏迷": "昏迷",
    "便血": "便血",
    "黑便": "黑便",
    "发黑发亮": "黑便",
    "呕血": "呕血",
    "单侧无力": "单侧无力",
    "言语不清": "言语不清",
    "高热不退": "高热不退",
    "嘴唇发胀": "疑似过敏性反应",
    "喉咙发紧": "疑似过敏性反应",
}

DISPOSITION_ACCEPT_PATTERNS = (
    "我现在去急诊",
    "我现在去医院",
    "我马上去医院",
    "我马上去急诊",
    "我这就去医院",
    "我这就去急诊",
    "我今天就去看",
    "我先去把这些检查做了",
    "好的，我现在去急诊",
    "好的，那我先去把这些检查做了",
    "明白了，我今天就去看",
)

DISPOSITION_DECLINE_PATTERNS = (
    "先不去",
    "暂时不去",
    "先观察",
)

USER_END_PATTERNS = (
    "先这样",
    "先到这",
    "没有了",
    "没有别的了",
    "先不问了",
    "谢谢",
    "好的谢谢",
)

TRIAGE_PRIORITY = {
    "routine_outpatient": 0,
    "urgent_outpatient": 1,
    "consider_admission": 2,
    "emergency": 3,
}


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        value = str(item).strip()
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def new_visit_id(user_id: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{user_id}_{stamp}_{uuid4().hex[:6]}"


def update_state_from_user_message(state: OrchestratorState) -> None:
    if not state.messages:
        return

    latest = state.messages[-1].content.strip()
    filled = dict(state.filled_slots)

    if state.user_context.age is not None:
        filled.setdefault("age", state.user_context.age)
    if state.user_context.sex:
        filled.setdefault("sex", state.user_context.sex)

    temp_match = TEMPERATURE_RE.search(latest)
    if temp_match:
        filled["temperature"] = temp_match.group(1)

    duration_match = DURATION_RE.search(latest)
    if duration_match:
        filled["duration"] = f"{duration_match.group(1)}{duration_match.group(2)}"

    if "干咳" in latest:
        filled["cough_type"] = "干咳"
    elif "有痰" in latest or "黄痰" in latest or "白痰" in latest:
        filled["cough_type"] = "有痰"

    if "胸痛" in latest or "胸闷" in latest:
        filled["chest_symptom"] = "有"
    if "喘" in latest or "气促" in latest or "呼吸困难" in latest:
        filled["dyspnea"] = "有"
    if "呕吐" in latest or "吐了" in latest:
        filled["vomiting"] = "有"
    if "腹痛" in latest or "肚子痛" in latest:
        filled["abdominal_pain"] = "有"
    if "怀孕" in latest or "孕" in latest or "末次月经" in latest:
        filled["pregnancy_related"] = latest[:80]
    if "没精神" in latest or "意识" in latest or "昏" in latest:
        filled["mental_status"] = latest[:80]

    red_flags = list(state.red_flags)
    for token, label in RED_FLAG_PATTERNS.items():
        if token in latest:
            red_flags.append(label)

    state.filled_slots = filled
    state.red_flags = _dedupe(red_flags)


def select_followup_questions(state: OrchestratorState, limit: int = 2) -> list[QuestionCandidate]:
    asked_questions = set(state.asked_slots)
    ranked = sorted(state.question_queue, key=lambda item: item.priority, reverse=True)
    selected: list[QuestionCandidate] = []
    remaining: list[QuestionCandidate] = []

    for candidate in ranked:
        if candidate.question in asked_questions:
            continue
        if len(selected) < limit:
            selected.append(candidate)
        else:
            remaining.append(candidate)

    state.question_queue = remaining
    state.asked_slots = _dedupe(state.asked_slots + [item.question for item in selected])
    return selected


def _user_accepts_disposition(text: str) -> bool:
    latest = text.strip()
    if not latest:
        return False
    if any(token in latest for token in DISPOSITION_DECLINE_PATTERNS):
        return False
    return any(token in latest for token in DISPOSITION_ACCEPT_PATTERNS)


def _user_ended_conversation(text: str) -> bool:
    latest = text.strip()
    if not latest:
        return False
    return any(token in latest for token in USER_END_PATTERNS)


def should_stop_visit(state: OrchestratorState, max_turns: int = 6) -> tuple[bool, str]:
    triage = state.artifacts.get("triage", {})
    triage_level = str(triage.get("level", "")).strip()
    latest_user_text = ""
    for item in reversed(state.messages):
        if item.role == "user":
            latest_user_text = item.content.strip()
            break

    if _user_accepts_disposition(latest_user_text):
        if triage_level in {"urgent_outpatient", "consider_admission", "emergency"}:
            return True, "patient_accepted_disposition"

    if _user_ended_conversation(latest_user_text):
        return True, "user_ended_conversation"
    if state.turn_index >= max_turns:
        return True, "max_turns_reached"
    if len(state.filled_slots) >= 5 and state.turn_index >= 3:
        return True, "enough_information_collected"
    if not state.question_queue and state.turn_index >= 2:
        return True, "question_queue_exhausted"
    return False, ""


def build_preliminary_assessment(state: OrchestratorState) -> str:
    intake = state.artifacts.get("intake", {})
    complaints = intake.get("chief_complaints", [])
    triage = state.artifacts.get("triage", {})
    triage_level = str(triage.get("level", "")).strip()

    if triage_level == "emergency":
        return "当前更像需要立即急诊排查的高风险问题，应先线下急诊评估，再决定进一步诊断。"
    if triage_level == "consider_admission":
        return "当前更像需要急诊评估并考虑留观或收治的问题，应优先完成线下评估。"

    if "发热/呼吸道症状" in complaints:
        return "考虑呼吸道感染相关问题，需结合体温、气促程度和影像检查进一步判断。"
    if "胸痛/胸闷" in complaints:
        return "考虑心血管或呼吸系统高风险问题，需尽快线下评估。"
    if "腹痛/消化道症状" in complaints:
        return "考虑消化系统或腹部急症可能，需结合腹痛部位和伴随症状进一步判断。"
    if "头痛/神经系统症状" in complaints:
        return "考虑神经系统或感染相关问题，需结合神经系统危险信号继续判断。"
    if "泌尿生殖系统症状" in complaints:
        return "考虑泌尿或妇科相关问题，需结合尿路症状、月经和妊娠情况继续评估。"
    if "皮疹/过敏" in complaints:
        return "考虑皮肤或过敏相关问题，需结合诱因和呼吸循环症状评估。"
    return "目前更像需要继续补病史后再做初步判断的问题。"


def build_human_readable_summary(record: VisitRecord, state: OrchestratorState) -> str:
    missing_slots = [
        key
        for key in ("temperature", "duration", "cough_type", "dyspnea", "mental_status", "pregnancy_related")
        if key not in state.filled_slots
    ]
    lines = [
        f"首程摘要：{record.chief_complaint or '未完整记录主诉'}",
        f"分诊结论：{record.triage_label or '待进一步评估'}",
        f"初步判断：{record.preliminary_assessment or '待进一步评估'}",
    ]
    if record.red_flags:
        lines.append(f"重点风险：{'、'.join(record.red_flags)}")
    if record.recommended_tests:
        lines.append(f"建议检查：{'、'.join(record.recommended_tests[:4])}")
    if missing_slots:
        lines.append(f"仍待补充：{'、'.join(missing_slots)}")
    return "\n".join(lines)


def refresh_visit_record(state: OrchestratorState) -> VisitRecord:
    record = state.visit_record or VisitRecord(visit_id=new_visit_id(state.user_context.user_id), user_id=state.user_context.user_id)
    user_messages = [item.content.strip() for item in state.messages if item.role == "user" and item.content.strip()]
    intake = state.artifacts.get("intake", {})
    triage = state.artifacts.get("triage", {})
    new_triage_level = str(triage.get("level", "")).strip()
    new_triage_label = str(triage.get("label", "")).strip()

    record.chief_complaint = record.chief_complaint or (user_messages[0] if user_messages else "")
    record.history_of_present_illness = "；".join(user_messages[-6:])[:1000]
    record.past_history = list(state.user_context.chronic_history)
    record.allergy_history = list(state.user_context.allergies)
    meds = state.user_context.profile_facts.get("current_meds", [])
    if isinstance(meds, str):
        meds = [meds]
    record.current_medications = [str(item) for item in meds]
    record.red_flags = list(state.red_flags)
    record.recommended_tests = list(intake.get("recommended_tests", [])[:4])
    record.preliminary_assessment = build_preliminary_assessment(state)
    current_priority = TRIAGE_PRIORITY.get(record.triage_level, -1)
    new_priority = TRIAGE_PRIORITY.get(new_triage_level, -1)
    if new_priority >= current_priority:
        record.triage_level = new_triage_level
        record.triage_label = new_triage_label
    record.source_documents = list(state.artifacts.get("knowledge_docs", [])[:3])
    record.human_readable_summary = build_human_readable_summary(record, state)
    state.visit_record = record
    return record


def serialize_visit_record(record: VisitRecord) -> dict:
    return asdict(record)
