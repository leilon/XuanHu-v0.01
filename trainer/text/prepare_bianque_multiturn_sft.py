#!/usr/bin/env python
"""Prepare concise BianQue multiturn SFT data from Huatuo dialogues."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer.text.prepare_bianque_intake_sft import (  # noqa: E402
    FEMALE_HINTS,
    HIGH_RISK_HINTS,
    MALE_HINTS,
    PEDIATRIC_HINTS,
    _infer_case,
    _looks_like_intake_question,
)

HUATUO_DEFAULT = (
    "/root/autodl-tmp/medagent/datasets/sft/"
    "FreedomIntelligence__HuatuoGPT-sft-data-v1/HuatuoGPT_sft_data_v1.jsonl"
)

ROLE_PREFIXES = {
    "问：": "user",
    "答：": "assistant",
    "患者：": "user",
    "用户：": "user",
    "病人：": "user",
    "医生：": "assistant",
    "问:": "user",
    "答:": "assistant",
    "患者:": "user",
    "用户:": "user",
    "病人:": "user",
    "医生:": "assistant",
    "闂細": "user",
    "绛旓細": "assistant",
    "鎮ｈ€?": "user",
    "鍖荤敓:": "assistant",
}

SLOT_CONFIG: dict[str, dict[str, tuple[str, ...] | str]] = {
    "onset": {
        "question": "症状从什么时候开始的，是持续还是一阵一阵",
        "keywords": ("开始", "多久", "几天", "多久了", "持续", "反复", "一阵一阵", "病程"),
    },
    "temperature": {
        "question": "体温大概多少，最高到过多少度",
        "keywords": ("体温", "发热", "发烧", "高烧", "39", "38", "低烧"),
    },
    "cough_sputum": {
        "question": "有没有咳嗽、咳痰，痰是什么颜色",
        "keywords": ("咳嗽", "咳痰", "痰", "黄痰", "白痰", "干咳"),
    },
    "dyspnea": {
        "question": "有没有胸闷、气短或呼吸困难",
        "keywords": ("气短", "胸闷", "呼吸困难", "喘", "憋", "上不来气"),
    },
    "exposure_history": {
        "question": "最近有没有接触生病的人，或去过人多密闭的地方",
        "keywords": ("接触", "流行病", "旅行", "外出", "家里有人", "同住", "聚集"),
    },
    "pain_location": {
        "question": "疼痛具体在什么位置",
        "keywords": ("上腹", "下腹", "左下腹", "右下腹", "肚脐", "胃", "部位", "哪里痛"),
    },
    "pain_pattern": {
        "question": "是持续痛还是阵发加重，疼得厉害吗",
        "keywords": ("持续", "阵发", "绞痛", "加重", "疼得厉害", "痛一阵", "隐痛"),
    },
    "vomiting_diarrhea": {
        "question": "有没有恶心、呕吐或腹泻",
        "keywords": ("恶心", "呕吐", "腹泻", "拉肚子", "稀便", "大便次数"),
    },
    "stool_bleeding": {
        "question": "有没有黑便、便血或大便颜色明显异常",
        "keywords": ("黑便", "便血", "血便", "柏油样", "大便发黑", "血丝"),
    },
    "diet_history": {
        "question": "最近饮食有没有不洁、辛辣生冷或明显变化",
        "keywords": ("饮食", "生冷", "辛辣", "不洁", "外卖", "火锅", "海鲜"),
    },
    "duration": {
        "question": "这种情况持续多久了，最近是在变重还是缓解",
        "keywords": ("多久", "持续", "几天", "几周", "几个月", "加重", "缓解"),
    },
    "exertion_relation": {
        "question": "活动后会不会更明显，休息能缓解吗",
        "keywords": ("活动", "运动", "走路", "爬楼", "休息", "劳累"),
    },
    "radiation": {
        "question": "疼痛会不会放射到后背、肩膀或左臂",
        "keywords": ("放射", "后背", "肩膀", "左臂", "颈部"),
    },
    "sweating_syncope": {
        "question": "有没有大汗、头晕、快要晕倒或真的晕过去",
        "keywords": ("大汗", "头晕", "晕", "晕倒", "眼前发黑", "出汗"),
    },
    "consciousness": {
        "question": "现在意识清楚吗，反应和平时比怎么样",
        "keywords": ("意识", "昏迷", "叫不醒", "反应差", "嗜睡", "精神差"),
    },
    "seizure": {
        "question": "有没有抽搐，抽了多久，发作后能不能叫应",
        "keywords": ("抽搐", "抽", "惊厥", "发作", "四肢抖", "翻白眼"),
    },
    "speech_gait": {
        "question": "有没有说话不清、走路不稳或单侧无力",
        "keywords": ("说话不清", "口齿不清", "走路不稳", "单侧无力", "偏瘫", "站不稳"),
    },
    "head_trauma_history": {
        "question": "最近有没有摔倒、撞头或头部外伤",
        "keywords": ("摔倒", "撞头", "头部外伤", "磕到", "受伤"),
    },
    "dysuria": {
        "question": "有没有尿频、尿急、尿痛或排尿不舒服",
        "keywords": ("尿频", "尿急", "尿痛", "排尿", "小便", "尿不尽"),
    },
    "fever_backpain": {
        "question": "有没有发热、腰痛或小腹坠痛",
        "keywords": ("发热", "发烧", "腰痛", "后腰", "小腹坠痛"),
    },
    "discharge_bleeding": {
        "question": "有没有白带异常、阴道流血或分泌物增多",
        "keywords": ("白带", "阴道流血", "流血", "分泌物", "异味"),
    },
    "pregnancy_menses": {
        "question": "末次月经是什么时候，近期有没有怀孕可能",
        "keywords": ("末次月经", "月经", "停经", "怀孕", "妊娠", "早孕"),
    },
    "itching_swelling": {
        "question": "是痒、痛还是肿，范围是在扩大吗",
        "keywords": ("痒", "瘙痒", "肿", "红肿", "风团", "扩大"),
    },
    "allergen_exposure": {
        "question": "最近有没有接触新食物、药物、化妆品或环境刺激",
        "keywords": ("新食物", "新药", "化妆品", "护肤品", "过敏原", "接触"),
    },
    "allergy_history": {
        "question": "以前有没有类似发作，或明确的过敏史",
        "keywords": ("过敏史", "以前也这样", "类似", "青霉素过敏", "药物过敏"),
    },
    "main_symptom": {
        "question": "现在最难受的症状是什么，主要影响到哪里",
        "keywords": ("最难受", "主要", "最明显", "哪里不舒服"),
    },
    "severity": {
        "question": "大概有多重，对吃饭睡觉和活动影响大吗",
        "keywords": ("严重", "厉害", "影响睡觉", "影响活动", "疼醒"),
    },
    "red_flags": {
        "question": "有没有发热、出血、呼吸困难、昏厥或意识变化",
        "keywords": ("发热", "出血", "呼吸困难", "昏厥", "意识变化", "抽搐"),
    },
    "past_history": {
        "question": "既往有没有基础病、长期用药或过敏史",
        "keywords": ("高血压", "糖尿病", "长期用药", "基础病", "过敏史", "手术史"),
    },
    "age_weight_mental_status": {
        "question": "孩子现在多大、体重多少，精神状态和进食怎么样",
        "keywords": ("几岁", "月龄", "体重", "精神状态", "吃奶", "进食"),
    },
    "pregnancy_related_history": {
        "question": "还要确认末次月经、是否可能怀孕，以及有没有阴道流血",
        "keywords": ("末次月经", "怀孕", "妊娠", "阴道流血", "停经"),
    },
}


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _strip_role_prefix(text: str) -> tuple[str, str]:
    cleaned = _clean_text(text)
    for prefix, role in ROLE_PREFIXES.items():
        if cleaned.startswith(prefix):
            return role, _clean_text(cleaned[len(prefix) :])
    return "", cleaned


def _parse_dialogue(data: Any) -> list[dict[str, str]] | None:
    if not isinstance(data, list) or len(data) < 4:
        return None

    messages: list[dict[str, str]] = []
    expected = "user"
    for idx, item in enumerate(data):
        text = _clean_text(item)
        if not text:
            continue
        role, content = _strip_role_prefix(text)
        if not role:
            role = expected if idx % 2 == 0 else ("assistant" if expected == "user" else "user")
        if not content:
            continue
        if messages and role == messages[-1]["role"]:
            continue
        messages.append({"role": role, "content": content})
        expected = "assistant" if role == "user" else "user"

    if len(messages) < 4 or messages[0]["role"] != "user":
        return None
    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if len(messages) < 4:
        return None
    return messages


def _history_to_prompt(case_type: str, history: list[dict[str, str]]) -> str:
    lines = [
        f"任务：你是 BianQue-Intake，当前病例类型倾向为 {case_type}。",
        "要求：不要长篇解释，不要急着下诊断，只输出下一轮最关键的追问；若已出现高风险信号，可额外给一句就医提示。",
        "",
        "对话历史：",
    ]
    for message in history:
        role = "用户" if message["role"] == "user" else "医生"
        lines.append(f"{role}：{message['content']}")
    return "\n".join(lines)


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        text = _clean_text(item).strip("；。?？")
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _latest_user_text(history: list[dict[str, str]]) -> str:
    for message in reversed(history):
        if message["role"] == "user":
            return message["content"]
    return ""


def _slot_answered(slot: str, history: list[dict[str, str]]) -> bool:
    user_text = " ".join(message["content"] for message in history if message["role"] == "user")
    config = SLOT_CONFIG.get(slot)
    if not config:
        return False
    return any(keyword in user_text for keyword in config["keywords"])


def _special_questions(case_type: str, history: list[dict[str, str]]) -> list[str]:
    user_text = " ".join(message["content"] for message in history if message["role"] == "user")
    questions: list[str] = []

    if any(keyword in user_text for keyword in HIGH_RISK_HINTS):
        questions.append("现在意识清楚吗，症状有没有还在持续")
    if any(keyword in user_text for keyword in ("药", "吃了", "用药", "输液")):
        questions.append("最近具体用了什么药，用了多久，用药后是缓解还是加重")
    if any(keyword in user_text for keyword in ("高血压", "糖尿病", "冠心病", "肿瘤", "手术", "住院")):
        questions.append("既往基础病现在控制得怎么样，平时长期在吃什么药")
    if any(keyword in user_text for keyword in PEDIATRIC_HINTS):
        questions.append("孩子现在多大，精神状态和进食怎么样")
    female = any(keyword in user_text for keyword in FEMALE_HINTS)
    male = any(keyword in user_text for keyword in MALE_HINTS)
    if (
        female
        and not male
        and any(keyword in user_text for keyword in ("腹痛", "下腹", "流血", "月经", "白带", "停经", "怀孕"))
    ):
        questions.append("末次月经是什么时候，是否可能怀孕，有没有阴道流血")
    if case_type == "chest_pain":
        questions.append("疼痛会不会放射到左臂、肩背，活动后会不会更明显")
    return questions


def _pending_questions(case_type: str, labels: dict[str, Any], history: list[dict[str, str]]) -> list[str]:
    latest_user = _latest_user_text(history)
    questions = _special_questions(case_type, history)

    # Use case-level default followups as fallback, but rewrite them into shorter style.
    rule = _infer_case(latest_user or " ".join(message["content"] for message in history))
    for question in list(rule.get("questions", [])):
        short = _clean_text(question).strip("；。?？")
        if short:
            questions.append(short)

    for slot in labels.get("required_slots", []):
        if _slot_answered(slot, history):
            continue
        config = SLOT_CONFIG.get(slot)
        if config:
            questions.append(str(config["question"]))
    return _dedupe_keep_order(questions)


def _select_questions(case_type: str, labels: dict[str, Any], history: list[dict[str, str]]) -> list[str]:
    questions = _pending_questions(case_type, labels, history)
    previous_doctor = " ".join(message["content"] for message in history if message["role"] == "assistant")
    filtered: list[str] = []
    for question in questions:
        if question and question not in previous_doctor:
            filtered.append(question)
    if filtered:
        return filtered
    return questions


def _turn_style(triage: str, raw_output: str, is_last_assistant: bool) -> str:
    if triage == "emergency":
        return "multiturn_emergency"
    if triage == "urgent" or any(keyword in raw_output for keyword in ("急诊", "尽快就医", "立刻就医")):
        return "multiturn_urgent"
    if is_last_assistant:
        return "multiturn_assessment"
    return "multiturn_followup"


def _distill_turn(
    case_type: str,
    history: list[dict[str, str]],
    raw_output: str,
    labels: dict[str, Any],
    is_last_assistant: bool,
) -> tuple[str, str]:
    triage = str(labels.get("triage", "outpatient"))
    questions = _select_questions(case_type, labels, history)
    max_questions = 2 if triage == "emergency" else 3
    selected = questions[:max_questions]

    if selected:
        prefix = "先确认" if len(history) <= 2 else "还想确认"
        body = "；".join(f"{question}？" for question in selected)
        response = f"{prefix}：{body}"
    else:
        response = "还想确认病程变化、严重程度，以及既往基础病和用药情况。"

    risk_line = ""
    if triage == "emergency":
        risk_line = "这种情况风险偏高，建议尽快急诊。"
    elif triage == "urgent" and is_last_assistant:
        risk_line = "这种情况建议尽快线下就诊。"
    elif is_last_assistant and any(keyword in raw_output for keyword in ("急诊", "尽快就医", "立刻就医")):
        risk_line = "这种情况建议尽快线下就诊。"

    if risk_line:
        response = f"{response} {risk_line}".strip()

    return _turn_style(triage, raw_output, is_last_assistant), response


def _infer_labels(messages: list[dict[str, str]]) -> dict[str, Any]:
    all_user_text = " ".join(message["content"] for message in messages if message["role"] == "user")
    all_text = " ".join(message["content"] for message in messages)
    red_flags = [flag for flag in HIGH_RISK_HINTS if flag in all_text]
    pediatric = any(hint in all_user_text for hint in PEDIATRIC_HINTS)
    female = any(hint in all_user_text for hint in FEMALE_HINTS)
    male = any(hint in all_user_text for hint in MALE_HINTS)

    if red_flags:
        triage = "emergency"
    elif any(keyword in all_text for keyword in ("大量出血", "呼吸困难", "明显胸痛", "黑便", "便血")):
        triage = "urgent"
    else:
        triage = "outpatient"

    return {
        "red_flags": red_flags,
        "triage": triage,
        "pediatric": pediatric,
        "female_case": female and not male,
    }


def _required_slots(case_type: str, first_user_turn: str) -> list[str]:
    base_slots = {
        "respiratory": ["onset", "temperature", "cough_sputum", "dyspnea", "exposure_history"],
        "abdominal": ["pain_location", "pain_pattern", "vomiting_diarrhea", "stool_bleeding", "diet_history"],
        "chest_pain": ["onset", "duration", "exertion_relation", "radiation", "sweating_syncope"],
        "neuro": ["onset", "consciousness", "seizure", "speech_gait", "head_trauma_history"],
        "urinary_gyne": ["duration", "dysuria", "fever_backpain", "discharge_bleeding", "pregnancy_menses"],
        "skin_allergy": ["onset", "itching_swelling", "allergen_exposure", "allergy_history"],
        "generic": ["onset", "main_symptom", "severity", "red_flags", "past_history"],
    }
    slots = list(base_slots.get(case_type, base_slots["generic"]))
    if any(hint in first_user_turn for hint in PEDIATRIC_HINTS) and "age_weight_mental_status" not in slots:
        slots.append("age_weight_mental_status")
    if any(hint in first_user_turn for hint in FEMALE_HINTS) and any(
        keyword in first_user_turn for keyword in ("下腹", "腹痛", "阴道", "流血", "月经", "白带", "怀孕", "停经")
    ):
        slots.append("pregnancy_related_history")
    return slots


def _iter_huatuo_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BianQue multiturn data from Huatuo")
    parser.add_argument("--huatuo-file", default=HUATUO_DEFAULT)
    parser.add_argument(
        "--conversation-out",
        default="/root/autodl-tmp/medagent/datasets/curated/bianque_multiturn/conversations.jsonl",
    )
    parser.add_argument(
        "--train-out",
        default="/root/autodl-tmp/medagent/datasets/curated/bianque_multiturn/train_turns.jsonl",
    )
    parser.add_argument(
        "--valid-out",
        default="/root/autodl-tmp/medagent/datasets/curated/bianque_multiturn/valid_turns.jsonl",
    )
    parser.add_argument(
        "--summary-out",
        default="/root/autodl-tmp/medagent/datasets/curated/bianque_multiturn/summary.json",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-conversations", type=int, default=50000)
    args = parser.parse_args()

    conversations: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []
    seen_conversations: set[str] = set()
    case_counter: Counter[str] = Counter()
    turn_counter: Counter[str] = Counter()
    triage_counter: Counter[str] = Counter()
    style_counter: Counter[str] = Counter()

    for idx, record in enumerate(_iter_huatuo_jsonl(Path(args.huatuo_file))):
        if len(conversations) >= args.max_conversations:
            break
        messages = _parse_dialogue(record.get("data"))
        if not messages:
            continue

        first_user = messages[0]["content"]
        if not _looks_like_intake_question(first_user):
            continue

        case_rule = _infer_case(first_user)
        case_type = str(case_rule.get("name", "generic"))
        dialogue_hash = hashlib.md5(
            "\n".join(f"{message['role']}:{message['content']}" for message in messages).encode("utf-8")
        ).hexdigest()
        if dialogue_hash in seen_conversations:
            continue
        seen_conversations.add(dialogue_hash)

        labels = _infer_labels(messages)
        labels["required_slots"] = _required_slots(case_type, first_user)

        dialogue_id = f"{case_type}_{idx:06d}"
        assistant_turn_count = 0
        distilled_turns: list[dict[str, Any]] = []
        assistant_indices = [turn_idx for turn_idx, message in enumerate(messages) if message["role"] == "assistant"]
        if not assistant_indices:
            continue
        last_assistant_idx = assistant_indices[-1]

        for turn_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            assistant_turn_count += 1
            history = messages[:turn_idx]
            history_prompt = _history_to_prompt(case_type, history)
            style, distilled_output = _distill_turn(
                case_type=case_type,
                history=history,
                raw_output=message["content"],
                labels=labels,
                is_last_assistant=(turn_idx == last_assistant_idx),
            )

            row = {
                "input": history_prompt,
                "output": distilled_output,
                "raw_output": message["content"],
                "task": "bianque_multiturn",
                "style": style,
                "case_type": case_type,
                "turn_index": assistant_turn_count,
                "dialogue_id": dialogue_id,
                "triage": labels["triage"],
                "source": "HuatuoGPT-sft-data-v1",
                "source_file": Path(args.huatuo_file).name,
            }
            turn_rows.append(row)
            distilled_turns.append(
                {
                    "turn_index": assistant_turn_count,
                    "style": style,
                    "history": history,
                    "raw_output": message["content"],
                    "distilled_output": distilled_output,
                }
            )
            turn_counter[case_type] += 1
            style_counter[style] += 1

        conversations.append(
            {
                "case_id": dialogue_id,
                "case_type": case_type,
                "source": "HuatuoGPT-sft-data-v1",
                "source_file": Path(args.huatuo_file).name,
                "dialogue": messages,
                "distilled_turns": distilled_turns,
                "labels": labels,
            }
        )
        case_counter[case_type] += 1
        triage_counter[labels["triage"]] += 1

    rng = random.Random(args.seed)
    rng.shuffle(turn_rows)
    valid_size = max(1, int(len(turn_rows) * args.valid_ratio)) if turn_rows else 0
    valid_rows = turn_rows[:valid_size]
    train_rows = turn_rows[valid_size:]

    output_paths = [
        Path(args.conversation_out),
        Path(args.train_out),
        Path(args.valid_out),
        Path(args.summary_out),
    ]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    with Path(args.conversation_out).open("w", encoding="utf-8") as handle:
        for row in conversations:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with Path(args.train_out).open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with Path(args.valid_out).open("w", encoding="utf-8") as handle:
        for row in valid_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "total_conversations": len(conversations),
        "train_turns": len(train_rows),
        "valid_turns": len(valid_rows),
        "total_turns": len(turn_rows),
        "case_counter": dict(case_counter),
        "turn_counter": dict(turn_counter),
        "triage_counter": dict(triage_counter),
        "style_counter": dict(style_counter),
    }
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[ok] conversations={len(conversations)} total_turns={len(turn_rows)} "
        f"train_turns={len(train_rows)} valid_turns={len(valid_rows)}"
    )


if __name__ == "__main__":
    main()
