#!/usr/bin/env python
"""Prepare BianQue multiturn SFT data from existing Huatuo dialogues."""

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
    _append_contextual_questions,
    _build_followup,
    _infer_case,
    _looks_like_intake_question,
)

HUATUO_DEFAULT = (
    "/root/autodl-tmp/medagent/datasets/sft/"
    "FreedomIntelligence__HuatuoGPT-sft-data-v1/HuatuoGPT_sft_data_v1.jsonl"
)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_turn_text(text: str) -> str:
    text = _clean_text(text)
    for prefix in ("问：", "答：", "患者:", "医生:"):
        if text.startswith(prefix):
            return _clean_text(text[len(prefix) :])
    return text


def _parse_dialogue(data: Any) -> list[dict[str, str]] | None:
    if not isinstance(data, list) or len(data) < 4:
        return None

    messages: list[dict[str, str]] = []
    expected_role = "user"
    for idx, item in enumerate(data):
        text = _clean_text(item)
        if not text:
            continue

        role = expected_role
        if text.startswith("问："):
            role = "user"
        elif text.startswith("答："):
            role = "assistant"
        elif idx % 2 == 0:
            role = "user"
        else:
            role = "assistant"

        normalized = _normalize_turn_text(text)
        if not normalized:
            continue

        if messages and role == messages[-1]["role"]:
            continue
        messages.append({"role": role, "content": normalized})
        expected_role = "assistant" if role == "user" else "user"

    if len(messages) < 4:
        return None
    if messages[0]["role"] != "user":
        return None

    while messages and messages[-1]["role"] != "assistant":
        messages.pop()
    if len(messages) < 4:
        return None
    return messages


def _is_doctor_followup(text: str) -> bool:
    markers = ("？", "?", "请问", "还需要", "有没有", "是否", "多久", "什么颜色", "什么位置", "几岁", "体温")
    return any(marker in text for marker in markers)


def _history_to_prompt(case_type: str, history: list[dict[str, str]]) -> str:
    lines = [
        f"任务：你是 BianQue-Intake，当前病例类型倾向为 {case_type}。",
        "请基于下面的既往对话，继续给出下一轮医生追问或风险提示。",
        "",
        "对话历史：",
    ]
    for message in history:
        role = "用户" if message["role"] == "user" else "医生"
        lines.append(f"{role}：{message['content']}")
    return "\n".join(lines)


def _infer_labels(messages: list[dict[str, str]]) -> dict[str, Any]:
    all_user_text = " ".join(message["content"] for message in messages if message["role"] == "user")
    all_text = " ".join(message["content"] for message in messages)
    red_flags = [flag for flag in HIGH_RISK_HINTS if flag in all_text]
    pediatric = any(hint in all_user_text for hint in PEDIATRIC_HINTS)
    female = any(hint in all_user_text for hint in FEMALE_HINTS)
    male = any(hint in all_user_text for hint in MALE_HINTS)

    if red_flags:
        triage = "emergency"
    elif any(keyword in all_text for keyword in ("大量出血", "呼吸困难", "胸痛明显", "黑便", "便血")):
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
        "skin_allergy": ["onset", "distribution", "itching_swelling", "allergen_exposure", "allergy_history"],
        "generic": ["onset", "main_symptom", "severity", "red_flags", "past_history"],
    }
    slots = list(base_slots.get(case_type, base_slots["generic"]))
    if any(hint in first_user_turn for hint in PEDIATRIC_HINTS) and "age_weight_mental_status" not in slots:
        slots.append("age_weight_mental_status")
    if any(hint in first_user_turn for hint in FEMALE_HINTS) and any(
        keyword in first_user_turn for keyword in ("下腹", "流血", "月经", "怀孕", "胎停")
    ):
        slots.append("pregnancy_related_history")
    return slots


def _read_huatuo_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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

    records = _read_huatuo_jsonl(Path(args.huatuo_file))
    conversations: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []
    seen_conversations: set[str] = set()
    case_counter: Counter[str] = Counter()
    turn_counter: Counter[str] = Counter()
    triage_counter: Counter[str] = Counter()

    for idx, record in enumerate(records):
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
            "\n".join(f"{m['role']}:{m['content']}" for m in messages).encode("utf-8")
        ).hexdigest()
        if dialogue_hash in seen_conversations:
            continue
        seen_conversations.add(dialogue_hash)

        labels = _infer_labels(messages)
        labels["required_slots"] = _required_slots(case_type, first_user)
        if messages and messages[-1]["role"] == "assistant" and labels["triage"] == "outpatient":
            if any(keyword in messages[-1]["content"] for keyword in ("急诊", "尽快就医", "立即就医", "立即急诊")):
                labels["triage"] = "emergency"

        dialogue_id = f"{case_type}_{idx:06d}"
        assistant_turn_count = 0
        for turn_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            assistant_turn_count += 1
            history = messages[:turn_idx]
            history_prompt = _history_to_prompt(case_type, history)
            turn_kind = "followup" if turn_idx < len(messages) - 1 and _is_doctor_followup(message["content"]) else "assessment"
            turn_rows.append(
                {
                    "input": history_prompt,
                    "output": message["content"],
                    "task": "bianque_multiturn",
                    "style": "multiturn_followup" if turn_kind == "followup" else "multiturn_assessment",
                    "case_type": case_type,
                    "turn_index": assistant_turn_count,
                    "dialogue_id": dialogue_id,
                    "triage": labels["triage"],
                    "source": "HuatuoGPT-sft-data-v1",
                    "source_file": Path(args.huatuo_file).name,
                }
            )
            turn_counter[case_type] += 1

        conversations.append(
            {
                "case_id": dialogue_id,
                "case_type": case_type,
                "source": "HuatuoGPT-sft-data-v1",
                "source_file": Path(args.huatuo_file).name,
                "dialogue": messages,
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
    }
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[ok] conversations={len(conversations)} total_turns={len(turn_rows)} "
        f"train_turns={len(train_rows)} valid_turns={len(valid_rows)}"
    )


if __name__ == "__main__":
    main()
