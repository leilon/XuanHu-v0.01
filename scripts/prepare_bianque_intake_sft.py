#!/usr/bin/env python
"""Prepare BianQue-Intake SFT data from mixed medical corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SOURCE_LIMITS = {
    "HuatuoGPT-sft-data-v1": 120_000,
    "Chinese-medical-dialogue-data": 80_000,
    "cMedQA-V2.0": 60_000,
    "Medical-R1-Distill-Data-Chinese": 40_000,
    "shibing624__medical": 80_000,
}

CASE_RULES = [
    {
        "name": "chest_pain",
        "keywords": ("胸痛", "胸闷", "胸口堵", "气短", "呼吸困难", "心慌", "心悸"),
        "questions": [
            "这种不适从什么时候开始，持续多久了",
            "是活动后更明显，还是静息时也会发作",
            "有没有胸痛、放射到肩背或左臂、出冷汗、头晕或濒死感",
            "有没有高血压、冠心病、哮喘或慢阻肺这类基础病",
        ],
        "urgent": "如果现在胸痛明显、呼吸困难、出冷汗或快要晕倒，请立即急诊。",
    },
    {
        "name": "respiratory",
        "keywords": ("发热", "发烧", "咳嗽", "咳痰", "咽痛", "鼻塞", "流涕", "喘"),
        "questions": [
            "症状是从什么时候开始的，最高体温大概多少",
            "咳嗽是干咳还是有痰，痰是什么颜色",
            "有没有胸闷、气短、呼吸费力，或者接触过同样发热咳嗽的人",
            "最近有没有熬夜、旅行、聚集接触史，或者本身有哮喘等慢病",
        ],
        "urgent": "如果持续高热不退、呼吸困难或精神状态明显变差，请尽快线下就医。",
    },
    {
        "name": "abdominal",
        "keywords": ("腹痛", "肚子痛", "恶心", "呕吐", "腹泻", "拉肚子", "黑便", "便血"),
        "questions": [
            "腹痛具体在上腹、脐周还是右下腹，是持续痛还是阵发性加重",
            "有没有发热、呕吐、腹泻、黑便、便血或者腹胀",
            "吃东西后是加重还是缓解，最近饮食有没有不洁或明显改变",
            "以前有没有胃病、胆囊、胰腺或阑尾方面的问题",
        ],
        "urgent": "如果腹痛剧烈、反复呕吐、黑便便血或肚子越来越硬，请尽快急诊评估。",
    },
    {
        "name": "neuro",
        "keywords": ("头痛", "头晕", "抽搐", "意识不清", "说话不清", "一侧无力", "麻木"),
        "questions": [
            "这些症状是突然出现还是逐渐加重，持续了多久",
            "有没有发热、呕吐、视物模糊、说话不清或者走路不稳",
            "有没有一侧肢体无力、口角歪斜、抽搐或意识不清",
            "既往有没有高血压、脑卒中、癫痫或近期头部外伤",
        ],
        "urgent": "如果是突发剧烈头痛、抽搐、意识改变或一侧肢体无力，请立即急诊。",
    },
    {
        "name": "urinary_gyne",
        "keywords": ("尿频", "尿急", "尿痛", "血尿", "下腹痛", "阴道流血", "白带", "月经"),
        "questions": [
            "症状持续多久了，排尿时是刺痛还是灼痛，下腹痛在什么位置",
            "有没有发热、腰痛、分泌物异常、月经推迟或阴道流血",
            "最近有没有性生活、憋尿、喝水少或反复泌尿感染史",
            "既往有没有妇科疾病、结石、肾病或类似发作",
        ],
        "urgent": "如果有明显发热伴腰痛、大量出血或疼痛越来越重，请尽快线下就诊。",
    },
    {
        "name": "skin_allergy",
        "keywords": ("皮疹", "过敏", "瘙痒", "红疹", "风团", "荨麻疹"),
        "questions": [
            "皮疹是什么时候开始的，主要长在哪些部位，有没有越来越多",
            "有没有瘙痒、发热、嘴唇眼睑肿、喘不过气或喉咙紧",
            "最近有没有接触新食物、新药、染发剂、化妆品或宠物",
            "以前有没有过敏史、哮喘史或类似发作",
        ],
        "urgent": "如果伴有呼吸困难、喉头紧缩或面唇肿胀，请立即急诊。",
    },
]

GENERAL_EXCLUDE = (
    "是什么",
    "什么意思",
    "原理",
    "机制",
    "定义",
    "概述",
    "科普",
    "正常值",
    "参考范围",
    "教我",
)

FEMALE_HINTS = ("女性", "女", "宝妈", "月经", "例假", "经期", "怀孕", "妊娠")
MALE_HINTS = ("男性", "男", "老公", "父亲", "爸爸")
PEDIATRIC_HINTS = ("孩子", "宝宝", "婴儿", "幼儿", "儿子", "女儿")
CHRONIC_HINTS = ("高血压", "糖尿病", "冠心病", "哮喘", "慢阻肺", "肾病", "肝病", "肿瘤")
MEDICATION_HINTS = ("吃药", "用药", "药", "布洛芬", "阿莫西林", "二甲双胍", "阿司匹林")


def _source_name(path: Path) -> str:
    text = str(path)
    for key in SOURCE_LIMITS:
        if key in text:
            return key
    return path.parent.name or path.stem


def _limit_for(path: Path) -> int:
    text = str(path)
    for key, value in SOURCE_LIMITS.items():
        if key in text:
            return value
    return 40_000


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, list):
        return _clean_text(" ".join(_stringify(item) for item in value if _stringify(item)))
    if isinstance(value, dict):
        return _clean_text(" ".join(_stringify(v) for v in value.values() if _stringify(v)))
    return _clean_text(str(value))


def _extract_pair(record: dict[str, Any]) -> tuple[str, str] | None:
    instruction = _clean_text(record.get("instruction", ""))
    in_text = _clean_text(record.get("input", ""))
    output = _clean_text(record.get("output", ""))
    if instruction and output:
        question = instruction if not in_text else f"{instruction} {in_text}"
        return question, output
    if in_text and output:
        return in_text, output

    question = _stringify(
        record.get("question")
        or record.get("query")
        or record.get("prompt")
        or record.get("title")
        or record.get("questions")
    )
    answer = _stringify(
        record.get("answer")
        or record.get("response")
        or record.get("summary")
        or record.get("answers")
    )
    if question and answer:
        return question, answer

    data = record.get("data")
    if isinstance(data, list) and len(data) >= 2:
        question = _clean_text(str(data[0]).replace("问：", ""))
        answer = _clean_text(str(data[1]).replace("答：", ""))
        if question and answer:
            return question, answer

    conversations = record.get("conversations")
    if isinstance(conversations, list):
        user_text = ""
        assistant_text = ""
        for message in conversations:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", message.get("from", ""))).lower()
            content = _clean_text(message.get("content", message.get("value", "")))
            if not user_text and role in {"user", "human"} and content:
                user_text = content
            elif user_text and role in {"assistant", "gpt"} and content:
                assistant_text = content
                break
        if user_text and assistant_text:
            return user_text, assistant_text
    return None


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row
        return

    if path.suffix != ".json":
        return

    line_mode_hit = False
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                line_mode_hit = True
                yield row
    if line_mode_hit:
        return

    if path.stat().st_size > 128 * 1024 * 1024:
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row
    elif isinstance(payload, dict):
        for key in ("data", "rows", "items", "train"):
            value = payload.get(key)
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        yield row
                return
        yield payload


def _looks_like_intake_question(question: str) -> bool:
    text = question.strip()
    if len(text) < 6 or len(text) > 800:
        return False
    symptom_hit = any(keyword in text for rule in CASE_RULES for keyword in rule["keywords"])
    if any(keyword in text for keyword in GENERAL_EXCLUDE) and not symptom_hit:
        return False
    person_hit = any(marker in text for marker in ("我", "孩子", "宝宝", "父亲", "母亲", "老人", "家里人"))
    return symptom_hit or person_hit


def _infer_case(question: str) -> dict[str, Any]:
    for rule in CASE_RULES:
        if any(keyword in question for keyword in rule["keywords"]):
            return rule
    return {
        "name": "generic",
        "questions": [
            "症状是从什么时候开始的，现在是持续存在还是一阵一阵发作",
            "最难受的部位和最明显的症状分别是什么，严重程度大概怎样",
            "有没有发热、明显疼痛、出血、呼吸困难、呕吐或意识变化这类危险信号",
            "既往有没有基础病、长期用药、过敏史，最近生活作息或饮食有没有明显变化",
        ],
        "urgent": "如果症状在快速加重，或者已经出现呼吸困难、出血、昏厥等危险表现，请尽快线下就医。",
    }


def _append_contextual_questions(question: str, questions: list[str]) -> list[str]:
    enriched = list(questions)
    if any(keyword in question for keyword in CHRONIC_HINTS):
        enriched.append("请补充一下既往基础病控制得怎么样，最近有没有复查，平时长期在吃什么药")
    elif any(keyword in question for keyword in MEDICATION_HINTS):
        enriched.append("请补充一下最近具体用了哪些药，吃了多久，用药后症状是缓解还是加重")

    female = any(keyword in question for keyword in FEMALE_HINTS)
    male = any(keyword in question for keyword in MALE_HINTS)
    pediatric = any(keyword in question for keyword in PEDIATRIC_HINTS)
    if pediatric:
        enriched.append("如果是孩子，请补充年龄、体重、精神状态、吃奶或进食情况，以及有没有高热或嗜睡")
    if female and not male and any(keyword in question for keyword in ("下腹", "腹痛", "阴道", "流血", "月经", "白带")):
        enriched.append("如果是育龄女性，还需要确认末次月经、是否可能怀孕，以及有没有阴道流血")
    return enriched


def _build_followup(question: str) -> tuple[str, str]:
    rule = _infer_case(question)
    questions = _append_contextual_questions(question, list(rule["questions"]))
    core = "；".join(questions[:5])
    response = f"先补充几项关键信息：{core}。{rule['urgent']}"
    return rule["name"], response


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BianQue-Intake SFT data")
    parser.add_argument("--sft-root", default="/root/autodl-tmp/medagent/datasets/sft")
    parser.add_argument("--train-out", default="/root/autodl-tmp/medagent/datasets/sft/bianque_intake_train.jsonl")
    parser.add_argument("--valid-out", default="/root/autodl-tmp/medagent/datasets/sft/bianque_intake_valid.jsonl")
    parser.add_argument("--summary-out", default="/root/autodl-tmp/medagent/datasets/sft/bianque_intake_summary.json")
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=180000)
    args = parser.parse_args()

    sft_root = Path(args.sft_root)
    output_paths = {Path(args.train_out).resolve(), Path(args.valid_out).resolve()}
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    source_counter: Counter[str] = Counter()
    case_counter: Counter[str] = Counter()

    files = sorted(list(sft_root.rglob("*.json")) + list(sft_root.rglob("*.jsonl")))
    rng = random.Random(args.seed)

    for file_path in files:
        if file_path.resolve() in output_paths:
            continue
        source_name = _source_name(file_path)
        source_cap = _limit_for(file_path)
        for record in _iter_records(file_path):
            if len(rows) >= args.max_samples or source_counter[source_name] >= source_cap:
                break
            pair = _extract_pair(record)
            if not pair:
                continue
            question, answer = pair
            if not _looks_like_intake_question(question):
                continue
            case_type, output = _build_followup(question)
            fingerprint = hashlib.md5(f"{question}\n{output}".encode("utf-8")).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            rows.append(
                {
                    "input": question,
                    "output": output,
                    "task": "bianque_intake",
                    "style": "first_turn_followup",
                    "case_type": case_type,
                    "source": source_name,
                    "source_file": file_path.name,
                    "reference_answer": answer[:500],
                }
            )
            source_counter[source_name] += 1
            case_counter[case_type] += 1
        if len(rows) >= args.max_samples:
            break

    rng.shuffle(rows)
    valid_size = max(1, int(len(rows) * args.valid_ratio)) if rows else 0
    valid_rows = rows[:valid_size]
    train_rows = rows[valid_size:]

    train_out = Path(args.train_out)
    valid_out = Path(args.valid_out)
    summary_out = Path(args.summary_out)
    for path in (train_out, valid_out, summary_out):
        path.parent.mkdir(parents=True, exist_ok=True)

    with train_out.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with valid_out.open("w", encoding="utf-8") as handle:
        for row in valid_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "train": len(train_rows),
        "valid": len(valid_rows),
        "total": len(rows),
        "source_counter": dict(source_counter),
        "case_counter": dict(case_counter),
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] train={len(train_rows)} valid={len(valid_rows)} total={len(rows)}")


if __name__ == "__main__":
    main()
