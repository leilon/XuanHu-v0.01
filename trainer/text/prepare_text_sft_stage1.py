#!/usr/bin/env python
"""
Prepare stage-1 text SFT train/valid files from downloaded medical datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SOURCE_BUCKETS = {
    "HuatuoGPT-sft-data-v1": "medical_qa",
    "Chinese-medical-dialogue-data": "medical_qa",
    "cMedQA-V2.0": "medical_qa",
    "Medical-R1-Distill-Data-Chinese": "medical_reasoning",
    "shibing624__medical": "medical_qa",
}

SOURCE_LIMITS = {
    "HuatuoGPT-sft-data-v1": 180_000,
    "Chinese-medical-dialogue-data": 100_000,
    "cMedQA-V2.0": 80_000,
    "Medical-R1-Distill-Data-Chinese": 80_000,
    "shibing624__medical": 120_000,
}

FOLLOWUP_RULES = [
    (
        ("胸痛", "胸闷", "胸口堵", "堵得慌", "喘", "气短", "呼吸困难"),
        "先补充几项关键信息：这种不适从什么时候开始？活动后会不会更明显？有没有胸痛、明显气短、出冷汗或头晕？",
    ),
    (
        ("发热", "发烧", "咳嗽", "咳痰", "咽痛", "鼻塞", "流涕"),
        "先告诉我症状从什么时候开始，体温大概最高到多少？咳嗽是干咳还是有痰？有没有胸闷、气促或接触过发热患者？",
    ),
    (
        ("腹痛", "肚子痛", "呕吐", "恶心", "腹泻"),
        "先告诉我腹痛具体在什么位置，是一直痛还是一阵一阵痛？有没有发热、腹泻、黑便或便血？",
    ),
    (
        ("头痛", "头晕", "意识不清", "抽搐", "无力"),
        "先补充一下：头痛或头晕是突然出现还是慢慢加重？有没有发热、呕吐、说话不清或一侧肢体无力？",
    ),
    (
        ("尿频", "尿急", "尿痛", "血尿", "下腹痛", "阴道流血"),
        "先说一下这些症状持续多久了？有没有发热、腰痛、分泌物异常或排尿疼痛？如果是育龄女性，还需要确认月经和妊娠可能。",
    ),
    (
        ("皮疹", "过敏", "瘙痒", "红疹"),
        "先补充几点：皮疹是什么时候出现的，分布在哪些部位？有没有瘙痒、呼吸不适，或者接触过新食物、新药物？",
    ),
]


def _quality_ok(question: str, answer: str) -> bool:
    q = question.strip()
    a = answer.strip()
    if len(q) < 5 or len(a) < 5:
        return False
    if len(q) > 1200 or len(a) > 4000:
        return False
    if q == a:
        return False
    return True


def _looks_like_short_followup(answer: str, max_chars: int) -> bool:
    text = answer.strip()
    if len(text) > max_chars:
        return False
    if "1." in text or "1：" in text or "1、" in text:
        return False
    return any(marker in text for marker in ("？", "请补充", "请告诉我", "先补充", "有没有", "从什么时候"))


def _generate_short_followup(question: str) -> str:
    cleaned = re.sub(r"\s+", " ", question).strip()
    for keywords, template in FOLLOWUP_RULES:
        if any(keyword in cleaned for keyword in keywords):
            return template
    return "先补充几点关键信息：这个不适从什么时候开始？是在加重还是缓解？有没有发热、疼痛、呼吸困难或出血这类危险信号？"


def _extract_pair_from_record(record: dict[str, Any]) -> tuple[str, str] | None:
    instruction = str(record.get("instruction", "")).strip()
    in_text = str(record.get("input", "")).strip()
    output = str(record.get("output", "")).strip()
    if instruction and output:
        question = instruction if not in_text else f"{instruction}\n{in_text}"
        return question.strip(), output

    question = str(record.get("question", record.get("prompt", record.get("query", "")))).strip()
    answer = str(record.get("answer", record.get("response", record.get("output", "")))).strip()
    if question and answer:
        return question, answer

    data = record.get("data")
    if isinstance(data, list) and len(data) >= 2:
        question = str(data[0]).strip().replace("问：", "").strip()
        answer = str(data[1]).strip().replace("答：", "").strip()
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
            content = str(message.get("content", message.get("value", ""))).strip()
            if not user_text and role in {"user", "human"} and content:
                user_text = content
            if user_text and role in {"assistant", "gpt"} and content:
                assistant_text = content
                break
        if user_text and assistant_text:
            return user_text, assistant_text
    return None


def _iter_json_or_jsonl(path: Path):
    line_success = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line or line[0] != "{":
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                line_success += 1
                yield obj
            if idx > 500 and line_success == 0:
                break
    if line_success > 0:
        return

    if path.stat().st_size > 300 * 1024 * 1024:
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        yield payload


def _bucket_for(path: Path) -> str:
    text = str(path)
    for key, bucket in SOURCE_BUCKETS.items():
        if key in text:
            return bucket
    return "medical_qa"


def _limit_for(path: Path) -> int:
    text = str(path)
    for key, limit in SOURCE_LIMITS.items():
        if key in text:
            return limit
    return 50_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare stage-1 text SFT data")
    parser.add_argument("--sft-root", default="/root/autodl-tmp/medagent/datasets/sft")
    parser.add_argument("--train-out", default="/root/autodl-tmp/medagent/datasets/curated/text_stage1/train.jsonl")
    parser.add_argument("--valid-out", default="/root/autodl-tmp/medagent/datasets/curated/text_stage1/valid.jsonl")
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-output-chars", type=int, default=120)
    parser.add_argument(
        "--preserve-original-output",
        action="store_true",
        help="Keep original answers even when they are long explanatory paragraphs.",
    )
    args = parser.parse_args()

    sft_root = Path(args.sft_root)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    source_counts: dict[str, int] = {}

    for file_path in list(sft_root.rglob("*.json")) + list(sft_root.rglob("*.jsonl")):
        source_key = str(file_path)
        limit = _limit_for(file_path)
        for record in _iter_json_or_jsonl(file_path):
            if source_counts.get(source_key, 0) >= limit:
                break
            pair = _extract_pair_from_record(record)
            if not pair:
                continue
            user_text, assistant_text = pair
            if not _quality_ok(user_text, assistant_text):
                continue
            output_text = assistant_text
            style = "source_answer"
            if not args.preserve_original_output:
                if not _looks_like_short_followup(assistant_text, args.max_output_chars):
                    output_text = _generate_short_followup(user_text)
                    style = "doctor_short_turn"
            fingerprint = hashlib.md5(f"{user_text}\n{assistant_text}".encode("utf-8")).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            rows.append(
                {
                    "input": user_text,
                    "output": output_text,
                    "bucket": _bucket_for(file_path),
                    "source": file_path.name,
                    "style": style,
                }
            )
            source_counts[source_key] = source_counts.get(source_key, 0) + 1

    random.seed(args.seed)
    random.shuffle(rows)
    valid_size = max(1, int(len(rows) * args.valid_ratio))
    valid_rows = rows[:valid_size]
    train_rows = rows[valid_size:]

    train_out = Path(args.train_out)
    valid_out = Path(args.valid_out)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    valid_out.parent.mkdir(parents=True, exist_ok=True)

    with open(train_out, "w", encoding="utf-8") as train_handle:
        for row in train_rows:
            train_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(valid_out, "w", encoding="utf-8") as valid_handle:
        for row in valid_rows:
            valid_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[ok] train={len(train_rows)} valid={len(valid_rows)}")


if __name__ == "__main__":
    main()
