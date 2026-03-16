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
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
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
    parser.add_argument("--train-out", default="/root/autodl-tmp/medagent/datasets/sft/train_stage1_text.jsonl")
    parser.add_argument("--valid-out", default="/root/autodl-tmp/medagent/datasets/sft/valid_stage1_text.jsonl")
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
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
            fingerprint = hashlib.md5(f"{user_text}\n{assistant_text}".encode("utf-8")).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            rows.append(
                {
                    "input": user_text,
                    "output": assistant_text,
                    "bucket": _bucket_for(file_path),
                    "source": file_path.name,
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
