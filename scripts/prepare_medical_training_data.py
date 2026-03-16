#!/usr/bin/env python
"""
Prepare unified SFT and Agentic-RL JSONL files from downloaded datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import zipfile

from datasets import load_dataset
import random


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_dialogue_pairs(obj: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        q = obj.get("question") or obj.get("query") or obj.get("instruction") or obj.get("input")
        a = obj.get("answer") or obj.get("response") or obj.get("output")
        if isinstance(q, str) and isinstance(a, str):
            pairs.append((q.strip(), a.strip()))
        for v in obj.values():
            pairs.extend(_iter_dialogue_pairs(v))
    elif isinstance(obj, list):
        for item in obj:
            pairs.extend(_iter_dialogue_pairs(item))
    return pairs


def _extract_pair_from_record(record: dict[str, Any]) -> tuple[str, str] | None:
    # Alpaca style
    instruction = str(record.get("instruction", "")).strip()
    in_text = str(record.get("input", "")).strip()
    output = str(record.get("output", "")).strip()
    if instruction and output:
        q = instruction if not in_text else f"{instruction}\n{in_text}"
        return q.strip(), output

    # QA style
    question = str(record.get("question", record.get("prompt", record.get("query", "")))).strip()
    answer = str(record.get("answer", record.get("response", record.get("output", "")))).strip()
    if question and answer:
        return question, answer

    # Huatuo style
    data = record.get("data")
    if isinstance(data, list) and len(data) >= 2:
        q = str(data[0]).strip().replace("问：", "").strip()
        a = str(data[1]).strip().replace("答：", "").strip()
        if q and a:
            return q, a

    # Conversation style
    conv = record.get("conversations")
    if isinstance(conv, list):
        user_msg = ""
        assistant_msg = ""
        for m in conv:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", m.get("from", ""))).lower()
            content = str(m.get("content", m.get("value", ""))).strip()
            if not user_msg and role in {"user", "human"} and content:
                user_msg = content
            if user_msg and role in {"assistant", "gpt"} and content:
                assistant_msg = content
                break
        if user_msg and assistant_msg:
            return user_msg, assistant_msg
    return None


def _parse_json_or_jsonl(path: Path) -> list[dict]:
    # Fast path: line-wise JSONL (also handles very large pseudo-jsonl files).
    records: list[dict] = []
    line_success = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if line[0] != "{":
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
                    line_success += 1
            except Exception:
                continue
            if idx > 500 and line_success == 0:
                break

    if line_success > 0:
        return records

    # Fallback: small JSON file load.
    if path.stat().st_size <= 300 * 1024 * 1024:
        try:
            obj = _read_json(path)
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
            if isinstance(obj, dict):
                return [obj]
        except Exception:
            return []
    return []


def build_sft(sft_root: Path, out_file: Path) -> int:
    rows: list[dict[str, str]] = []
    for json_file in list(sft_root.rglob("*.json")) + list(sft_root.rglob("*.jsonl")):
        try:
            recs = _parse_json_or_jsonl(json_file)
            for rec in recs:
                pair = _extract_pair_from_record(rec)
                if pair:
                    q, a = pair
                    if q and a and len(q) > 2 and len(a) > 2:
                        rows.append({"input": q, "output": a})
        except Exception:
            continue

    for zf in sft_root.rglob("*.zip"):
        try:
            with zipfile.ZipFile(zf, "r") as z:
                json_names = [n for n in z.namelist() if n.endswith(".json")]
                for name in json_names:
                    try:
                        with z.open(name) as fp:
                            data = json.loads(fp.read().decode("utf-8", errors="ignore"))
                        for q, a in _iter_dialogue_pairs(data):
                            if q and a and len(q) > 2 and len(a) > 2:
                                rows.append({"input": q, "output": a})
                    except Exception:
                        continue
        except Exception:
            continue

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        random.shuffle(rows)
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def build_rl(rl_root: Path, out_file: Path) -> int:
    rows: list[dict[str, Any]] = []
    for pq_file in rl_root.rglob("*.parquet"):
        try:
            ds = load_dataset("parquet", data_files=str(pq_file), split="train")
        except Exception:
            continue
        for r in ds:
            prompt = ""
            chosen = ""
            rejected = ""

            if "prompt" in r and "chosen" in r and "rejected" in r:
                prompt = str(r.get("prompt", "")).strip()
                chosen_obj = r.get("chosen")
                rejected_obj = r.get("rejected")
                if isinstance(chosen_obj, list):
                    chosen = next(
                        (
                            str(m.get("content", "")).strip()
                            for m in reversed(chosen_obj)
                            if isinstance(m, dict) and m.get("role") == "assistant"
                        ),
                        "",
                    )
                    if not prompt:
                        prompt = next(
                            (
                                str(m.get("content", "")).strip()
                                for m in chosen_obj
                                if isinstance(m, dict) and m.get("role") == "user"
                            ),
                            "",
                        )
                else:
                    chosen = str(chosen_obj).strip()

                if isinstance(rejected_obj, list):
                    rejected = next(
                        (
                            str(m.get("content", "")).strip()
                            for m in reversed(rejected_obj)
                            if isinstance(m, dict) and m.get("role") == "assistant"
                        ),
                        "",
                    )
                else:
                    rejected = str(rejected_obj).strip()

            elif "chosen" in r and "rejected" in r:
                chosen_obj = r.get("chosen")
                rejected_obj = r.get("rejected")
                if isinstance(chosen_obj, list):
                    prompt = next(
                        (
                            str(m.get("content", "")).strip()
                            for m in chosen_obj
                            if isinstance(m, dict) and m.get("role") == "user"
                        ),
                        "",
                    )
                    chosen = next(
                        (
                            str(m.get("content", "")).strip()
                            for m in reversed(chosen_obj)
                            if isinstance(m, dict) and m.get("role") == "assistant"
                        ),
                        "",
                    )
                if isinstance(rejected_obj, list):
                    rejected = next(
                        (
                            str(m.get("content", "")).strip()
                            for m in reversed(rejected_obj)
                            if isinstance(m, dict) and m.get("role") == "assistant"
                        ),
                        "",
                    )

            if prompt and chosen and rejected:
                rows.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                        "meta": {"source_file": str(pq_file)},
                    }
                )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare medical SFT and RL datasets")
    parser.add_argument("--root", default="/root/autodl-tmp/medagent/datasets")
    parser.add_argument("--sft-out", default="/root/autodl-tmp/medagent/datasets/sft/train_sft_v1.jsonl")
    parser.add_argument("--rl-out", default="/root/autodl-tmp/medagent/datasets/rl/agentic_pairs_v2.jsonl")
    args = parser.parse_args()

    root = Path(args.root)
    sft_count = build_sft(root / "sft", Path(args.sft_out))
    rl_count = build_rl(root / "rl", Path(args.rl_out))
    print(f"[ok] SFT rows: {sft_count}")
    print(f"[ok] RL rows: {rl_count}")


if __name__ == "__main__":
    main()
