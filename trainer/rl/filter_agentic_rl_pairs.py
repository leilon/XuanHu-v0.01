#!/usr/bin/env python
"""
Filter preference pairs to medical-relevant and quality-controlled subset.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MED_RE = re.compile(
    r"(medical|patient|disease|diagnos|symptom|drug|therapy|treatment|clinical|hospital|cancer|covid|"
    r"医生|患者|疾病|诊断|症状|用药|治疗|临床|医院|癌|化验|报告|分诊|药物)",
    re.IGNORECASE,
)

GIBBERISH_RE = re.compile(r"[^\w\s\u4e00-\u9fff,.;:?!%()/\-]{20,}")


def is_medical(text: str) -> bool:
    return bool(MED_RE.search(text))


def quality_ok(text: str) -> bool:
    if len(text.strip()) < 40:
        return False
    if GIBBERISH_RE.search(text):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter agentic RL pairs")
    parser.add_argument("--in-file", required=True)
    parser.add_argument("--out-file", required=True)
    args = parser.parse_args()

    in_path = Path(args.in_file)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            row = json.loads(line)
            prompt = row.get("prompt", "")
            chosen = row.get("chosen", "")
            rejected = row.get("rejected", "")

            if not is_medical(prompt):
                continue
            if not quality_ok(chosen) or not quality_ok(rejected):
                continue
            if chosen.strip() == rejected.strip():
                continue

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[ok] kept {kept}/{total} -> {out_path}")


if __name__ == "__main__":
    main()
