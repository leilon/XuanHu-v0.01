#!/usr/bin/env python
"""
Validate vision SFT JSONL schema before training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = ("image", "prompt", "response")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate vision SFT data")
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.file)
    base_dir = dataset_path.resolve().parent
    total = 0
    errors: list[str] = []

    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            for field in REQUIRED_FIELDS:
                if field not in row or not str(row[field]).strip():
                    errors.append(f"line {line_no}: missing field {field}")
            image_path = Path(str(row.get("image", "")))
            if not image_path.is_absolute():
                image_path = base_dir / image_path
            if not image_path.exists():
                errors.append(f"line {line_no}: image not found -> {image_path}")

    if errors:
        print("[error] vision dataset validation failed")
        for item in errors[:20]:
            print(item)
        raise SystemExit(1)
    print(f"[ok] validated {total} vision records")


if __name__ == "__main__":
    main()
