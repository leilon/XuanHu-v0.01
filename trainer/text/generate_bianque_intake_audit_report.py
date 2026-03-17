#!/usr/bin/env python
"""Generate a Markdown audit report for BianQue-Intake curated data."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sample_by_case_type(
    rows: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("case_type", "unknown"))].append(row)

    counts = {case_type: len(items) for case_type, items in grouped.items()}
    rng = random.Random(seed)
    sampled: dict[str, list[dict[str, Any]]] = {}
    for case_type, items in sorted(grouped.items()):
        if len(items) <= sample_size:
            sampled[case_type] = list(items)
        else:
            sampled[case_type] = rng.sample(items, sample_size)
    return sampled, counts


def _fmt(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _build_report(
    dataset_path: Path,
    sampled: dict[str, list[dict[str, Any]]],
    counts: dict[str, int],
    sample_size: int,
) -> str:
    total = sum(counts.values())
    lines: list[str] = []
    lines.append("# BianQue-Intake 数据审核报告")
    lines.append("")
    lines.append(f"- 数据文件：`{dataset_path}`")
    lines.append(f"- 总条数：`{total}`")
    lines.append(f"- 抽样规则：每个 `case_type` 最多抽 `100` 条")
    lines.append("")
    lines.append("## 类型分布")
    lines.append("")
    for case_type, count in sorted(counts.items()):
        lines.append(f"- `{case_type}`: `{count}`")

    for case_type, rows in sampled.items():
        lines.append("")
        lines.append(f"## {case_type}")
        lines.append("")
        lines.append(f"- 原始条数：`{counts[case_type]}`")
        lines.append(f"- 抽样条数：`{min(sample_size, counts[case_type])}`")
        lines.append("")
        for idx, row in enumerate(rows, start=1):
            lines.append(f"### {case_type}-{idx:03d}")
            lines.append("")
            lines.append(f"- source: `{row.get('source', '')}`")
            lines.append(f"- source_file: `{row.get('source_file', '')}`")
            lines.append(f"- style: `{row.get('style', '')}`")
            lines.append("")
            lines.append("**问题**")
            lines.append("")
            lines.append(_fmt(row.get("input", "")))
            lines.append("")
            lines.append("**输出**")
            lines.append("")
            lines.append(_fmt(row.get("output", "")))
            lines.append("")
            reference_answer = _fmt(row.get("reference_answer", ""))
            if reference_answer:
                lines.append("**参考答案截断**")
                lines.append("")
                lines.append(reference_answer)
                lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BianQue-Intake audit report")
    parser.add_argument(
        "--dataset-file",
        default="/root/autodl-tmp/medagent/datasets/curated/bianque_intake/train.jsonl",
    )
    parser.add_argument(
        "--out-file",
        default="/root/autodl-tmp/medagent/outputs/reports/bianque_intake_audit_report_zh.md",
    )
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_file)
    rows = _load_jsonl(dataset_path)
    sampled, counts = _sample_by_case_type(rows, args.sample_size, args.seed)
    report = _build_report(dataset_path, sampled, counts, args.sample_size)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[ok] wrote audit report to {out_path}")


if __name__ == "__main__":
    main()
