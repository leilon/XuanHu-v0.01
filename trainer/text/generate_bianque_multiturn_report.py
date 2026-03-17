#!/usr/bin/env python
"""Generate a Markdown audit report for cleaned BianQue multiturn data."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
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


def _sample_by_case_type(rows: list[dict[str, Any]], sample_size: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("case_type", "unknown"))].append(row)

    rng = random.Random(seed)
    sampled: dict[str, list[dict[str, Any]]] = {}
    for case_type, items in sorted(grouped.items()):
        if len(items) <= sample_size:
            sampled[case_type] = list(items)
        else:
            sampled[case_type] = rng.sample(items, sample_size)
    return sampled


def _fmt_dialogue(dialogue: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for turn in dialogue:
        role = "用户" if turn.get("role") == "user" else "医生"
        lines.append(f"{role}：{str(turn.get('content', '')).strip()}")
    return "\n".join(lines)


def _build_report(
    rows: list[dict[str, Any]],
    sampled: dict[str, list[dict[str, Any]]],
    sample_size: int,
    dataset_path: Path,
) -> str:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("case_type", "unknown"))] += 1

    lines: list[str] = []
    lines.append("# BianQue 多轮数据审核报告")
    lines.append("")
    lines.append(f"- 数据文件：`{dataset_path}`")
    lines.append(f"- 总对话数：`{len(rows)}`")
    lines.append(f"- 抽样规则：每个 `case_type` 抽 `10` 条")
    lines.append("")
    lines.append("## 类型分布")
    lines.append("")
    for case_type, count in sorted(counts.items()):
        lines.append(f"- `{case_type}`: `{count}`")

    for case_type, items in sampled.items():
        lines.append("")
        lines.append(f"## {case_type}")
        lines.append("")
        lines.append(f"- 原始条数：`{counts[case_type]}`")
        lines.append(f"- 抽样条数：`{min(sample_size, counts[case_type])}`")
        lines.append("")
        for idx, row in enumerate(items, start=1):
            labels = row.get("labels", {})
            lines.append(f"### {case_type}-{idx:03d}")
            lines.append("")
            lines.append(f"- case_id: `{row.get('case_id', '')}`")
            lines.append(f"- source: `{row.get('source', '')}`")
            lines.append(f"- triage: `{labels.get('triage', '')}`")
            lines.append(f"- red_flags: `{', '.join(labels.get('red_flags', []))}`")
            lines.append(f"- required_slots: `{', '.join(labels.get('required_slots', []))}`")
            lines.append("")
            lines.append("**完整对话**")
            lines.append("")
            lines.append(_fmt_dialogue(row.get("dialogue", [])))
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audit report for BianQue multiturn data")
    parser.add_argument(
        "--conversation-file",
        default="/root/autodl-tmp/medagent/datasets/curated/bianque_multiturn/conversations.jsonl",
    )
    parser.add_argument(
        "--out-file",
        default="/root/autodl-tmp/medagent/outputs/reports/bianque_multiturn_report_zh.md",
    )
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.conversation_file)
    rows = _load_jsonl(dataset_path)
    sampled = _sample_by_case_type(rows, args.sample_size, args.seed)
    report = _build_report(rows, sampled, args.sample_size, dataset_path)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[ok] wrote multiturn report to {out_path}")


if __name__ == "__main__":
    main()
