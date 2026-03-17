#!/usr/bin/env python
"""Generate a markdown audit report for downloaded HF medical corpora."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_rag_corpus import _default_hf_manifest_path, _iter_json_records, _load_hf_manifest_map, _normalize_hf_row


def _clean_preview(value: Any, limit: int = 180) -> str:
    text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _discover_candidate_files(repo_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(repo_dir.rglob("*")):
        if path.name == "source_meta.json":
            continue
        if path.suffix not in {".json", ".jsonl"}:
            continue
        files.append(path)
    return files


def _collect_repo_samples(
    repo_dir: Path,
    sample_target: int,
    seed: int,
    manifest_map: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta_file = repo_dir / "source_meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    else:
        meta = manifest_map.get(repo_dir.name)
    if not meta:
        return {}, []
    files = _discover_candidate_files(repo_dir)
    summary = {
        "repo_id": meta.get("repo_id", repo_dir.name),
        "local_name": repo_dir.name,
        "bucket": meta.get("bucket", ""),
        "source_type": meta.get("source_type", "cn_medical_corpus"),
        "file_count": len(files),
        "size_mb": round(sum(path.stat().st_size for path in files) / (1024 * 1024), 2),
    }

    if not files or sample_target <= 0:
        return summary, []

    rng = random.Random(seed)
    ordered_files = files[:]
    rng.shuffle(ordered_files)

    samples: list[dict[str, Any]] = []
    per_file_cap = max(1, math.ceil(sample_target / max(1, min(len(ordered_files), sample_target))))

    for file_path in ordered_files:
        taken = 0
        for idx, row in enumerate(_iter_json_records(file_path), start=1):
            normalized = _normalize_hf_row(row, meta)
            if not normalized:
                continue
            samples.append(
                {
                    "repo_id": meta.get("repo_id", repo_dir.name),
                    "local_name": repo_dir.name,
                    "source_type": normalized["source_type"],
                    "title": normalized["title"],
                    "preview": _clean_preview(normalized["text"], 220),
                    "file": str(file_path.relative_to(repo_dir)).replace('\\', '/'),
                    "row_index": idx,
                    "raw_question": _clean_preview(
                        row.get("question")
                        or row.get("Question")
                        or row.get("query")
                        or row.get("instruction")
                        or row.get("title")
                        or row.get("input")
                        or row.get("prompt")
                        or row.get("questions"),
                        120,
                    ),
                    "raw_answer": _clean_preview(
                        row.get("answer")
                        or row.get("Answer")
                        or row.get("response")
                        or row.get("Response")
                        or row.get("output")
                        or row.get("summary")
                        or row.get("answers"),
                        140,
                    ),
                }
            )
            taken += 1
            if taken >= per_file_cap or len(samples) >= sample_target:
                break
        if len(samples) >= sample_target:
            break

    summary["sampled_items"] = len(samples)
    return summary, samples


def _render_report(root: Path, summaries: list[dict[str, Any]], samples: list[dict[str, Any]], sample_target: int) -> str:
    lines: list[str] = []
    lines.append("# HF 中文医疗语料抽样审核报告")
    lines.append("")
    lines.append(f"- 数据根目录：`{root}`")
    lines.append(f"- 审核目标：不少于 `{sample_target}` 条样本")
    lines.append(f"- 实际样本数：`{len(samples)}`")
    lines.append("")
    lines.append("## 数据集概览")
    lines.append("")
    lines.append("| 数据集 | 本地目录 | bucket | source_type | 文件数 | 体积(MB) | 抽样数 |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: |")
    for item in summaries:
        lines.append(
            f"| `{item['repo_id']}` | `{item['local_name']}` | `{item.get('bucket', '')}` | `{item.get('source_type', '')}` | {item.get('file_count', 0)} | {item.get('size_mb', 0)} | {item.get('sampled_items', 0)} |"
        )
    lines.append("")
    lines.append("## 抽样样本")
    lines.append("")
    for idx, item in enumerate(samples, start=1):
        lines.append(f"### 样本 {idx}")
        lines.append(f"- 数据集：`{item['repo_id']}`")
        lines.append(f"- 文件：`{item['local_name']}/{item['file']}`")
        lines.append(f"- 行号：`{item['row_index']}`")
        lines.append(f"- 标题：{item['title']}")
        if item["raw_question"]:
            lines.append(f"- 原始问题：{item['raw_question']}")
        if item["raw_answer"]:
            lines.append(f"- 原始回答：{item['raw_answer']}")
        lines.append(f"- 规范化预览：{item['preview']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HF corpus audit report")
    parser.add_argument("--root", required=True, help="HF corpora root directory")
    parser.add_argument("--out", required=True, help="Output markdown file")
    parser.add_argument("--sample-count", type=int, default=120)
    parser.add_argument("--hf-manifest", default=str(_default_hf_manifest_path()))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    manifest_map = _load_hf_manifest_map(Path(args.hf_manifest))
    repo_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    repo_dirs = [path for path in repo_dirs if (path / "source_meta.json").exists() or path.name in manifest_map]
    if not repo_dirs:
        raise SystemExit(f"No corpus repos found under {root}")

    per_repo = max(1, math.ceil(args.sample_count / len(repo_dirs)))
    summaries: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []

    for offset, repo_dir in enumerate(repo_dirs):
        summary, samples = _collect_repo_samples(
            repo_dir,
            per_repo,
            seed=args.seed + offset,
            manifest_map=manifest_map,
        )
        summaries.append(summary)
        all_samples.extend(samples)

    if len(all_samples) < args.sample_count:
        # Second pass: top up from all repos if some files were sparse.
        for offset, repo_dir in enumerate(repo_dirs):
            needed = args.sample_count - len(all_samples)
            if needed <= 0:
                break
            _, extra = _collect_repo_samples(
                repo_dir,
                needed,
                seed=args.seed + 100 + offset,
                manifest_map=manifest_map,
            )
            seen = {(item["repo_id"], item["file"], item["row_index"]) for item in all_samples}
            for item in extra:
                key = (item["repo_id"], item["file"], item["row_index"])
                if key in seen:
                    continue
                all_samples.append(item)
                seen.add(key)
                if len(all_samples) >= args.sample_count:
                    break

    all_samples = all_samples[: args.sample_count]
    summary_map = defaultdict(int)
    for item in all_samples:
        summary_map[item["repo_id"]] += 1
    for summary in summaries:
        summary["sampled_items"] = summary_map.get(summary["repo_id"], 0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_report(root, summaries, all_samples, args.sample_count), encoding="utf-8")
    print(f"[ok] wrote audit report to {out_path} with {len(all_samples)} samples")


if __name__ == "__main__":
    main()
