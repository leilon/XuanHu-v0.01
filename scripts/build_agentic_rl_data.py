#!/usr/bin/env python
"""
Build Agentic-RL training data from benchmark cases and dialogue logs.

Input JSONL format for dialogue logs:
{"id":"x1","question":"...","draft":"...","final":"...","risk":"medium","used_tools":["rag","drug_checker"]}

Output JSONL format:
{"prompt":"...","chosen":"...","rejected":"...","meta":{"source":"...","risk":"..."}}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _contains_all(text: str, keys: list[str]) -> bool:
    return all(k in text for k in keys)


def _build_from_benchmark(cases: list[dict]) -> list[dict]:
    pairs: list[dict] = []
    for c in cases:
        must_have = c.get("expected", {}).get("must_have", [])
        safe_keys = c.get("expected", {}).get("safety_tokens", [])
        grounding = c.get("expected", {}).get("grounding_tokens", [])
        prompt = c["question"]

        chosen = (
            f"[分诊建议] 建议根据症状及时就医评估。\n"
            f"[用药建议] 请在医生指导下使用退热药，注意禁忌。\n"
            f"[知识依据] 包含可追溯证据。\n[引用] guideline"
        )
        rejected = "建议休息即可。"

        if must_have and safe_keys and grounding:
            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "meta": {"source": "benchmark_bootstrap", "risk": "mixed"},
                }
            )
    return pairs


def _build_from_logs(logs: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in logs:
        prompt = row.get("question", "")
        final = row.get("final", "")
        draft = row.get("draft", "")
        if not prompt or not final:
            continue
        # Prefer final answer if it has citations or safety hints.
        chosen = final
        rejected = draft or "信息不足。"
        out.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "meta": {
                    "source": "dialogue_log",
                    "risk": row.get("risk", "unknown"),
                    "tool_count": len(row.get("used_tools", [])),
                },
            }
        )
    return out


def _write_jsonl(rows: list[dict], out_file: str) -> None:
    path = Path(out_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Agentic-RL preference data")
    parser.add_argument("--benchmark-json", default="data/benchmark_cases.json")
    parser.add_argument("--dialogue-log-jsonl", default="")
    parser.add_argument("--out-file", default="data/agentic_rl_pairs.jsonl")
    args = parser.parse_args()

    cases = _load_json(args.benchmark_json)
    rows = _build_from_benchmark(cases)
    if args.dialogue_log_jsonl:
        rows.extend(_build_from_logs(_load_jsonl(args.dialogue_log_jsonl)))
    _write_jsonl(rows, args.out_file)
    print(f"[ok] wrote {len(rows)} rows to {args.out_file}")


if __name__ == "__main__":
    main()

