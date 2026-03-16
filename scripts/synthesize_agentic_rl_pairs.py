#!/usr/bin/env python
"""
Synthesize medical agentic RL preference pairs from SFT prompts.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from medagent.orchestrator import Orchestrator


def _read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _make_rejected(prompt: str) -> str:
    # Intentionally weaker response as rejected sample.
    return (
        "先多喝水休息，观察即可。"
        "如果不舒服可以以后再看。"
        f"（问题：{prompt[:80]}）"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize agentic RL pairs from SFT data")
    parser.add_argument("--sft-file", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    orch = Orchestrator()
    rows = list(_read_jsonl(args.sft_file))
    random.shuffle(rows)
    rows = rows[: args.max_samples]

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in rows:
            prompt = str(item.get("input", "")).strip()
            if not prompt:
                continue
            chosen = orch.run(user_id=f"syn_{n}", user_text=prompt)
            rejected = _make_rejected(prompt)
            row = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "meta": {"source": "synthetic_agentic", "seed": args.seed},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"[ok] wrote {n} synthetic pairs to {out_path}")


if __name__ == "__main__":
    main()

