#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official HealthBench via OpenAI simple-evals checkout.")
    parser.add_argument(
        "--simple-evals-dir",
        default="runtime_assets/benchmarks/simple-evals",
        help="Path to the official openai/simple-evals checkout.",
    )
    parser.add_argument(
        "--eval",
        default="healthbench",
        choices=["healthbench", "healthbench_consensus", "healthbench_hard"],
        help="HealthBench split to run.",
    )
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--examples", type=int, default=30)
    parser.add_argument("--n-threads", type=int, default=16)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--grader-model", default="", help="Optional grader model override if supported by local simple-evals version.")
    args = parser.parse_args()

    repo_dir = Path(args.simple_evals_dir).resolve()
    runner = repo_dir / "simple_evals.py"
    if not repo_dir.exists():
        raise SystemExit(
            f"simple-evals 目录不存在：{repo_dir}\n"
            "请先把官方仓库放到该目录，或通过 --simple-evals-dir 指定路径。"
        )
    if not runner.exists():
        raise SystemExit(
            f"未找到官方 runner：{runner}\n"
            "请确认这是 openai/simple-evals 的仓库根目录。"
        )

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(
            f"环境变量 {args.api_key_env} 未设置。\n"
            "拿到 API 后，先 export/set 这个变量，再重新运行。"
        )

    command = [
        sys.executable,
        str(runner),
        f"--eval={args.eval}",
        f"--model={args.model}",
        "--examples",
        str(args.examples),
        "--n-threads",
        str(args.n_threads),
        "--n-repeats",
        str(args.n_repeats),
    ]
    if args.grader_model:
        command.extend(["--grader-model", args.grader_model])

    print("[run]", " ".join(command))
    subprocess.run(command, cwd=repo_dir, check=True)


if __name__ == "__main__":
    main()
