#!/usr/bin/env python
"""
Download Qwen 2.5 text and vision models to a local directory.

Example:
  export HF_ENDPOINT=https://hf-mirror.com
  python scripts/download_qwen_models.py --root /root/autodl-tmp/medagent/models
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_SPECS = [
    ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct"),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "qwen2.5-vl-7b-instruct"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen 2.5 models")
    parser.add_argument("--root", default="/root/autodl-tmp/medagent/models")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    for repo_id, dirname in MODEL_SPECS:
        target = root / dirname
        print(f"[download] {repo_id} -> {target}", flush=True)
        snapshot_download(repo_id, local_dir=str(target))
        print(f"[done] {repo_id}", flush=True)


if __name__ == "__main__":
    main()
