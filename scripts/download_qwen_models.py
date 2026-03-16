#!/usr/bin/env python
"""
Download QingNang-ClinicOS base models to a local directory.

Example:
  export HF_ENDPOINT=https://hf-mirror.com
  python scripts/download_qwen_models.py --root /root/autodl-tmp/medagent/models --with-huatuo-vision
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_SPECS = [
    ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct"),
    ("FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL", "huatuogpt-vision-7b-qwen2.5vl"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download QingNang-ClinicOS base models")
    parser.add_argument("--root", default="/root/autodl-tmp/medagent/models")
    parser.add_argument("--with-qwen-vl", action="store_true", help="Also download raw Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--with-huatuo-vision", action="store_true", help="Download HuatuoGPT-Vision-7B-Qwen2.5VL")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    specs = [("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct")]
    if args.with_huatuo_vision or not args.with_qwen_vl:
        specs.append(("FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL", "huatuogpt-vision-7b-qwen2.5vl"))
    if args.with_qwen_vl:
        specs.append(("Qwen/Qwen2.5-VL-7B-Instruct", "qwen2.5-vl-7b-instruct"))

    for repo_id, dirname in specs:
        target = root / dirname
        print(f"[download] {repo_id} -> {target}", flush=True)
        snapshot_download(repo_id, local_dir=str(target))
        print(f"[done] {repo_id}", flush=True)


if __name__ == "__main__":
    main()
