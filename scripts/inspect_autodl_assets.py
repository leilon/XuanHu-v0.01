#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    total = 0
    if path.is_file():
        total = path.stat().st_size
    else:
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(total)
    unit = units[0]
    for cand in units[1:]:
        if size < 1024:
            break
        size /= 1024
        unit = cand
    return f"{size:.1f}{unit}"


def _model_status(model_dir: Path) -> dict[str, object]:
    status: dict[str, object] = {
        "exists": model_dir.exists(),
        "size": _fmt_size(model_dir),
        "files": [],
        "complete": False,
        "missing": [],
    }
    if not model_dir.exists():
        return status

    safetensors = sorted(p.name for p in model_dir.glob("*.safetensors"))
    status["files"] = safetensors
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        payload = json.loads(index_file.read_text(encoding="utf-8"))
        needed = sorted(set(payload.get("weight_map", {}).values()))
        missing = [name for name in needed if name not in safetensors]
        status["missing"] = missing
        status["complete"] = not missing
    else:
        status["complete"] = bool(safetensors)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect AutoDL model and dataset assets")
    parser.add_argument("--root", default="/root/autodl-tmp/medagent")
    args = parser.parse_args()

    root = Path(args.root)
    models_root = root / "models"
    datasets_root = root / "datasets"

    print("== Models ==")
    for model_dir in sorted(models_root.glob("*")):
        if not model_dir.is_dir():
            continue
        status = _model_status(model_dir)
        print(f"{model_dir.name}: size={status['size']} complete={status['complete']}")
        if status["missing"]:
            print(f"  missing: {', '.join(status['missing'])}")

    print("\n== Datasets ==")
    for bucket in sorted(datasets_root.glob("*")):
        if bucket.is_dir():
            print(f"{bucket.name}: {_fmt_size(bucket)}")

    print("\n== Key Files ==")
    for key_file in [
        datasets_root / "sft" / "train_sft_v1.jsonl",
        datasets_root / "rl" / "agentic_pairs_v2.jsonl",
        datasets_root / "rl" / "agentic_pairs_medical_v1.jsonl",
    ]:
        print(f"{key_file}: {_fmt_size(key_file)}")


if __name__ == "__main__":
    main()
