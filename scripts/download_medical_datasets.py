#!/usr/bin/env python
"""
Download selected medical datasets to AutoDL data disk using HF mirror.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

SFT_DATASETS = [
    "BillGPT/Chinese-medical-dialogue-data",
    "wangrongsheng/cMedQA-V2.0",
    "medalpaca/medical_meadow_medqa",
    "FreedomIntelligence/Medical-R1-Distill-Data-Chinese",
]

RL_DATASETS = [
    "saepark/ultrafeedback-binarized-preferences-medical-cldfilter-train",
    "saepark/ultrafeedback-binarized-preferences-medical-cldfilter-validation-1k",
    "saepark/ultrafeedback-binarized-preferences-medical-cldfilter-test-1k",
    "saepark/explicitMedical-medical-preference-pubmed-olmo-normal-rollouts-graded-by-claude",
]

EVAL_DATASETS = [
    "FreedomIntelligence/CMB",
    "GBaker/MedQA-USMLE-4-options-hf",
]

RAG_DATASETS = [
    "prognosis/guidelines_medqa_qa_v1",
]


def _download(repo_id: str, local_root: Path) -> None:
    from huggingface_hub import snapshot_download

    target = local_root / repo_id.replace("/", "__")
    target.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id} -> {target}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download medical datasets to local dir")
    parser.add_argument("--root", default="/root/autodl-tmp/medagent/datasets")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    root = Path(args.root)
    (root / "sft").mkdir(parents=True, exist_ok=True)
    (root / "rl").mkdir(parents=True, exist_ok=True)
    (root / "eval").mkdir(parents=True, exist_ok=True)
    (root / "rag_raw").mkdir(parents=True, exist_ok=True)

    for ds in SFT_DATASETS:
        _download(ds, root / "sft")
    for ds in RL_DATASETS:
        _download(ds, root / "rl")
    for ds in EVAL_DATASETS:
        _download(ds, root / "eval")
    for ds in RAG_DATASETS:
        _download(ds, root / "rag_raw")

    print("[ok] all selected datasets downloaded")


if __name__ == "__main__":
    main()
