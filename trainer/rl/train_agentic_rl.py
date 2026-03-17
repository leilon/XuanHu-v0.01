#!/usr/bin/env python
"""
Train a preference-optimized adapter for agent decisions (DPO as practical RLHF stage).

Input JSONL:
{"prompt":"...","chosen":"...","rejected":"..."}
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medagent.services.adapter_bank import AdapterBank, AdapterMeta


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train agentic preference adapter (DPO)")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--pairs-file", required=True)
    parser.add_argument("--task", default="agentic_policy")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-dir", default="")
    parser.add_argument("--adapter-bank-dir", default="adapters")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=2048)
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from trl import DPOTrainer
    except Exception as exc:
        raise RuntimeError(
            "Missing deps. Install with: "
            "pip install transformers datasets peft bitsandbytes accelerate trl"
        ) from exc

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["HF_DATASETS_CACHE"] = str(Path(args.cache_dir) / "datasets")
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_dir:
            os.environ["WANDB_DIR"] = args.wandb_dir

    rows = _load_jsonl(args.pairs_file)
    ds = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    peft_cfg = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=False,
        fp16=True,
        report_to=["wandb"] if args.wandb_project else [],
        run_name=args.wandb_run_name or None,
    )

    trainer = DPOTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        processing_class=tokenizer,
        max_length=args.max_len,
        max_prompt_length=min(1024, args.max_len // 2),
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    bank = AdapterBank(root_dir=args.adapter_bank_dir)
    bank.register(
        AdapterMeta(
            task=args.task,
            adapter_path=str(output_dir),
            base_model=args.base_model,
            step=trainer.state.global_step,
            dataset=args.pairs_file,
        )
    )
    print(f"[ok] agentic adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
