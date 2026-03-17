#!/usr/bin/env python
"""
Stage-1 text SFT for QingNang-ClinicOS.

Recommended launch on 4x5090:
accelerate launch --config_file configs/accelerate_4x5090.yaml trainer/text/train_text_sft_stage1.py \
  --base-model /root/autodl-tmp/medagent/models/qwen2.5-7b-instruct \
  --train-file /root/autodl-tmp/medagent/datasets/curated/text_stage1/train.jsonl \
  --eval-file /root/autodl-tmp/medagent/datasets/curated/text_stage1/valid.jsonl \
  --output-dir /root/autodl-tmp/medagent/outputs/adapters/bianque_text_stage1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medagent.services.adapter_bank import AdapterBank, AdapterMeta


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _set_hf_env(cache_dir: str) -> None:
    if not cache_dir:
        return
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = str(Path(cache_dir) / "datasets")


def _set_wandb_env(args: argparse.Namespace) -> None:
    if not args.wandb_project:
        return
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_dir:
        os.environ["WANDB_DIR"] = args.wandb_dir


def _format_chat(tokenizer: Any, user_text: str, assistant_text: str) -> tuple[str, str]:
    prompt_messages = [{"role": "user", "content": user_text}]
    full_messages = prompt_messages + [{"role": "assistant", "content": assistant_text}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt_text, full_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 text SFT for QingNang-ClinicOS")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default="bianque_text_stage1")
    parser.add_argument("--dataset-name", default="text_stage1")
    parser.add_argument("--adapter-bank-dir", default="adapters")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-dir", default="")
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise RuntimeError(
            "Missing training deps. Install with: "
            "pip install transformers datasets peft bitsandbytes accelerate wandb"
        ) from exc

    _set_hf_env(args.cache_dir)
    _set_wandb_env(args)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    train_rows = _load_jsonl(args.train_file)
    eval_rows = _load_jsonl(args.eval_file) if args.eval_file else []

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map={"": local_rank},
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    def tokenize_record(record: dict[str, Any]) -> dict[str, Any]:
        prompt_text, full_text = _format_chat(
            tokenizer,
            str(record.get("input", "")).strip(),
            str(record.get("output", "")).strip(),
        )
        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=args.max_len,
        )
        prompt_tokens = tokenizer(
            prompt_text,
            truncation=True,
            max_length=args.max_len,
        )
        labels = list(full_tokens["input_ids"])
        prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))
        for idx in range(prompt_len):
            labels[idx] = -100
        return {
            "input_ids": full_tokens["input_ids"],
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels,
        }

    train_ds = Dataset.from_list(train_rows).map(tokenize_record, remove_columns=list(train_rows[0].keys()))
    eval_ds = None
    if eval_rows:
        eval_ds = Dataset.from_list(eval_rows).map(tokenize_record, remove_columns=list(eval_rows[0].keys()))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_ds is not None else None,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        save_strategy="steps",
        save_total_limit=2,
        bf16=True,
        fp16=False,
        report_to=["wandb"] if args.wandb_project else [],
        run_name=args.wandb_run_name or None,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    bank = AdapterBank(root_dir=args.adapter_bank_dir)
    bank.register(
        AdapterMeta(
            task=args.task,
            adapter_path=str(output_dir),
            base_model=args.base_model,
            step=trainer.state.global_step,
            dataset=args.dataset_name,
        )
    )
    manifest = {
        "base_model": args.base_model,
        "train_file": args.train_file,
        "eval_file": args.eval_file,
        "task": args.task,
        "dataset_name": args.dataset_name,
        "global_step": trainer.state.global_step,
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ok] stage-1 text adapter saved to {output_dir}")


if __name__ == "__main__":
    main()

