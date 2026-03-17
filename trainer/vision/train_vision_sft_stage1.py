#!/usr/bin/env python
"""
Stage-1 vision SFT for QingNang-ClinicOS.

Recommended launch on 4x5090:
accelerate launch --config_file configs/accelerate_4x5090.yaml trainer/vision/train_vision_sft_stage1.py \
  --base-model /root/autodl-tmp/medagent/models/huatuogpt-vision-7b-qwen2.5vl \
  --train-file /root/autodl-tmp/medagent/datasets/sft/vision_report_train.jsonl \
  --eval-file /root/autodl-tmp/medagent/datasets/sft/vision_report_valid.jsonl \
  --output-dir /root/autodl-tmp/medagent/outputs/adapters/canggong_vision_stage1
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


def _set_env(cache_dir: str, args: argparse.Namespace) -> None:
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = str(Path(cache_dir) / "datasets")
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_dir:
            os.environ["WANDB_DIR"] = args.wandb_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 vision SFT for QingNang-ClinicOS")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default="canggong_vision_stage1")
    parser.add_argument("--dataset-name", default="vision_stage1")
    parser.add_argument("--adapter-bank-dir", default="adapters")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=3072)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-dir", default="")
    args = parser.parse_args()

    try:
        import torch
        from PIL import Image
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from torch.utils.data import Dataset
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VisionModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as VisionModelClass
    except Exception as exc:
        raise RuntimeError(
            "Missing vision training deps. Install with: "
            "pip install transformers peft bitsandbytes accelerate pillow wandb"
        ) from exc

    _set_env(args.cache_dir, args)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    train_rows = _load_jsonl(args.train_file)
    eval_rows = _load_jsonl(args.eval_file) if args.eval_file else []

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = VisionModelClass.from_pretrained(
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

    class VisionSFTDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = self.rows[index]
            image_path = Path(str(item["image"]))
            if not image_path.is_absolute():
                image_path = Path(args.train_file).resolve().parent / image_path
            return {
                "image_path": str(image_path),
                "prompt": str(item["prompt"]).strip(),
                "response": str(item["response"]).strip(),
            }

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = []
        full_texts = []
        prompt_texts = []
        for item in batch:
            image = Image.open(item["image_path"]).convert("RGB")
            images.append(image)
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": item["prompt"]},
                    ],
                }
            ]
            full_messages = user_messages + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["response"]}],
                }
            ]
            prompt_texts.append(
                processor.apply_chat_template(
                    user_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            full_texts.append(
                processor.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

        full_batch = processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=args.max_len,
            return_tensors="pt",
        )
        prompt_batch = processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=args.max_len,
            return_tensors="pt",
        )

        labels = full_batch["input_ids"].clone()
        pad_token_id = processor.tokenizer.pad_token_id
        for idx in range(labels.size(0)):
            prompt_len = int(prompt_batch["attention_mask"][idx].sum().item())
            labels[idx, :prompt_len] = -100
        labels[labels == pad_token_id] = -100
        full_batch["labels"] = labels
        return full_batch

    train_ds = VisionSFTDataset(train_rows)
    eval_ds = VisionSFTDataset(eval_rows) if eval_rows else None

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
        data_collator=collate_fn,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

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
    print(f"[ok] stage-1 vision adapter saved to {output_dir}")


if __name__ == "__main__":
    main()

