#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

DEFAULT_MODEL_DIR = "/root/autodl-tmp/medagent/models/huatuogpt-vision-7b-qwen2.5vl"
DEFAULT_BASE_PROCESSOR = "Qwen/Qwen2.5-VL-7B-Instruct"


def load_processor(model_dir: Path):
    local_processor_files = [
        model_dir / 'preprocessor_config.json',
        model_dir / 'processor_config.json',
        model_dir / 'tokenizer_config.json',
    ]
    if any(path.exists() for path in local_processor_files):
        return AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    return AutoProcessor.from_pretrained(DEFAULT_BASE_PROCESSOR, trust_remote_code=True)


def build_messages(prompt: str, image_path: str | None) -> list[dict]:
    content: list[dict] = []
    if image_path:
        image_uri = image_path if image_path.startswith(('http://', 'https://', 'file://')) else f'file://{image_path}'
        content.append({'type': 'image', 'image': image_uri})
    content.append({'type': 'text', 'text': prompt})
    return [{'role': 'user', 'content': content}]


def main() -> None:
    parser = argparse.ArgumentParser(description='Smoke test HuatuoGPT-Vision-7B-Qwen2.5VL')
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR)
    parser.add_argument('--prompt', default='你是一个医疗助手。请用一句话总结这段描述涉及的主要症状。')
    parser.add_argument('--image', default=None, help='Optional local image path')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f'model dir not found: {model_dir}')

    processor = load_processor(model_dir)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    model.eval()

    messages = build_messages(args.prompt, args.image)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text[0])


if __name__ == '__main__':
    main()
