from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class HuatuoRuntime:
    """Lazy Huatuo-VL runtime wrapper for local single-model inference."""

    DEFAULT_REMOTE_MODEL = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"
    DEFAULT_LOCAL_MODEL = Path("/root/autodl-tmp/medagent/models/huatuogpt-vision-7b-qwen2.5vl")
    DEFAULT_BASE_PROCESSOR = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(
        self,
        model_ref: str | None = None,
        local_model_dir: str | Path | None = None,
        allow_remote_fallback: bool = False,
    ) -> None:
        self.model_ref = model_ref or self.DEFAULT_REMOTE_MODEL
        self.local_model_dir = Path(local_model_dir) if local_model_dir else self.DEFAULT_LOCAL_MODEL
        self.allow_remote_fallback = allow_remote_fallback
        self._model: Any | None = None
        self._processor: Any | None = None
        self._process_vision_info: Any | None = None
        self._device: Any | None = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        return self.local_model_dir.exists() or self.allow_remote_fallback

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _resolve_model_source(self) -> str:
        env_path = os.getenv("MEDAGENT_MODEL_DIR")
        if env_path:
            env_dir = Path(env_path)
            if env_dir.exists():
                return str(env_dir)
        if self.local_model_dir.exists():
            return str(self.local_model_dir)
        if self.allow_remote_fallback:
            return self.model_ref
        raise FileNotFoundError(f"Huatuo-VL model not found: {self.local_model_dir}")

    def _load(self) -> None:
        if self._model is not None and self._processor is not None and self._process_vision_info is not None:
            return
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            self._load_error = str(exc)
            raise RuntimeError("Huatuo-VL runtime dependencies are not installed.") from exc

        model_source = self._resolve_model_source()
        processor_source = model_source
        processor_files = (
            Path(model_source) / "preprocessor_config.json",
            Path(model_source) / "processor_config.json",
            Path(model_source) / "tokenizer_config.json",
        )
        if not any(path.exists() for path in processor_files):
            processor_source = self.DEFAULT_BASE_PROCESSOR

        self._processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        self._process_vision_info = process_vision_info
        self._device = next(self._model.parameters()).device

    def _build_messages(self, system_prompt: str, user_prompt: str, image_path: str | None) -> list[dict[str, Any]]:
        content: list[dict[str, str]] = []
        if image_path:
            image_uri = image_path
            if not image_uri.startswith(("http://", "https://", "file://")):
                image_uri = f"file://{image_uri}"
            content.append({"type": "image", "image": image_uri})
        content.append({"type": "text", "text": user_prompt})
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]})
        messages.append({"role": "user", "content": content})
        return messages

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_path: str | None = None,
        max_new_tokens: int = 384,
    ) -> str:
        if not self.available:
            raise RuntimeError("Huatuo-VL local model is not available.")

        self._load()
        assert self._model is not None
        assert self._processor is not None
        assert self._process_vision_info is not None
        assert self._device is not None

        import torch

        messages = self._build_messages(system_prompt=system_prompt, user_prompt=user_prompt, image_path=image_path)
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        trimmed_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self._processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return outputs[0].strip()
