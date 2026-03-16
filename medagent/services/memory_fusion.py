from dataclasses import dataclass
from typing import Any

from medagent.services.adapter_bank import AdapterBank
from medagent.services.memory import MemoryStore


@dataclass
class FusionContext:
    task: str
    profile: dict[str, str]
    recent_turns: list[str]
    episodes: list[dict[str, str]]


class MemoryFusionEngine:
    """
    Combines external long-term memory with task-specific adapters.
    If local HF stack is unavailable, falls back to deterministic template generation.
    """

    def __init__(self, memory: MemoryStore, adapter_bank: AdapterBank) -> None:
        self.memory = memory
        self.adapter_bank = adapter_bank

    def build_context(self, user_id: str, query: str) -> FusionContext:
        task = self.adapter_bank.pick_task(query)
        return FusionContext(
            task=task,
            profile=self.memory.get_profile(user_id),
            recent_turns=self.memory.get_recent(user_id),
            episodes=self.memory.recall_episodes(user_id, query, top_k=3),
        )

    def _build_prompt(self, query: str, draft: str, ctx: FusionContext) -> str:
        profile = ", ".join(f"{k}={v}" for k, v in ctx.profile.items()) or "none"
        recent = " | ".join(ctx.recent_turns[-3:]) or "none"
        epis = " | ".join(item["content"] for item in ctx.episodes) or "none"
        return (
            "You are a safe medical assistant.\n"
            f"Task: {ctx.task}\n"
            f"User query: {query}\n"
            f"Draft answer: {draft}\n"
            f"Long-term profile: {profile}\n"
            f"Recent context: {recent}\n"
            f"Episodic memory: {epis}\n"
            "Rewrite a concise, safer, personalized final answer."
        )

    def _generate_with_hf(self, prompt: str, adapter_path: str | None = None) -> str | None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception:
            return None

        base_model = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=220, do_sample=False)
        return tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()

    def generate(self, user_id: str, query: str, draft: str) -> str:
        ctx = self.build_context(user_id, query)
        adapter = self.adapter_bank.get(ctx.task)
        prompt = self._build_prompt(query, draft, ctx)
        generated = self._generate_with_hf(prompt, adapter.adapter_path if adapter else None)
        if generated:
            return generated

        memory_hint = []
        if ctx.profile:
            memory_hint.append("已结合你的历史画像")
        if ctx.episodes:
            memory_hint.append("已参考你过往就诊对话")
        hint = "，".join(memory_hint) if memory_hint else "基于当前对话"
        return f"{draft}\n[记忆融合] {hint}，建议遵医嘱并持续复诊跟踪。"

