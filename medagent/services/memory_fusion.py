from dataclasses import dataclass

from medagent.services.adapter_bank import AdapterBank
from medagent.services.memory import MemoryStore


@dataclass
class FusionContext:
    task: str
    profile: dict[str, object]
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
            profile=self.memory.build_clinical_snapshot(user_id),
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

        memory_hint: list[str] = []
        if ctx.profile:
            memory_hint.append("已结合长期记忆中的既往病史、过敏史和长期用药")
        if ctx.episodes:
            memory_hint.append("已参考既往同类就诊记录")
        hint = "；".join(memory_hint) if memory_hint else "当前回答主要基于本轮对话"
        return f"{draft}\n[SiMiao-Memory] {hint}，如症状持续或加重请尽快线下复诊。"
