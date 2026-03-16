from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class SimConfig:
    model: str = "gpt-4o-mini"
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"


class PatientSimulator:
    """
    OpenAI-compatible patient simulator.
    Falls back to rule-based replies when API config is unavailable.
    """

    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config or SimConfig()

    def _call_api(self, system_prompt: str, user_prompt: str) -> str | None:
        api_key = os.getenv(self.config.api_key_env, "")
        if not api_key:
            return None
        try:
            from openai import OpenAI
        except Exception:
            return None

        kwargs = {"api_key": api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=self.config.model,
            temperature=0.6,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    def respond(self, scenario: dict, dialogue_history: list[dict]) -> str:
        system_prompt = (
            "你是一个模拟病人。请仅根据给定设定作答，不要主动给出诊断结论。"
            "你可以提供症状、持续时间、过敏史、基础病、用药史。"
        )
        user_prompt = (
            f"病人设定: {scenario}\n"
            f"对话历史: {dialogue_history}\n"
            "请输出病人下一句回答，尽量自然、简洁。"
        )
        api_text = self._call_api(system_prompt, user_prompt)
        if api_text:
            return api_text.strip()

        # Fallback simulation.
        if scenario.get("name") == "fever_cough_case":
            return "发烧两天，最高39度，咳嗽有痰，无已知药物过敏。"
        if scenario.get("name") == "report_case":
            return "我有体检报告，白细胞升高，最近乏力，想知道要不要马上去医院。"
        return "我最近不太舒服，想咨询一下。"

