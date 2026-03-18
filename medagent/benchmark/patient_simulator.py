from __future__ import annotations

import os
from dataclasses import dataclass
import re


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
        self._answered_slots: dict[str, set[str]] = {}

    def reset_case(self, scenario_id: str) -> None:
        self._answered_slots.pop(scenario_id, None)

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

    def _match_rule(self, latest_agent_reply: str, rule: dict) -> bool:
        text = latest_agent_reply.lower()
        keywords = [str(item).lower() for item in rule.get("match_any", [])]
        if not keywords:
            return False
        return any(keyword in text for keyword in keywords)

    def _pick_rule_responses(self, scenario: dict, latest_agent_reply: str) -> list[str]:
        scenario_id = str(scenario.get("id", scenario.get("name", "default_case")))
        answered = self._answered_slots.setdefault(scenario_id, set())
        rules = scenario.get("hidden_case", {}).get("qa_rules", [])
        responses: list[str] = []

        for rule in rules:
            slot = str(rule.get("slot", "")).strip()
            if slot and slot in answered:
                continue
            if self._match_rule(latest_agent_reply, rule):
                response = str(rule.get("response", "")).strip()
                if response:
                    responses.append(response)
                    if slot:
                        answered.add(slot)
            if len(responses) >= 2:
                break
        return responses

    def _style_wrap(self, scenario: dict, text: str) -> str:
        style = str(scenario.get("speaking_style", "plain")).lower()
        education = str(scenario.get("education_level", "")).lower()
        if style == "fragmented":
            return text.replace("；", "，").replace("。", "")
        if style == "anxious":
            return f"{text}，我有点担心。"
        if education in {"middle_school", "junior_middle", "高中", "初中"}:
            return text.replace("呼吸困难", "喘不上气").replace("腹痛", "肚子痛")
        return text

    def _fallback_rule_based(self, scenario: dict, latest_agent_reply: str, dialogue_history: list[dict]) -> str:
        responses = self._pick_rule_responses(scenario, latest_agent_reply)
        if responses:
            return self._style_wrap(scenario, "；".join(responses))

        text = latest_agent_reply
        if any(keyword in text for keyword in ("急诊", "120", "立即就诊")):
            return "好的，我现在去急诊。"
        if any(keyword in text for keyword in ("发热门诊", "呼吸内科", "尽快线下就诊")):
            return "明白了，我今天就去看。"
        if any(keyword in text for keyword in ("检查", "化验", "胸片", "CT", "血常规")):
            return "好的，那我先去把这些检查做了。"

        opening = str(scenario.get("opening", "我最近不太舒服。")).strip()
        if dialogue_history:
            return "还有别的需要我补充的吗？"
        return opening

    def respond(
        self,
        scenario: dict,
        dialogue_history: list[dict],
        *,
        latest_agent_reply: str = "",
        visit_state: dict | None = None,
    ) -> str:
        system_prompt = (
            "你是一个模拟病人。请只根据隐藏病例和已知对话信息作答，不要主动给出诊断结论。"
            "每次回答尽量自然、简洁，不要一次性把所有病史都说完，除非医生明确追问。"
            "如果医生问了多个问题，也优先回答最关键的 1 到 2 个。"
        )
        user_prompt = (
            f"病人设定: {scenario}\n"
            f"对话历史: {dialogue_history}\n"
            f"医生上一句: {latest_agent_reply}\n"
            f"当前 visit 状态: {visit_state or {}}\n"
            "请输出病人下一句回答，尽量自然、简洁。"
        )
        api_text = self._call_api(system_prompt, user_prompt)
        if api_text:
            return api_text.strip()

        return self._fallback_rule_based(scenario, latest_agent_reply, dialogue_history)
