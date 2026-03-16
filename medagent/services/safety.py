from medagent.config import AppConfig


class SafetyGuard:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def detect_risk(self, text: str) -> str:
        if any(k in text for k in self.config.emergency_keywords):
            return "high"
        if any(k in text for k in ("发烧", "咳嗽", "头痛", "腹痛")):
            return "medium"
        return "low"

    def enforce(self, draft: str, risk_level: str) -> str:
        if risk_level == "high":
            return (
                draft
                + "\n\n[安全提示] 该描述可能涉及急症风险，请立即线下就医或呼叫急救。"
            )
        return draft

