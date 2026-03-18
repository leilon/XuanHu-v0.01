from medagent.config import AppConfig


class SafetyGuard:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def detect_risk(self, text: str) -> str:
        if any(k in text for k in ("胸口压着疼", "喘不过气", "发黑发亮", "嘴唇发胀", "喉咙发紧", "说话不清")):
            return "high"
        if any(k in text for k in self.config.emergency_keywords):
            return "high"
        if any(k in text for k in ("发热", "咳嗽", "头痛", "腹痛", "尿痛", "皮疹")):
            return "medium"
        return "low"

    def enforce(self, draft: str, risk_level: str) -> str:
        if risk_level == "high":
            return draft + "\n\n[安全提示] 当前描述可能涉及急危重风险，请立即线下就医或呼叫急救。"
        return draft
