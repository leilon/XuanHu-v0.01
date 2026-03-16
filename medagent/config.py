from dataclasses import dataclass, field


@dataclass
class ModelProfile:
    name: str
    provider: str = "local"
    supports_vision: bool = False
    max_context: int = 8192


@dataclass
class AppConfig:
    default_profile: ModelProfile = field(
        default_factory=lambda: ModelProfile(name="medagent-7b-base")
    )
    max_turns: int = 6
    require_citations: bool = True
    adapter_bank_dir: str = "adapters"
    enable_memory_fusion: bool = True
    emergency_keywords: tuple[str, ...] = (
        "胸痛",
        "呼吸困难",
        "喘不上气",
        "意识不清",
        "抽搐",
        "便血",
        "黑便",
        "高热不退",
    )
