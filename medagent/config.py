from dataclasses import dataclass, field


@dataclass(frozen=True)
class BrandingConfig:
    project_name_en: str = "QingNang-ClinicOS"
    project_name_zh: str = "青囊智诊"
    project_tagline: str = "基于单一医学多模态基模的轻量级医疗 Agent Demo"
    orchestrator_name: str = "总控调度"
    intake_name: str = "门诊专家"
    triage_name: str = "紧急分诊"
    medication_name: str = "用药医师"
    report_name: str = "影像医师"
    education_name: str = "健康顾问"
    memory_name: str = "长期记忆"
    safety_name: str = "安全守卫"

    def section_label(self, task: str) -> str:
        labels = {
            "intake": f"{self.intake_name} | 首程问诊",
            "triage": f"{self.triage_name} | 分诊决策",
            "report": f"{self.report_name} | 报告解读",
            "education": f"{self.education_name} | 医学解释",
            "medication": f"{self.medication_name} | 用药建议",
            "rag_summary": f"{self.orchestrator_name} | 知识依据",
        }
        return labels.get(task, task)


@dataclass
class ModelProfile:
    name: str
    provider: str = "local"
    supports_vision: bool = False
    max_context: int = 8192


@dataclass
class AppConfig:
    default_profile: ModelProfile = field(
        default_factory=lambda: ModelProfile(
            name="FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL",
            provider="local",
            supports_vision=True,
            max_context=16384,
        )
    )
    branding: BrandingConfig = field(default_factory=BrandingConfig)
    runtime_mode: str = "single_model_prompt_routing"
    max_turns: int = 6
    require_citations: bool = True
    adapter_bank_dir: str = "adapters"
    enable_memory_fusion: bool = False
    use_adapter_bank: bool = False
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
