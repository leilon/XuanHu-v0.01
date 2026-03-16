from dataclasses import dataclass, field


@dataclass(frozen=True)
class BrandingConfig:
    project_name_en: str = "QingNang-ClinicOS"
    project_name_zh: str = "青囊"
    project_tagline: str = "多 Agent 医疗问诊与报告解读系统"
    orchestrator_name: str = "QiBo-Orchestrator"
    intake_name: str = "BianQue-Intake"
    triage_name: str = "HuaTuo-Triage"
    medication_name: str = "ShenNong-Medication"
    report_name: str = "CangGong-Report"
    memory_name: str = "SiMiao-Memory"
    safety_name: str = "ZhongJing-Guard"

    def section_label(self, task: str) -> str:
        labels = {
            "intake": f"{self.intake_name} | 病情采集",
            "triage": f"{self.triage_name} | 分诊建议",
            "report": f"{self.report_name} | 报告分析",
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
        default_factory=lambda: ModelProfile(name="qingnang-qwen2.5-7b")
    )
    branding: BrandingConfig = field(default_factory=BrandingConfig)
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
