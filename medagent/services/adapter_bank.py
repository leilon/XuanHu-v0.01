import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class AdapterMeta:
    task: str
    adapter_path: str
    base_model: str
    step: int = 0
    dataset: str = ""


class AdapterBank:
    """Simple adapter registry for continual QLoRA learning."""

    def __init__(self, root_dir: str = "adapters") -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_file = self.root / "index.json"
        if not self.index_file.exists():
            self.index_file.write_text("{}", encoding="utf-8")

    def _load_index(self) -> dict[str, dict]:
        return json.loads(self.index_file.read_text(encoding="utf-8"))

    def _save_index(self, payload: dict[str, dict]) -> None:
        self.index_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def register(self, meta: AdapterMeta) -> None:
        index = self._load_index()
        index[meta.task] = asdict(meta)
        self._save_index(index)

    def get(self, task: str) -> AdapterMeta | None:
        index = self._load_index()
        item = index.get(task)
        if not item:
            return None
        return AdapterMeta(**item)

    def pick_task(self, query: str) -> str:
        if any(tok in query for tok in ("报告", "影像", "化验", "白细胞", "CT", "MRI")):
            return "report_qa"
        if any(tok in query for tok in ("药", "用药", "剂量", "禁忌")):
            return "medication_qa"
        return "general_intake"

