from __future__ import annotations

from pathlib import Path


class PromptRegistry:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[1] / "prompts"

    def load(self, name: str) -> str:
        path = self.root / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(f"prompt profile not found: {path}")
        return path.read_text(encoding="utf-8")
