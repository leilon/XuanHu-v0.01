from dataclasses import dataclass


@dataclass
class RetrievedDoc:
    source: str
    chunk: str
    score: float


class RAGService:
    def __init__(self) -> None:
        self.docs = [
            RetrievedDoc(
                source="guide:fever-home-care",
                chunk="低热可先补液休息，持续高热或呼吸困难需及时就医。",
                score=0.82,
            ),
            RetrievedDoc(
                source="drug:acetaminophen",
                chunk="对乙酰氨基酚可用于退热镇痛，需注意总剂量与肝功能风险。",
                score=0.81,
            ),
            RetrievedDoc(
                source="drug:ibuprofen",
                chunk="布洛芬可退热止痛，胃溃疡和肾功能异常人群应谨慎。",
                score=0.77,
            ),
        ]

    def retrieve(self, query: str, top_k: int = 2) -> list[RetrievedDoc]:
        # Placeholder scoring: keep deterministic for benchmark reproducibility.
        query = query.lower()
        ranked = sorted(
            self.docs,
            key=lambda d: d.score + (0.05 if any(tok in d.chunk for tok in [query[:2], "退热", "咳嗽"]) else 0),
            reverse=True,
        )
        return ranked[:top_k]

