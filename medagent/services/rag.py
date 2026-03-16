from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path

from medagent.services.query_expansion import QueryExpander


@dataclass
class RetrievedDoc:
    source: str
    chunk: str
    score: float
    title: str = ""
    source_type: str = ""
    url: str = ""


class RAGService:
    def __init__(self, chunk_file: str = "rag/chunks/medical_corpus.jsonl") -> None:
        self.expander = QueryExpander()
        self.chunk_file = self._resolve_chunk_file(chunk_file)
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
        if self.chunk_file.exists():
            loaded = self._load_chunk_file(self.chunk_file)
            if loaded:
                self.docs = loaded

    def _resolve_chunk_file(self, chunk_file: str) -> Path:
        candidates = [
            Path(chunk_file),
            Path("runtime_assets/rag/chunks/medical_corpus.jsonl"),
            Path("/root/autodl-tmp/medagent/rag/chunks/medical_corpus.jsonl"),
        ]
        env_path = Path(Path.cwd(), "")
        if env_path:
            candidates.append(Path.cwd() / "rag/chunks/medical_corpus.jsonl")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path(chunk_file)

    def _load_chunk_file(self, path: Path) -> list[RetrievedDoc]:
        docs: list[RetrievedDoc] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                docs.append(
                    RetrievedDoc(
                        source=str(row.get("source_id", "unknown")),
                        title=str(row.get("title", "")),
                        chunk=str(row.get("chunk", "")),
                        score=float(row.get("score", 0.0)),
                        source_type=str(row.get("source_type", "")),
                        url=str(row.get("url", "")),
                    )
                )
        return docs

    def _tokenize(self, text: str) -> list[str]:
        lowered = text.lower()
        english_tokens = re.findall(r"[a-z0-9]+", lowered)
        chinese_segments = re.findall(r"[\u4e00-\u9fff]+", lowered)
        chinese_tokens: list[str] = []
        for seg in chinese_segments:
            chinese_tokens.extend(seg[i : i + 2] for i in range(max(len(seg) - 1, 1)))
            chinese_tokens.extend(list(seg))
        return english_tokens + chinese_tokens

    def retrieve(self, query: str, top_k: int = 2) -> list[RetrievedDoc]:
        expansion = self.expander.expand(query)
        query_terms = set(self._tokenize(expansion.original_query))
        rewrite_terms = set(self._tokenize(expansion.rewritten_query))
        hyde_terms = set(self._tokenize(expansion.hyde_document))

        def score(doc: RetrievedDoc) -> float:
            doc_terms = set(self._tokenize(f"{doc.title} {doc.chunk} {doc.source_type}"))
            overlap_query = len(query_terms & doc_terms)
            overlap_rewrite = len(rewrite_terms & doc_terms)
            overlap_hyde = len(hyde_terms & doc_terms)
            return doc.score + overlap_query * 0.18 + overlap_rewrite * 0.1 + overlap_hyde * 0.04

        ranked = sorted(self.docs, key=score, reverse=True)
        return ranked[:top_k]
