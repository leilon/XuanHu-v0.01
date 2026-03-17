from __future__ import annotations

from dataclasses import dataclass
import heapq
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
    def __init__(self, chunk_file: str = "rag/chunks/medical_corpus_cn.jsonl", preload_limit_mb: int = 128) -> None:
        self.expander = QueryExpander()
        self.chunk_file = self._resolve_chunk_file(chunk_file)
        self.preload_limit_bytes = preload_limit_mb * 1024 * 1024
        self.streaming_mode = False
        self.docs = [
            RetrievedDoc(
                source="guide:fever-home-care",
                chunk="低热可先补液休息，持续高热或呼吸困难需要及时就医。",
                score=0.82,
            ),
            RetrievedDoc(
                source="drug:acetaminophen",
                chunk="对乙酰氨基酚可用于退热镇痛，需要注意总剂量与肝功能风险。",
                score=0.81,
            ),
            RetrievedDoc(
                source="drug:ibuprofen",
                chunk="布洛芬可退热止痛，胃溃疡和肾功能异常人群应谨慎。",
                score=0.77,
            ),
        ]
        if self.chunk_file.exists():
            if self.chunk_file.stat().st_size <= self.preload_limit_bytes:
                loaded = self._load_chunk_file(self.chunk_file)
                if loaded:
                    self.docs = loaded
            else:
                self.streaming_mode = True

    def _resolve_chunk_file(self, chunk_file: str) -> Path:
        candidates = [
            Path(chunk_file),
            Path("runtime_assets/rag/chunks/medical_corpus_cn.jsonl"),
            Path("runtime_assets/rag/chunks/medical_corpus.jsonl"),
            Path("/root/autodl-tmp/medagent/rag/chunks/medical_corpus_cn.jsonl"),
            Path("/root/autodl-tmp/medagent/rag/chunks/medical_corpus.jsonl"),
            Path.cwd() / "rag/chunks/medical_corpus_cn.jsonl",
            Path.cwd() / "rag/chunks/medical_corpus.jsonl",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path(chunk_file)

    def _iter_chunk_file(self, path: Path):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield RetrievedDoc(
                    source=str(row.get("source_id", "unknown")),
                    title=str(row.get("title", "")),
                    chunk=str(row.get("chunk", "")),
                    score=float(row.get("score", 0.0)),
                    source_type=str(row.get("source_type", "")),
                    url=str(row.get("url", "")),
                )

    def _load_chunk_file(self, path: Path) -> list[RetrievedDoc]:
        return list(self._iter_chunk_file(path))

    def _tokenize(self, text: str) -> list[str]:
        lowered = text.lower()
        english_tokens = re.findall(r"[a-z0-9]+", lowered)
        chinese_segments = re.findall(r"[\u4e00-\u9fff]+", lowered)
        chinese_tokens: list[str] = []
        for seg in chinese_segments:
            chinese_tokens.extend(seg[i : i + 2] for i in range(max(len(seg) - 1, 1)))
            chinese_tokens.extend(list(seg))
        return english_tokens + chinese_tokens

    def _score_doc(
        self,
        doc: RetrievedDoc,
        query_terms: set[str],
        rewrite_terms: set[str],
        hyde_terms: set[str],
    ) -> float:
        doc_terms = set(self._tokenize(f"{doc.title} {doc.chunk} {doc.source_type}"))
        overlap_query = len(query_terms & doc_terms)
        overlap_rewrite = len(rewrite_terms & doc_terms)
        overlap_hyde = len(hyde_terms & doc_terms)
        return doc.score + overlap_query * 0.18 + overlap_rewrite * 0.1 + overlap_hyde * 0.04

    def _prefilter_terms(self, query_terms: set[str], rewrite_terms: set[str], hyde_terms: set[str]) -> list[str]:
        candidates = query_terms | rewrite_terms | hyde_terms
        filtered = []
        for term in candidates:
            if len(term) >= 2 and re.search(r"[\u4e00-\u9fff]|[a-z0-9]{3,}", term):
                filtered.append(term.lower())
        filtered = sorted(set(filtered), key=len, reverse=True)
        return filtered[:24]

    def retrieve(self, query: str, top_k: int = 2) -> list[RetrievedDoc]:
        expansion = self.expander.expand(query)
        query_terms = set(self._tokenize(expansion.original_query))
        rewrite_terms = set(self._tokenize(expansion.rewritten_query))
        hyde_terms = set(self._tokenize(expansion.hyde_document))

        if self.streaming_mode and self.chunk_file.exists():
            prefilter_terms = self._prefilter_terms(query_terms, rewrite_terms, hyde_terms)
            heap: list[tuple[float, int, RetrievedDoc]] = []
            with open(self.chunk_file, "r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    line = line.strip()
                    if not line:
                        continue
                    lowered = line.lower()
                    if prefilter_terms and not any(term in lowered for term in prefilter_terms):
                        continue
                    row = json.loads(line)
                    doc = RetrievedDoc(
                        source=str(row.get("source_id", "unknown")),
                        title=str(row.get("title", "")),
                        chunk=str(row.get("chunk", "")),
                        score=float(row.get("score", 0.0)),
                        source_type=str(row.get("source_type", "")),
                        url=str(row.get("url", "")),
                    )
                    score = self._score_doc(doc, query_terms, rewrite_terms, hyde_terms)
                    item = (score, idx, doc)
                    if len(heap) < top_k:
                        heapq.heappush(heap, item)
                    elif score > heap[0][0]:
                        heapq.heapreplace(heap, item)
            ranked = sorted(heap, key=lambda item: item[0], reverse=True)
            return [item[2] for item in ranked]

        ranked = sorted(
            self.docs,
            key=lambda doc: self._score_doc(doc, query_terms, rewrite_terms, hyde_terms),
            reverse=True,
        )
        return ranked[:top_k]
