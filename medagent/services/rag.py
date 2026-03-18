from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import re
from pathlib import Path
from typing import Any

from medagent.services.memory import MemoryStore
from medagent.services.query_expansion import QueryExpander
from medagent.services.vector_retriever import FaissVectorRetriever


@dataclass
class RetrievedDoc:
    source: str
    chunk: str
    score: float
    title: str = ""
    source_type: str = ""
    url: str = ""


@dataclass
class RetrievedMemory:
    source: str
    content: str
    score: float
    memory_type: str


@dataclass
class RAGBundle:
    knowledge_docs: list[RetrievedDoc]
    memory_hits: list[RetrievedMemory]


class RAGService:
    def __init__(
        self,
        chunk_file: str = "rag/chunks/medical_corpus_cn.jsonl",
        vector_index_dir: str = "rag/index/bge_base_zh_v1_5",
        preload_limit_mb: int = 128,
        max_stream_lines: int = 250000,
        max_stream_hits: int = 5000,
    ) -> None:
        self.expander = QueryExpander()
        self.chunk_file = self._resolve_chunk_file(chunk_file)
        self.vector_index_dir = self._resolve_index_dir(vector_index_dir)
        self.preload_limit_bytes = preload_limit_mb * 1024 * 1024
        self.max_stream_lines = max_stream_lines
        self.max_stream_hits = max_stream_hits
        self.streaming_mode = False
        self.vector_retriever = FaissVectorRetriever(self.chunk_file, self.vector_index_dir)
        self.docs = [
            RetrievedDoc(
                source="guide:fever-home-care",
                chunk="低热可先补液休息，持续高热、胸痛或呼吸困难需要及时就医。",
                score=0.82,
                source_type="symptom_guide",
            ),
            RetrievedDoc(
                source="drug:acetaminophen",
                chunk="对乙酰氨基酚可用于退热镇痛，需要关注总剂量以及肝功能风险。",
                score=0.81,
                source_type="drug_label",
            ),
            RetrievedDoc(
                source="drug:ibuprofen",
                chunk="布洛芬可退热止痛，胃溃疡、消化道出血和肾功能异常人群应谨慎使用。",
                score=0.77,
                source_type="drug_label",
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

    def _resolve_index_dir(self, vector_index_dir: str) -> Path:
        candidates = [
            Path(vector_index_dir),
            Path("runtime_assets/rag/index/bge_base_zh_v1_5"),
            Path("/root/autodl-tmp/medagent/rag/index/bge_base_zh_v1_5"),
            Path.cwd() / "rag/index/bge_base_zh_v1_5",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path(vector_index_dir)

    def _iter_chunk_file(self, path: Path):
        with path.open("r", encoding="utf-8") as handle:
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

    def _score_text_overlap(self, text: str, query_terms: set[str], rewrite_terms: set[str], hyde_terms: set[str]) -> float:
        doc_terms = set(self._tokenize(text))
        overlap_query = len(query_terms & doc_terms)
        overlap_rewrite = len(rewrite_terms & doc_terms)
        overlap_hyde = len(hyde_terms & doc_terms)
        return overlap_query * 0.18 + overlap_rewrite * 0.1 + overlap_hyde * 0.04

    def _score_doc(
        self,
        doc: RetrievedDoc,
        query_terms: set[str],
        rewrite_terms: set[str],
        hyde_terms: set[str],
    ) -> float:
        text = f"{doc.title} {doc.chunk} {doc.source_type}"
        return doc.score + self._score_text_overlap(text, query_terms, rewrite_terms, hyde_terms)

    def _prefilter_terms(self, query_terms: set[str], rewrite_terms: set[str], hyde_terms: set[str]) -> list[str]:
        candidates = query_terms | rewrite_terms | hyde_terms
        filtered = []
        for term in candidates:
            if len(term) >= 2 and re.search(r"[\u4e00-\u9fff]|[a-z0-9]{3,}", term):
                filtered.append(term.lower())
        filtered = sorted(set(filtered), key=len, reverse=True)
        return filtered[:24]

    def retrieve_knowledge(self, query: str, top_k: int = 2) -> list[RetrievedDoc]:
        if self.vector_retriever.available:
            try:
                vector_hits = self.vector_retriever.retrieve(query, top_k=top_k)
                if vector_hits:
                    return [
                        RetrievedDoc(
                            source=item["source"],
                            title=item.get("title", ""),
                            chunk=item["chunk"],
                            score=float(item["score"]),
                            source_type=item.get("source_type", ""),
                            url=item.get("url", ""),
                        )
                        for item in vector_hits
                    ]
            except Exception:
                pass

        expansion = self.expander.expand(query)
        query_terms = set(self._tokenize(expansion.original_query))
        rewrite_terms = set(self._tokenize(expansion.rewritten_query))
        hyde_terms = set(self._tokenize(expansion.hyde_document))

        if self.streaming_mode and self.chunk_file.exists():
            prefilter_terms = self._prefilter_terms(query_terms, rewrite_terms, hyde_terms)
            heap: list[tuple[float, int, RetrievedDoc]] = []
            matched_hits = 0
            with self.chunk_file.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    if idx >= self.max_stream_lines:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    lowered = line.lower()
                    if prefilter_terms and not any(term in lowered for term in prefilter_terms):
                        continue
                    matched_hits += 1
                    if matched_hits > self.max_stream_hits:
                        break
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

    def _iter_memory_candidates(self, user_id: str, memory_store: MemoryStore) -> list[RetrievedMemory]:
        profile = memory_store.build_clinical_snapshot(user_id)
        candidates: list[RetrievedMemory] = []

        important_profile_keys = {
            "allergies": 0.42,
            "current_meds": 0.38,
            "chronic_history": 0.34,
            "recent_assessment": 0.28,
            "pregnancy_status": 0.3,
        }
        for key, value in profile.items():
            if isinstance(value, list):
                for item in value:
                    text = f"{key}: {item}"
                    candidates.append(
                        RetrievedMemory(
                            source=f"profile:{key}",
                            content=text,
                            score=important_profile_keys.get(key, 0.2),
                            memory_type="profile",
                        )
                    )
            elif value not in (None, "", []):
                text = f"{key}: {value}"
                candidates.append(
                    RetrievedMemory(
                        source=f"profile:{key}",
                        content=text,
                        score=important_profile_keys.get(key, 0.16),
                        memory_type="profile",
                    )
                )

        recent_turns = memory_store.get_recent(user_id)
        for idx, turn in enumerate(recent_turns[-6:], start=1):
            candidates.append(
                RetrievedMemory(
                    source=f"recent:{idx}",
                    content=turn,
                    score=0.08,
                    memory_type="recent_turn",
                )
            )

        for idx, episode in enumerate(memory_store.recall_episodes(user_id, "", top_k=8), start=1):
            candidates.append(
                RetrievedMemory(
                    source=f"episode:{idx}",
                    content=f"{episode.get('topic', 'episode')}: {episode.get('content', '')}",
                    score=0.18,
                    memory_type="episode",
                )
            )

        for idx, record in enumerate(memory_store.get_visit_records(user_id)[-5:], start=1):
            summary = str(record.get("human_readable_summary", "") or record.get("chief_complaint", "")).strip()
            if not summary:
                continue
            candidates.append(
                RetrievedMemory(
                    source=f"visit_record:{idx}",
                    content=summary,
                    score=0.24,
                    memory_type="visit_record",
                )
            )

        for idx, document in enumerate(memory_store.get_source_documents(user_id)[-5:], start=1):
            summary = str(document.get("summary", "") or document.get("title", "")).strip()
            if not summary:
                continue
            candidates.append(
                RetrievedMemory(
                    source=f"source_document:{idx}",
                    content=summary,
                    score=0.22,
                    memory_type="source_document",
                )
            )
        return candidates

    def retrieve_memory(
        self,
        user_id: str,
        query: str,
        memory_store: MemoryStore,
        top_k: int = 3,
    ) -> list[RetrievedMemory]:
        expansion = self.expander.expand(query)
        query_terms = set(self._tokenize(expansion.original_query))
        rewrite_terms = set(self._tokenize(expansion.rewritten_query))
        hyde_terms = set(self._tokenize(expansion.hyde_document))

        scored: list[RetrievedMemory] = []
        for candidate in self._iter_memory_candidates(user_id, memory_store):
            bonus = self._score_text_overlap(candidate.content, query_terms, rewrite_terms, hyde_terms)
            score = candidate.score + bonus
            scored.append(
                RetrievedMemory(
                    source=candidate.source,
                    content=candidate.content,
                    score=score,
                    memory_type=candidate.memory_type,
                )
            )

        ranked = sorted(scored, key=lambda item: item.score, reverse=True)
        output: list[RetrievedMemory] = []
        seen_content: set[str] = set()
        for item in ranked:
            if item.content in seen_content:
                continue
            seen_content.add(item.content)
            output.append(item)
            if len(output) >= top_k:
                break
        return output

    def build_bundle(
        self,
        user_id: str,
        query: str,
        memory_store: MemoryStore,
        knowledge_top_k: int = 3,
        memory_top_k: int = 3,
    ) -> RAGBundle:
        return RAGBundle(
            knowledge_docs=self.retrieve_knowledge(query, top_k=knowledge_top_k),
            memory_hits=self.retrieve_memory(user_id, query, memory_store, top_k=memory_top_k),
        )

    def retrieve(self, query: str, top_k: int = 2) -> list[RetrievedDoc]:
        return self.retrieve_knowledge(query, top_k=top_k)
