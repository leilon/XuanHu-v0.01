#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medagent.services.rag import RAGService


def _measure_queries(rag: RAGService, queries: list[str], top_k: int) -> list[dict]:
    rows = []
    for query in queries:
        print(f"[query] {query}", flush=True)
        started = time.perf_counter()
        docs = rag.retrieve_knowledge(query, top_k=top_k)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "query": query,
                "elapsed_sec": elapsed,
                "top_sources": [doc.source for doc in docs[:top_k]],
            }
        )
    return rows


def _summary(rows: list[dict]) -> dict:
    latencies = [row["elapsed_sec"] for row in rows]
    return {
        "count": len(rows),
        "avg_sec": mean(latencies) if latencies else 0.0,
        "median_sec": median(latencies) if latencies else 0.0,
        "min_sec": min(latencies) if latencies else 0.0,
        "max_sec": max(latencies) if latencies else 0.0,
    }


def _write_markdown(output_path: Path, vector_rows: list[dict], lexical_rows: list[dict]) -> None:
    vector_summary = _summary(vector_rows)
    lexical_summary = _summary(lexical_rows)
    improvement = (
        lexical_summary["avg_sec"] / max(vector_summary["avg_sec"], 1e-9)
        if vector_summary["avg_sec"]
        else 0.0
    )
    lines = [
        "# RAG 向量化效率对比",
        "",
        "## 总览",
        f"- `vector_avg_sec`: {vector_summary['avg_sec']:.4f}",
        f"- `lexical_avg_sec`: {lexical_summary['avg_sec']:.4f}",
        f"- `speedup_x`: {improvement:.2f}",
        "",
        "## 向量检索明细",
    ]
    for row in vector_rows:
        lines.append(f"- `{row['query']}` -> {row['elapsed_sec']:.4f}s")
    lines.extend(["", "## 词法/流式检索明细"])
    for row in lexical_rows:
        lines.append(f"- `{row['query']}` -> {row['elapsed_sec']:.4f}s")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure vector vs lexical RAG latency.")
    parser.add_argument("--queries-json", default="data/multiturn_casebook_10.json")
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json-out", default="docs/reports/rag_latency_report.json")
    parser.add_argument("--md-out", default="docs/reports/rag_latency_report_zh.md")
    parser.add_argument("--chunk-file", default="rag/chunks/medical_corpus_cn.jsonl")
    parser.add_argument("--vector-index-dir", default="rag/index/bge_base_zh_v1_5")
    args = parser.parse_args()

    cases = json.loads(Path(args.queries_json).read_text(encoding="utf-8"))
    queries = [item["opening"] for item in cases[: args.max_queries]]

    print(f"[init] measuring {len(queries)} queries", flush=True)
    vector_rag = RAGService(chunk_file=args.chunk_file, vector_index_dir=args.vector_index_dir)
    lexical_rag = RAGService(chunk_file=args.chunk_file, vector_index_dir="rag/index/__missing__")

    # Warm up vector path once so the timing focuses on retrieval rather than first import cost.
    if queries:
        print("[warmup] vector retriever", flush=True)
        vector_rag.retrieve_knowledge(queries[0], top_k=args.top_k)

    print("[measure] vector", flush=True)
    vector_rows = _measure_queries(vector_rag, queries, args.top_k)
    print("[measure] lexical", flush=True)
    lexical_rows = _measure_queries(lexical_rag, queries, args.top_k)

    payload = {
        "vector": {
            "summary": _summary(vector_rows),
            "rows": vector_rows,
        },
        "lexical": {
            "summary": _summary(lexical_rows),
            "rows": lexical_rows,
        },
    }

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_out = Path(args.md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(md_out, vector_rows, lexical_rows)

    print(f"[json] wrote {json_out}")
    print(f"[md] wrote {md_out}")


if __name__ == "__main__":
    main()
