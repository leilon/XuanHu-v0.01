#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from medagent.services.vector_retriever import HFTextEmbedder


def count_lines_and_offsets(path: Path) -> list[int]:
    offsets: list[int] = []
    offset = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                offsets.append(offset)
            offset += len(line)
    return offsets


def iter_texts(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield str(row.get("chunk", ""))


def build_embeddings(
    chunk_file: Path,
    embeddings_path: Path,
    offsets_path: Path,
    *,
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str | None,
) -> tuple[int, int]:
    import numpy as np

    offsets = count_lines_and_offsets(chunk_file)
    total = len(offsets)
    offsets_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(offsets_path, np.asarray(offsets, dtype=np.int64))

    embedder = HFTextEmbedder(model_name=model_name, max_length=max_length, device=device)
    probe = embedder.encode(["医学检索测试"], is_query=False, batch_size=1)
    dim = int(probe.shape[1])

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    memmap = np.memmap(embeddings_path, dtype="float16", mode="w+", shape=(total, dim))

    batch_texts: list[str] = []
    start_idx = 0
    processed = 0
    started = time.time()
    for text in iter_texts(chunk_file):
        batch_texts.append(text)
        if len(batch_texts) < batch_size:
            continue
        vectors = embedder.encode(batch_texts, is_query=False, batch_size=batch_size)
        end_idx = start_idx + len(batch_texts)
        memmap[start_idx:end_idx] = vectors.astype("float16")
        processed = end_idx
        start_idx = end_idx
        batch_texts = []
        if processed % (batch_size * 20) == 0:
            elapsed = time.time() - started
            print(f"[embed] processed {processed}/{total} chunks in {elapsed:.1f}s")
            memmap.flush()

    if batch_texts:
        vectors = embedder.encode(batch_texts, is_query=False, batch_size=batch_size)
        end_idx = start_idx + len(batch_texts)
        memmap[start_idx:end_idx] = vectors.astype("float16")
        processed = end_idx

    memmap.flush()
    return total, dim


def build_faiss_index(
    embeddings_path: Path,
    index_path: Path,
    *,
    total: int,
    dim: int,
    nlist: int,
    train_size: int,
) -> None:
    import faiss
    import numpy as np

    memmap = np.memmap(embeddings_path, dtype="float16", mode="r", shape=(total, dim))
    sample_size = min(total, train_size)
    step = max(total // sample_size, 1)
    sample = np.asarray(memmap[::step][:sample_size], dtype="float32")

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(sample)

    add_batch = 50000
    for start in range(0, total, add_batch):
        end = min(start + add_batch, total)
        batch = np.asarray(memmap[start:end], dtype="float32")
        index.add(batch)
        print(f"[faiss] added {end}/{total}")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def write_manifest(
    manifest_path: Path,
    *,
    model_name: str,
    dim: int,
    total: int,
    nlist: int,
    max_length: int,
    device: str | None,
) -> None:
    manifest = {
        "model_name": model_name,
        "dim": dim,
        "total": total,
        "metric": "ip",
        "index_type": "IndexIVFFlat",
        "nlist": nlist,
        "nprobe": min(24, max(8, nlist // 128)),
        "max_length": max_length,
        "query_prefix": HFTextEmbedder.DEFAULT_QUERY_PREFIX,
        "device": device or "",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector embeddings and FAISS index for QingNang RAG.")
    parser.add_argument("--chunk-file", default="/root/autodl-tmp/medagent/rag/chunks/medical_corpus_cn.jsonl")
    parser.add_argument("--index-dir", default="/root/autodl-tmp/medagent/rag/index/bge_base_zh_v1_5")
    parser.add_argument("--model-name", default="BAAI/bge-base-zh-v1.5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    chunk_file = Path(args.chunk_file)
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = index_dir / "embeddings.f16.memmap"
    offsets_path = index_dir / "offsets.npy"
    index_path = index_dir / "faiss_ivf_ip.index"
    manifest_path = index_dir / "manifest.json"

    print(f"[start] chunk_file={chunk_file}")
    total, dim = build_embeddings(
        chunk_file,
        embeddings_path,
        offsets_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    print(f"[done] embeddings total={total} dim={dim}")

    adjusted_nlist = min(args.nlist, max(256, int(math.sqrt(total) * 4)))
    build_faiss_index(
        embeddings_path,
        index_path,
        total=total,
        dim=dim,
        nlist=adjusted_nlist,
        train_size=args.train_size,
    )
    write_manifest(
        manifest_path,
        model_name=args.model_name,
        dim=dim,
        total=total,
        nlist=adjusted_nlist,
        max_length=args.max_length,
        device=args.device,
    )
    print(f"[ok] index written to {index_dir}")


if __name__ == "__main__":
    main()
