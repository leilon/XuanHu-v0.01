from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class HFTextEmbedder:
    """Lazy HF embedder for Chinese retrieval."""

    DEFAULT_MODEL = "BAAI/bge-base-zh-v1.5"
    DEFAULT_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

    def __init__(
        self,
        model_name: str | None = None,
        query_prefix: str | None = None,
        max_length: int = 256,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self.query_prefix = query_prefix or self.DEFAULT_QUERY_PREFIX
        self.max_length = max_length
        self.device = device or os.getenv("MEDAGENT_EMBED_DEVICE")
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._nnf: Any | None = None

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Embedding dependencies are not installed.") from exc

        self._torch = torch
        self._nnf = F
        target_device = self.device
        if not target_device:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = target_device
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self._model.to(self.device)
        self._model.eval()

    def _mean_pool(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * input_mask_expanded).sum(1)
        counts = input_mask_expanded.sum(1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int = 64):
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._nnf is not None

        all_embeddings = []
        prefix = f"{self.query_prefix} " if is_query and self.query_prefix else ""

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            if prefix:
                batch = [f"{prefix}{text}" for text in batch]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.no_grad():
                output = self._model(**encoded)
                embeddings = self._mean_pool(output, encoded["attention_mask"])
                embeddings = self._nnf.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.detach().cpu())
        return self._torch.cat(all_embeddings, dim=0).numpy()


class FaissVectorRetriever:
    def __init__(self, chunk_file: str | Path, index_dir: str | Path) -> None:
        self.chunk_file = Path(chunk_file)
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "faiss_ivf_ip.index"
        self.offsets_path = self.index_dir / "offsets.npy"
        self.manifest_path = self.index_dir / "manifest.json"
        self._faiss = None
        self._np = None
        self._index = None
        self._offsets = None
        self._manifest: dict[str, Any] | None = None
        self._embedder: HFTextEmbedder | None = None

    @property
    def available(self) -> bool:
        return self.index_path.exists() and self.offsets_path.exists() and self.manifest_path.exists() and self.chunk_file.exists()

    def _load(self) -> None:
        if self._index is not None and self._offsets is not None and self._manifest is not None:
            return
        if not self.available:
            raise FileNotFoundError("FAISS index artifacts are missing.")
        try:
            import faiss
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("faiss/numpy dependencies are not installed.") from exc

        self._faiss = faiss
        self._np = np
        self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._offsets = np.load(self.offsets_path, mmap_mode="r")
        self._index = faiss.read_index(str(self.index_path))
        self._index.nprobe = int(self._manifest.get("nprobe", 24))
        self._embedder = HFTextEmbedder(
            model_name=str(self._manifest.get("model_name", HFTextEmbedder.DEFAULT_MODEL)),
            query_prefix=str(self._manifest.get("query_prefix", HFTextEmbedder.DEFAULT_QUERY_PREFIX)),
            max_length=int(self._manifest.get("max_length", 256)),
            device=str(self._manifest.get("device", "")) or None,
        )

    def _read_row(self, idx: int) -> dict[str, Any]:
        assert self._offsets is not None
        with self.chunk_file.open("rb") as handle:
            handle.seek(int(self._offsets[idx]))
            line = handle.readline().decode("utf-8").strip()
        return json.loads(line)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        self._load()
        assert self._index is not None
        assert self._embedder is not None
        assert self._np is not None

        query_vector = self._embedder.encode([query], is_query=True, batch_size=1).astype("float32")
        scores, indices = self._index.search(query_vector, top_k)
        output: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self._read_row(int(idx))
            output.append(
                {
                    "source": str(row.get("source_id", "unknown")),
                    "title": str(row.get("title", "")),
                    "chunk": str(row.get("chunk", "")),
                    "score": float(score),
                    "source_type": str(row.get("source_type", "")),
                    "url": str(row.get("url", "")),
                }
            )
        return output
