#!/usr/bin/env python
"""
Build a chunked diagnosis-oriented RAG corpus from downloaded medical sources.

CPU-only preprocessing:
1. Read downloaded raw files
2. Clean and normalize text
3. Chunk into retrieval units
4. Emit JSONL for later BM25 / vector indexing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET


def _require_bs4():
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        raise RuntimeError("Missing deps. Install with: pip install beautifulsoup4 lxml") from exc
    return BeautifulSoup


def _html_soup(BeautifulSoup, text: str):
    for parser in ("lxml", "html.parser"):
        try:
            return BeautifulSoup(text, parser)
        except Exception:
            continue
    raise RuntimeError("Could not initialize an HTML parser for BeautifulSoup.")


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_text(text: str, max_chars: int = 420, overlap: int = 80) -> list[str]:
    text = _clean_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[。！？?!；;\.])", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) <= max_chars:
            current += sentence
            continue
        if current:
            chunks.append(current)
            current = current[-overlap:] + sentence if overlap else sentence
        else:
            chunks.append(sentence[:max_chars])
            current = sentence[max_chars - overlap :] if overlap and len(sentence) > max_chars else ""
    if current:
        chunks.append(current)
    return [_clean_text(chunk) for chunk in chunks if _clean_text(chunk)]


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _find_text(node: ET.Element, candidates: tuple[str, ...]) -> str:
    for elem in node.iter():
        tag = _strip_ns(elem.tag).lower()
        if tag in candidates and elem.text:
            return _clean_text(elem.text)
    return ""


def _find_all_texts(node: ET.Element, candidates: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for elem in node.iter():
        tag = _strip_ns(elem.tag).lower()
        if tag in candidates and elem.text:
            text = _clean_text(elem.text)
            if text and text not in values:
                values.append(text)
    return values


def _yield_medlineplus_topics(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for archive in (root / "medlineplus_health_topics").glob("*.zip"):
        with zipfile.ZipFile(archive, "r") as bundle:
            xml_names = [name for name in bundle.namelist() if name.endswith(".xml")]
            for xml_name in xml_names:
                data = bundle.read(xml_name)
                tree = ET.fromstring(data)
                for topic in tree.iter():
                    if _strip_ns(topic.tag).lower() != "health-topic":
                        continue
                    title = _find_text(topic, ("title",))
                    summary = _find_text(topic, ("full-summary", "summary"))
                    also_called = _find_all_texts(topic, ("also-called",))
                    groups = _find_all_texts(topic, ("group",))
                    url = _find_text(topic, ("url",))
                    body = " ".join(part for part in [summary, "；".join(also_called)] if part)
                    if not title or not body:
                        continue
                    records.append(
                        {
                            "source_id": f"medlineplus_topic:{title}",
                            "source_type": "health_topic",
                            "title": title,
                            "url": url,
                            "text": body,
                            "tags": groups,
                            "score": 0.65,
                        }
                    )
    return records


def _yield_openfda_labels(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for json_file in (root / "openfda_drug_labels").glob("*.json"):
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        results = payload.get("results", [])
        if not results:
            continue
        row = results[0]
        openfda = row.get("openfda", {})
        title = ", ".join(openfda.get("generic_name", []) or [json_file.stem])
        sections = []
        for key in (
            "indications_and_usage",
            "contraindications",
            "warnings",
            "dosage_and_administration",
            "drug_interactions",
            "use_in_specific_populations",
        ):
            value = row.get(key, [])
            if isinstance(value, list):
                sections.extend(str(item) for item in value[:2])
        text = _clean_text(" ".join(sections))
        if not text:
            continue
        records.append(
            {
                "source_id": f"openfda_drug:{json_file.stem}",
                "source_type": "drug_label",
                "title": title,
                "url": "",
                "text": text,
                "tags": openfda.get("pharm_class_epc", [])[:5],
                "score": 0.82,
            }
        )
    return records


def _yield_seed_pages(root: Path) -> list[dict[str, Any]]:
    BeautifulSoup = _require_bs4()
    records: list[dict[str, Any]] = []
    seed_root = root / "seed_pages"
    for meta_file in seed_root.glob("*.meta.json"):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        html_file = seed_root / f"{meta_file.stem[:-5]}.html"
        if not html_file.exists():
            continue
        soup = _html_soup(BeautifulSoup, html_file.read_text(encoding="utf-8", errors="ignore"))
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(attrs={"role": "main"})
            or soup.find(id="maincontent")
            or soup.body
        )
        if not main:
            continue
        blocks: list[str] = []
        for node in main.find_all(["h1", "h2", "h3", "p", "li"]):
            text = _clean_text(node.get_text(" ", strip=True))
            if not text or len(text) < 20:
                continue
            if text not in blocks:
                blocks.append(text)
        text = " ".join(blocks)
        if not text:
            continue
        records.append(
            {
                "source_id": meta["source_id"],
                "source_type": meta["source_type"],
                "title": meta["title"],
                "url": meta["url"],
                "text": text,
                "tags": meta.get("tags", []) + [meta.get("specialty", "")],
                "score": 0.78,
            }
        )
    return records


def _default_hf_manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "cn_rag_hf_manifest.json"


def _load_hf_manifest_map(manifest_path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = manifest_path or _default_hf_manifest_path()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict) or "repo_id" not in item:
            continue
        local_name = item.get("local_name") or item["repo_id"].replace("/", "__")
        mapping[local_name] = item
    return mapping




def _iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row
        return

    if path.suffix != ".json":
        return

    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row
        return

    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row
        return

    if isinstance(payload, dict):
        for key in ("data", "rows", "items", "train"):
            value = payload.get(key)
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        yield row
                return
        for value in payload.values():
            if isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        yield row
                return


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return _clean_text(" ".join(_stringify(item) for item in value if _stringify(item)))
    if isinstance(value, dict):
        return _clean_text(
            " ".join(f"{key}: {_stringify(val)}" for key, val in value.items() if _stringify(val))
        )
    return _clean_text(str(value))


def _pick_field(row: dict[str, Any], candidates: tuple[str, ...]) -> str:
    lowered = {key.lower(): key for key in row.keys()}
    for candidate in candidates:
        actual = lowered.get(candidate.lower())
        if actual is None:
            continue
        value = _stringify(row.get(actual))
        if value:
            return value
    return ""


def _normalize_hf_row(row: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any] | None:
    text = _pick_field(
        row,
        (
            "text",
            "content",
            "document",
            "body",
            "paragraph",
            "article",
            "context",
        ),
    )
    question = _pick_field(
        row,
        (
            "question",
            "Question",
            "query",
            "ask",
            "instruction",
            "title",
            "input",
            "prompt",
            "questions",
        ),
    )
    answer = _pick_field(
        row,
        (
            "answer",
            "Answer",
            "response",
            "Response",
            "output",
            "summary",
            "answers",
        ),
    )

    if text and len(text) >= 80:
        body = text
    elif question and answer:
        body = f"问题：{question} 回答：{answer}"
    elif question and text:
        body = f"问题：{question} 资料：{text}"
    else:
        return None

    body = _clean_text(body)
    if len(body) < 40:
        return None

    title = question[:80] if question else meta.get("title", meta["repo_id"])
    tags = list(dict.fromkeys(meta.get("tags", []) + [meta.get("bucket", "")]))
    return {
        "source_id": f"{meta['repo_id']}:{hashlib.sha1(body.encode('utf-8')).hexdigest()[:16]}",
        "source_type": meta.get("source_type", "cn_medical_corpus"),
        "title": title,
        "url": f"https://hf-mirror.com/datasets/{meta['repo_id']}",
        "text": body,
        "tags": [tag for tag in tags if tag],
        "score": float(meta.get("base_score", 0.45)),
    }


def _yield_hf_cn_corpora(root: Path, manifest_path: Path | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    corpora_root = root / "hf_corpora"
    if not corpora_root.exists():
        return records

    manifest_map = _load_hf_manifest_map(manifest_path)
    for repo_dir in sorted(corpora_root.iterdir()):
        if not repo_dir.is_dir():
            continue
        meta_file = repo_dir / "source_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        else:
            meta = manifest_map.get(repo_dir.name)
        if not meta:
            continue
        for file_path in repo_dir.rglob("*"):
            if file_path.name == "source_meta.json" or file_path.suffix not in {".json", ".jsonl"}:
                continue
            for row in _iter_json_records(file_path):
                normalized = _normalize_hf_row(row, meta)
                if normalized:
                    records.append(normalized)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunked RAG corpus for QingNang-ClinicOS")
    parser.add_argument("--raw-root", default="/root/autodl-tmp/medagent/datasets/rag_raw")
    parser.add_argument("--out-file", default="/root/autodl-tmp/medagent/rag/chunks/medical_corpus_cn.jsonl")
    parser.add_argument("--max-chars", type=int, default=420)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--hf-manifest", default=str(_default_hf_manifest_path()))
    parser.add_argument("--only-cn", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    if not args.only_cn:
        records.extend(_yield_medlineplus_topics(raw_root))
        records.extend(_yield_openfda_labels(raw_root))
        records.extend(_yield_seed_pages(raw_root))
    records.extend(_yield_hf_cn_corpora(raw_root, manifest_path=Path(args.hf_manifest)))

    total_chunks = 0
    with open(out_file, "w", encoding="utf-8") as handle:
        for record in records:
            chunks = _chunk_text(record["text"], max_chars=args.max_chars, overlap=args.overlap)
            for idx, chunk in enumerate(chunks):
                row = {
                    "source_id": record["source_id"],
                    "source_type": record["source_type"],
                    "title": record["title"],
                    "url": record["url"],
                    "chunk_id": idx,
                    "chunk": chunk,
                    "tags": record["tags"],
                    "score": record["score"],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_chunks += 1
    print(f"[ok] wrote {total_chunks} chunks to {out_file}")


if __name__ == "__main__":
    main()
