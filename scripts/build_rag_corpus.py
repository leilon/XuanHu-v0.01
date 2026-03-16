#!/usr/bin/env python
"""
Build a chunked diagnosis-oriented RAG corpus from open medical sources.

CPU-only preprocessing:
1. Read downloaded raw files
2. Clean and normalize text
3. Chunk into retrieval units
4. Emit JSONL for later BM25 / vector indexing
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


def _require_bs4():
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        raise RuntimeError("Missing deps. Install with: pip install beautifulsoup4 lxml") from exc
    return BeautifulSoup


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_text(text: str, max_chars: int = 420, overlap: int = 80) -> list[str]:
    text = _clean_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[。！？.!?])", text)
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
                        }
                    )
    return records


def _yield_lab_tests(root: Path) -> list[dict[str, Any]]:
    BeautifulSoup = _require_bs4()
    records: list[dict[str, Any]] = []
    for html_file in sorted((root / "medlineplus_lab_tests").glob("*.html")):
        if html_file.name == "index.html":
            continue
        soup = BeautifulSoup(html_file.read_text(encoding="utf-8"), "lxml")
        title = _clean_text(soup.title.get_text(" ", strip=True) if soup.title else html_file.stem)
        main = soup.find("main") or soup.find("article") or soup.body
        if not main:
            continue
        paragraphs = [
            _clean_text(node.get_text(" ", strip=True))
            for node in main.find_all(["p", "li", "h2", "h3"])
        ]
        text = " ".join(item for item in paragraphs if item)
        if not text:
            continue
        records.append(
            {
                "source_id": f"medlineplus_lab:{html_file.stem}",
                "source_type": "medical_test",
                "title": title,
                "url": "",
                "text": text,
                "tags": ["lab_test", "diagnostics"],
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
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunked RAG corpus for QingNang-ClinicOS")
    parser.add_argument("--raw-root", default="/root/autodl-tmp/medagent/datasets/rag_raw")
    parser.add_argument("--out-file", default="/root/autodl-tmp/medagent/rag/chunks/medical_corpus.jsonl")
    parser.add_argument("--max-chars", type=int, default=420)
    parser.add_argument("--overlap", type=int, default=80)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    records = []
    records.extend(_yield_medlineplus_topics(raw_root))
    records.extend(_yield_lab_tests(raw_root))
    records.extend(_yield_openfda_labels(raw_root))

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
                    "score": 0.2,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_chunks += 1
    print(f"[ok] wrote {total_chunks} chunks to {out_file}")


if __name__ == "__main__":
    main()
