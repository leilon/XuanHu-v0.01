#!/usr/bin/env python
"""
Download diagnosis-oriented open medical sources for RAG.

This step is CPU/network-bound and can be done before renting GPUs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urljoin


MEDLINEPLUS_XML_PAGE = "https://medlineplus.gov/xml.html"
MEDLINEPLUS_LAB_TESTS_INDEX = "https://medlineplus.gov/lab-tests/"
OPENFDA_ENDPOINT = "https://api.fda.gov/drug/label.json"

DEFAULT_DRUGS = [
    "acetaminophen",
    "ibuprofen",
    "amoxicillin",
    "azithromycin",
    "metformin",
    "omeprazole",
    "losartan",
    "amlodipine",
    "aspirin",
    "albuterol",
]


def _require_deps():
    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception as exc:
        raise RuntimeError(
            "Missing RAG download deps. Install with: pip install requests beautifulsoup4 lxml"
        ) from exc
    return requests, BeautifulSoup


def _download_medlineplus_xml(root: Path, requests, BeautifulSoup) -> None:
    response = requests.get(MEDLINEPLUS_XML_PAGE, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    xml_link = None
    for anchor in soup.find_all("a", href=True):
        text = anchor.get_text(" ", strip=True).lower()
        href = anchor["href"]
        if "compressed health topic xml" in text and href.endswith(".zip"):
            xml_link = urljoin(MEDLINEPLUS_XML_PAGE, href)
            break
    if not xml_link:
        raise RuntimeError("Could not locate MedlinePlus compressed XML link.")

    target_dir = root / "medlineplus_health_topics"
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = requests.get(xml_link, timeout=120)
    payload.raise_for_status()
    archive_path = target_dir / Path(xml_link).name
    archive_path.write_bytes(payload.content)
    (target_dir / "source_url.txt").write_text(xml_link, encoding="utf-8")


def _download_lab_tests(root: Path, requests, BeautifulSoup, max_pages: int) -> None:
    response = requests.get(MEDLINEPLUS_LAB_TESTS_INDEX, timeout=60)
    response.raise_for_status()
    target_dir = root / "medlineplus_lab_tests"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "index.html").write_text(response.text, encoding="utf-8")

    soup = BeautifulSoup(response.text, "lxml")
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if href.startswith("/lab-tests/") and href != "/lab-tests/":
            absolute = urljoin(MEDLINEPLUS_LAB_TESTS_INDEX, href)
            if absolute not in links:
                links.append(absolute)
    for idx, link in enumerate(links[:max_pages]):
        page = requests.get(link, timeout=60)
        page.raise_for_status()
        slug = link.rstrip("/").split("/")[-1]
        (target_dir / f"{idx:04d}_{slug}.html").write_text(page.text, encoding="utf-8")


def _download_openfda_labels(root: Path, requests, limit: int) -> None:
    target_dir = root / "openfda_drug_labels"
    target_dir.mkdir(parents=True, exist_ok=True)
    for drug in DEFAULT_DRUGS[:limit]:
        params = {
            "search": f'openfda.generic_name:"{drug}"',
            "limit": 1,
        }
        response = requests.get(OPENFDA_ENDPOINT, params=params, timeout=60)
        if response.status_code != 200:
            continue
        payload = response.json()
        (target_dir / f"{drug}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download open RAG sources for QingNang-ClinicOS")
    parser.add_argument("--root", default="/root/autodl-tmp/medagent/datasets/rag_raw")
    parser.add_argument("--max-lab-tests", type=int, default=200)
    parser.add_argument("--drug-limit", type=int, default=10)
    args = parser.parse_args()

    requests, BeautifulSoup = _require_deps()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    _download_medlineplus_xml(root, requests, BeautifulSoup)
    _download_lab_tests(root, requests, BeautifulSoup, args.max_lab_tests)
    _download_openfda_labels(root, requests, args.drug_limit)
    print(f"[ok] downloaded RAG raw sources into {root}")


if __name__ == "__main__":
    main()
