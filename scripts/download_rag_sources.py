#!/usr/bin/env python
"""
Download diagnosis-oriented medical sources for RAG.

This step is CPU/network-bound and can be done before renting GPUs.
It supports:
1. Structured official sources for symptom/drug grounding
2. Chinese HF corpora via hf-mirror
3. Chinese MSD Manuals professional pages via sitemap filtering
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urljoin
import xml.etree.ElementTree as ET


MEDLINEPLUS_XML_PAGE = "https://medlineplus.gov/xml.html"
OPENFDA_ENDPOINT = "https://api.fda.gov/drug/label.json"
MSD_MANUALS_CN_SITEMAP = "https://www.msdmanuals.cn/sitemap.xml"

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

MSD_SPECIALTY_RULES = {
    "diagnostics": (
        "special-subjects",
        "resources/normal-laboratory-values",
        "pages-with-widgets/procedures-and-exams",
    ),
    "internal_medicine": (
        "cardiology",
        "pulmonology",
        "gastroenterology",
        "hematology",
        "endocrinology",
        "urology",
        "infectious-disease",
        "critical-care-medicine",
        "hepatology",
        "clinical-pharmacology",
        "allergy-and-immunology",
        "nutrition",
    ),
    "surgery": (
        "injuries-poisoning",
        "rheumatology-and-orthopedics",
        "otolaryngology",
        "ophthalmology",
        "dentistry",
    ),
    "gynecology": ("gynecology-and-obstetrics",),
    "pediatrics": ("pediatrics",),
    "neurology": ("neurology",),
}


def _require_deps():
    try:
        import requests
        from bs4 import BeautifulSoup
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "Missing RAG download deps. Install with: "
            "pip install requests beautifulsoup4 lxml huggingface_hub"
        ) from exc
    return requests, BeautifulSoup, snapshot_download


def _html_soup(BeautifulSoup, text: str):
    for parser in ("lxml", "html.parser"):
        try:
            return BeautifulSoup(text, parser)
        except Exception:
            continue
    raise RuntimeError("Could not initialize an HTML parser for BeautifulSoup.")


def _repair_mojibake_url(url: str) -> str:
    if "Ã" not in url and "Â" not in url:
        return url
    try:
        return url.encode("latin-1").decode("utf-8")
    except Exception:
        return url


def _download_medlineplus_xml(root: Path, requests, BeautifulSoup) -> None:
    print("[rag] downloading MedlinePlus health topics xml")
    response = requests.get(MEDLINEPLUS_XML_PAGE, timeout=60)
    response.raise_for_status()
    soup = _html_soup(BeautifulSoup, response.text)
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


def _download_openfda_labels(root: Path, requests, limit: int) -> None:
    print(f"[rag] downloading openFDA labels, limit={limit}")
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


def _download_seed_pages(root: Path, requests, manifest_path: Path) -> None:
    print(f"[rag] downloading seed pages from {manifest_path}")
    target_dir = root / "seed_pages"
    target_dir.mkdir(parents=True, exist_ok=True)
    seeds = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in seeds:
        source_id = item["source_id"]
        url = item["url"]
        response = requests.get(
            url,
            timeout=90,
            headers={"User-Agent": "QingNang-ClinicOS/0.1 (+research rag builder)"},
        )
        response.raise_for_status()
        (target_dir / f"{source_id}.html").write_text(response.text, encoding="utf-8")
        (target_dir / f"{source_id}.meta.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _iter_msd_professional_urls(requests, max_pages: int) -> list[dict[str, str]]:
    response = requests.get(
        MSD_MANUALS_CN_SITEMAP,
        timeout=120,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    root = ET.fromstring(response.text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [loc.text for loc in root.findall(".//sm:loc", ns) if loc.text]

    candidates: list[dict[str, str]] = []
    for url in urls:
        if not url.startswith("https://www.msdmanuals.cn/professional/"):
            continue
        if any(fragment in url for fragment in ("/multimedia/", "/authors/", "/content/")):
            continue

        path = url.split("https://www.msdmanuals.cn/professional/", 1)[-1].strip("/")
        if not path:
            continue

        specialty = ""
        for name, prefixes in MSD_SPECIALTY_RULES.items():
            if any(path.startswith(prefix) for prefix in prefixes):
                specialty = name
                break
        if not specialty:
            continue

        candidates.append(
            {
                "url": _repair_mojibake_url(url),
                "specialty": specialty,
                "path": path,
            }
        )
    return candidates[:max_pages]


def _download_msd_manual_cn(root: Path, requests, max_pages: int) -> None:
    target_dir = root / "msd_manual_cn_professional"
    target_dir.mkdir(parents=True, exist_ok=True)
    candidates = _iter_msd_professional_urls(requests, max_pages=max_pages)
    downloaded = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    print(f"[rag] downloading MSD CN pages, candidates={len(candidates)}")
    for idx, item in enumerate(candidates):
        stem = f"{idx:05d}"
        html_path = target_dir / f"{stem}.html"
        meta_path = target_dir / f"{stem}.meta.json"
        if html_path.exists() and meta_path.exists():
            skipped += 1
            continue
        try:
            response = requests.get(
                item["url"],
                timeout=90,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()
            html_path.write_text(response.text, encoding="utf-8")
            meta_path.write_text(
                json.dumps(
                    {
                        "source_id": f"msd_cn:{item['path']}",
                        "source_type": "clinical_reference_cn",
                        "title": item["path"].split("/")[-1].replace("-", " "),
                        "url": item["url"],
                        "specialty": item["specialty"],
                        "tags": ["msd_cn_manual", item["specialty"]],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            downloaded += 1
            if downloaded % 50 == 0:
                print(f"[rag] MSD downloaded={downloaded}, skipped={skipped}, failures={len(failures)}")
        except Exception as exc:
            failures.append({"url": item["url"], "error": str(exc)})
            if len(failures) <= 10:
                print(f"[rag][warn] skip MSD page: {item['url']} -> {exc}")
            continue
    (target_dir / "_summary.json").write_text(
        json.dumps(
            {
                "downloaded": downloaded,
                "skipped_existing": skipped,
                "failures": len(failures),
                "sample_failures": failures[:20],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[rag] MSD CN complete: downloaded={downloaded}, skipped_existing={skipped}, failures={len(failures)}"
    )


def _download_hf_corpora(root: Path, snapshot_download, manifest_path: Path, endpoint: str) -> None:
    target_root = root / "hf_corpora"
    target_root.mkdir(parents=True, exist_ok=True)
    sources = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[dict[str, str]] = []
    for item in sources:
        local_name = item.get("local_name") or item["repo_id"].replace("/", "__")
        repo_dir = target_root / local_name
        print(f"[rag] downloading HF corpus {item['repo_id']} -> {repo_dir.name}")
        try:
            snapshot_download(
                repo_id=item["repo_id"],
                repo_type=item.get("repo_type", "dataset"),
                local_dir=repo_dir,
                allow_patterns=item.get("allow_patterns"),
                endpoint=endpoint,
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=int(item.get("max_workers", 4)),
            )
            (repo_dir / "source_meta.json").write_text(
                json.dumps(item, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            failures.append({"repo_id": item["repo_id"], "error": str(exc)})
            print(f"[rag][warn] HF corpus failed, continue: {item['repo_id']} -> {exc}")
            continue
    (target_root / "_summary.json").write_text(
        json.dumps(
            {
                "completed": len(sources) - len(failures),
                "failures": len(failures),
                "failed_items": failures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[rag] HF corpora stage complete: completed={len(sources) - len(failures)}, failures={len(failures)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RAG sources for QingNang-ClinicOS")
    parser.add_argument("--root", default="/root/autodl-tmp/medagent/datasets/rag_raw")
    parser.add_argument("--drug-limit", type=int, default=10)
    parser.add_argument("--seed-manifest", default="configs/rag_seed_manifest.json")
    parser.add_argument("--with-cn-hf", action="store_true")
    parser.add_argument("--hf-manifest", default="configs/cn_rag_hf_manifest.json")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--with-msd-cn", action="store_true")
    parser.add_argument("--msd-max-pages", type=int, default=1200)
    parser.add_argument("--only-cn", action="store_true")
    args = parser.parse_args()

    requests, BeautifulSoup, snapshot_download = _require_deps()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"[rag] target root={root}")

    if not args.only_cn:
        _download_medlineplus_xml(root, requests, BeautifulSoup)
        _download_openfda_labels(root, requests, args.drug_limit)

    manifest_path = Path(args.seed_manifest)
    if manifest_path.exists() and not args.only_cn:
        _download_seed_pages(root, requests, manifest_path)

    if args.with_cn_hf:
        _download_hf_corpora(root, snapshot_download, Path(args.hf_manifest), args.hf_endpoint)

    if args.with_msd_cn:
        _download_msd_manual_cn(root, requests, max_pages=args.msd_max_pages)

    print(f"[ok] downloaded RAG raw sources into {root}")


if __name__ == "__main__":
    main()
