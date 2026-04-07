from __future__ import annotations

import csv
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from pathlib import Path
import requests
import pandas as pd


URL = "https://www.tsmc.com/english/aboutTSMC/dc_csr_report"
LINKS_CSV = Path("data/raw/tsmc_esg_download_links.csv")
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.tsmc.com/",
        "Connection": "keep-alive",
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def parse_download_links(html: str, base_url: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, str]] = []

    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        href = a["href"].strip()

        if not text:
            continue

        if "Download" not in text and "Report" not in text:
            continue

        full_url = urljoin(base_url, href)

        rows.append({
            "label": text,
            "url": full_url
        })

    # dedupe
    seen = set()
    deduped = []
    for row in rows:
        key = (row["label"], row["url"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    return deduped


def save_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "url"])
        writer.writeheader()
        writer.writerows(rows)

def download_file(url: str, output_path: Path) -> None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://esg.tsmc.com/",
    }

    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"Downloaded: {output_path}")


def main() -> None:
    html = fetch_html(URL)
    rows = parse_download_links(html, URL)

    output_csv = OUTPUT_DIR / "tsmc_esg_download_links.csv"
    save_csv(rows, output_csv)

    print(f"Saved {len(rows)} rows to {output_csv}")

    # 只保留 PDF
    pdf_rows = [r for r in rows if r["url"].endswith(".pdf")]

    print(f"\nFound {len(pdf_rows)} PDF files\n")

    for row in pdf_rows:
        url = row["url"]

        # 用 label 當檔名（清理一下）
        filename = row["label"].replace(" ", "_").replace("/", "_") + ".pdf"
        output_path = OUTPUT_DIR / filename

        try:
            download_file(url, output_path)
        except Exception as e:
            print(f"Failed: {url} | {e}")

if __name__ == "__main__":
    main()