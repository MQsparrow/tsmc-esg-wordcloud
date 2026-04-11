from __future__ import annotations

import csv
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


URL = "https://www.tsmc.com/english/aboutTSMC/dc_csr_report"
OUTPUT_DIR = Path("data/raw")
LINKS_CSV = OUTPUT_DIR / "tsmc_esg_download_links.csv"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


def fetch_html(url: str) -> str:
    headers = {
        **REQUEST_HEADERS,
        "Referer": "https://www.tsmc.com/",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def parse_download_links(html: str, base_url: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, str]] = []

    for anchor in soup.find_all("a", href=True):
        label = anchor.get_text(" ", strip=True)
        href = anchor["href"].strip()

        if not label:
            continue
        if "Download" not in label and "Report" not in label:
            continue

        rows.append(
            {
                "label": label,
                "url": urljoin(base_url, href),
            }
        )

    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for row in rows:
        key = (row["label"], row["url"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped


def save_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["label", "url"])
        writer.writeheader()
        writer.writerows(rows)


def sanitize_filename(label: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", label).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:150] or "download"


def download_file(url: str, output_path: Path) -> None:
    headers = {
        **REQUEST_HEADERS,
        "Referer": "https://esg.tsmc.com/",
    }
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    print(f"Downloaded: {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    html = fetch_html(URL)
    rows = parse_download_links(html, URL)
    save_csv(rows, LINKS_CSV)

    print(f"Saved {len(rows)} rows to {LINKS_CSV}")

    pdf_rows = [row for row in rows if row["url"].lower().endswith(".pdf")]
    print(f"\nFound {len(pdf_rows)} PDF files\n")

    for row in pdf_rows:
        output_path = OUTPUT_DIR / f"{sanitize_filename(row['label'])}.pdf"
        try:
            download_file(row["url"], output_path)
        except Exception as exc:
            print(f"Failed: {row['url']} | {exc}")


if __name__ == "__main__":
    main()
