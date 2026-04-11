from __future__ import annotations

from pathlib import Path

import pdfplumber


PDF_PATH = Path("data/raw/2024-TSMC-Sustainability-Report-e.pdf")
OUTPUT_TXT = Path("data/raw/tsmc_report.txt")


def extract_pdf_text(pdf_path: Path) -> str:
    texts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if not page_text:
                continue
            texts.append(f"\n--- PAGE {page_number} ---\n")
            texts.append(page_text)
    return "\n".join(texts)


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    text = extract_pdf_text(PDF_PATH)
    OUTPUT_TXT.write_text(text, encoding="utf-8")
    print(f"Saved extracted text to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
