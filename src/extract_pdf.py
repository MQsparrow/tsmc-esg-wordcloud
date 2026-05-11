from __future__ import annotations

from pathlib import Path

import pdfplumber


DATA_RAW = Path("data/raw")

# Map PDF filenames to years
PDF_MAPPINGS = {
    "2022": "e-all_2022.pdf",
    "2023": "e-all_2023.pdf",
    "2024": "2024-TSMC-Sustainability-Report-e.pdf",
}


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


def extract_all_pdfs() -> dict[str, Path]:
    """
    Extract all TSMC reports from 2022, 2023, 2024.
    Returns mapping of year -> output text file path.
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    results = {}
    
    for year, pdf_filename in PDF_MAPPINGS.items():
        pdf_path = DATA_RAW / pdf_filename
        if not pdf_path.exists():
            print(f"Warning: {pdf_filename} not found, skipping year {year}")
            continue
        
        output_txt = DATA_RAW / f"tsmc_report_{year}.txt"
        print(f"Extracting {year}: {pdf_path} → {output_txt}")
        
        text = extract_pdf_text(pdf_path)
        output_txt.write_text(text, encoding="utf-8")
        results[year] = output_txt
        print(f"  ✓ Saved {len(text)} chars to {output_txt}")
    
    return results


def main() -> None:
    results = extract_all_pdfs()
    if not results:
        raise FileNotFoundError("No PDF files found in data/raw/")
    
    print(f"\nSuccessfully extracted {len(results)} reports:")
    for year, path in sorted(results.items()):
        print(f"  {year}: {path}")


if __name__ == "__main__":
    main()
