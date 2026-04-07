from pathlib import Path
import pdfplumber

PDF_PATH = Path("data/raw/2024-TSMC-Sustainability-Report-e.pdf")
OUTPUT_TXT = Path("data/raw/tsmc_report.txt")


def extract_pdf_text(pdf_path: Path) -> str:
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                texts.append(f"\n--- PAGE {i} ---\n")
                texts.append(page_text)
    return "\n".join(texts)


def main() -> None:
    text = extract_pdf_text(PDF_PATH)
    OUTPUT_TXT.write_text(text, encoding="utf-8")
    print(f"Saved extracted text to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()