from __future__ import annotations

import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RAW_DIR / "tsmc_report.txt"
OUTPUT_CSV = PROCESSED_DIR / "tsmc_report_processed.csv"

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(text: str, max_words: int = 200) -> list[str]:
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def normalize_and_tokenize(text: str) -> list[str]:
    text = text.lower()

    # 只保留英文、數字、空白、連字號
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    cleaned_tokens = []
    for token in tokens:
        if token in STOP_WORDS:
            continue
        if len(token) < 2:
            continue
        if token.isdigit():
            continue
        cleaned_tokens.append(token)

    return cleaned_tokens


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    raw_text = INPUT_FILE.read_text(encoding="utf-8")
    cleaned_text = clean_text(raw_text)
    chunks = split_into_chunks(cleaned_text, max_words=200)

    rows = []
    for idx, chunk in enumerate(chunks, start=1):
        tokens = normalize_and_tokenize(chunk)
        processed_text = " ".join(tokens)

        rows.append(
            {
                "chunk_id": idx,
                "raw_text": chunk,
                "processed_text": processed_text,
                "token_count": len(tokens),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved processed data to: {OUTPUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()