from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .state import ESGState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def clean_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\x00", " ")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 1200, min_chars: int = 180) -> list[str]:
    cleaned = clean_text(text)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{paragraph}".strip()
            continue
        if len(buffer) >= min_chars:
            chunks.append(buffer)
            buffer = paragraph
        else:
            combined = f"{buffer}\n\n{paragraph}".strip()
            while len(combined) > max_chars:
                chunks.append(combined[:max_chars].strip())
                combined = combined[max_chars:].strip()
            buffer = combined
    if buffer:
        chunks.append(buffer)
    return chunks or ([cleaned] if cleaned else [])


def load_processed_chunks(year: str = "2024") -> list[dict[str, Any]]:
    path = OUTPUTS_DIR / str(year) / "chunks_processed.csv"
    if not path.exists():
        path = OUTPUTS_DIR / "chunks_processed.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    preferred_cols = ["raw_text", "chunk_text", "text", "clean_text"]
    text_col = next((col for col in preferred_cols if col in df.columns), None)
    if text_col is None:
        text_candidates = [col for col in df.columns if "text" in col.lower() or "chunk" in col.lower()]
        text_col = text_candidates[0] if text_candidates else df.columns[0]
    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        text = clean_text(str(row.get(text_col, "")))
        if not text:
            continue
        records.append(
            {
                "chunk_id": int(idx),
                "text": text,
                "section_label": row.get("section_label", ""),
                "sdg_labels": row.get("sdg_labels", ""),
                "year": str(year),
            }
        )
    return records


def make_chunk_records(chunks: list[str], year: str = "custom") -> list[dict[str, Any]]:
    return [{"chunk_id": idx, "text": chunk, "section_label": "", "sdg_labels": "", "year": year} for idx, chunk in enumerate(chunks)]


def preprocess_agent(state: ESGState) -> ESGState:
    year = str(state.get("year", "2024"))
    raw_text = state.get("raw_text", "")

    if raw_text.strip():
        cleaned = clean_text(raw_text)
        chunks = chunk_text(cleaned)
        return {
            "cleaned_text": cleaned,
            "chunks": chunks,
            "chunk_records": make_chunk_records(chunks, year=year),
            "source": state.get("source", "uploaded text"),
        }

    records = load_processed_chunks(year)
    chunks = [record["text"] for record in records]
    return {
        "cleaned_text": "\n\n".join(chunks[:300]),
        "chunks": chunks,
        "chunk_records": records,
        "source": f"processed outputs/{year}/chunks_processed.csv",
    }
