"""Multi-year chunk corpus loader for cross-year Q&A retrieval.

Read-only: loads the existing per-year `chunks_processed.csv` files, tags every
chunk with its source year, and returns a single pooled list. Results are cached
so the three CSVs are read at most once per process.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .preprocess import load_processed_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CROSS_YEARS: tuple[str, ...] = ("2022", "2023", "2024")


@lru_cache(maxsize=None)
def _load_year_records(year: str) -> tuple[dict[str, Any], ...]:
    """Cached per-year records (tuple so it is hashable for lru_cache)."""
    records = load_processed_chunks(year)
    for rec in records:
        rec["year"] = str(year)
    return tuple(records)


def load_multi_year_chunks(years: tuple[str, ...] = CROSS_YEARS) -> list[dict[str, Any]]:
    """Pooled, year-tagged chunk records across the requested years.

    Each record keeps the per-year fields (text, section_label, sdg_labels) and
    adds a `year` tag plus a globally-unique `chunk_id` of the form "<year>#<id>"
    so chunks from different years never collide in the retrieval pool.
    """
    pooled: list[dict[str, Any]] = []
    for year in years:
        for rec in _load_year_records(str(year)):
            item = dict(rec)
            item["year"] = str(year)
            item["chunk_id"] = f"{year}#{rec.get('chunk_id')}"
            pooled.append(item)
    return pooled


def available_years() -> list[str]:
    """Years whose chunks_processed.csv actually exists (read-only check)."""
    found = []
    for year in CROSS_YEARS:
        path = OUTPUTS_DIR / year / "chunks_processed.csv"
        if path.exists():
            found.append(year)
    return found
