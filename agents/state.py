from __future__ import annotations

from typing import Any, TypedDict


class ESGState(TypedDict, total=False):
    raw_text: str
    cleaned_text: str
    chunks: list[str]
    chunk_records: list[dict[str, Any]]
    keywords: list[dict[str, Any]]
    esg_classifications: list[dict[str, Any]]
    esg_counts: dict[str, int]
    retrieved_chunks: list[dict[str, Any]]
    user_question: str
    qa_answer: str
    summary: str
    chart_data: dict[str, Any]
    source: str
    year: str
    mode: str
    errors: list[str]
