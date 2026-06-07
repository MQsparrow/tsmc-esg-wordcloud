from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .corpus import CROSS_YEARS, load_multi_year_chunks
from .state import ESGState


def retrieve_records(records: list[dict[str, Any]], question: str, top_k: int = 4) -> list[dict[str, Any]]:
    """TF-IDF cosine retrieval over year-tagged chunk records.

    Each record must have a "text" field; "year" and "chunk_id" are carried through
    so the answer layer can tell which year each piece of evidence came from.
    """
    records = [r for r in records if str(r.get("text", "")).strip()]
    if not records or not question.strip():
        return []
    texts = [str(r["text"]) for r in records]
    corpus = texts + [question]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(corpus)
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
    except Exception:
        q_terms = {part.lower() for part in question.split() if len(part) > 2}
        scored = [
            (idx, float(sum(1 for term in q_terms if term in text.lower())))
            for idx, text in enumerate(texts)
        ]
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

    results = [
        {
            "chunk_id": records[idx].get("chunk_id", idx),
            "year": str(records[idx].get("year", "")),
            "score": round(float(score), 4),
            "text": texts[idx][:1200],
        }
        for idx, score in ranked
    ]
    non_zero = [item for item in results if item["score"] > 0]
    return non_zero or results[: min(top_k, len(results))]


def _single_year_records(state: ESGState) -> list[dict[str, Any]]:
    """Build year-tagged records from the already-loaded single-year state."""
    year = str(state.get("year", ""))
    records = state.get("chunk_records") or []
    if records:
        return [{**r, "year": str(r.get("year", year) or year)} for r in records]
    return [
        {"chunk_id": idx, "text": text, "year": year}
        for idx, text in enumerate(state.get("chunks", []) or [])
    ]


def retrieval_agent(state: ESGState) -> ESGState:
    question = state.get("user_question", "")
    scope = str(state.get("qa_scope", "cross_year"))
    if scope == "cross_year":
        years = tuple(state.get("qa_years") or CROSS_YEARS)
        records = load_multi_year_chunks(years)
    else:
        records = _single_year_records(state)
    retrieved = retrieve_records(records, question, top_k=4)
    return {"retrieved_chunks": retrieved}
