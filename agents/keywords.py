from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .state import ESGState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "has", "have",
    "will", "our", "its", "their", "about", "into", "more", "also", "than", "such", "through",
    "tsmc", "report", "annual", "sustainability", "page", "table", "figure", "chapter",
}


def _load_existing_tfidf(year: str, top_n: int) -> list[dict[str, Any]]:
    path = OUTPUTS_DIR / str(year) / "tfidf_by_section.csv"
    if not path.exists():
        path = OUTPUTS_DIR / "tfidf_by_section.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if not {"term", "tfidf_score"}.issubset(df.columns):
        return []
    group_col = "section_label" if "section_label" in df.columns else None
    rows = df.sort_values("tfidf_score", ascending=False).head(top_n)
    result = []
    for _, row in rows.iterrows():
        result.append(
            {
                "term": str(row["term"]),
                "score": float(row["tfidf_score"]),
                "frequency": None,
                "group": str(row.get(group_col, "")) if group_col else "",
            }
        )
    return result


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    return [token.strip("-") for token in tokens if token not in STOPWORDS and len(token) > 2]


def extract_keywords_from_chunks(chunks: list[str], top_n: int = 40) -> list[dict[str, Any]]:
    if not chunks:
        return []
    try:
        vectorizer = TfidfVectorizer(
            stop_words=list(STOPWORDS),
            ngram_range=(1, 2),
            min_df=1,
            max_features=500,
        )
        matrix = vectorizer.fit_transform(chunks)
        scores = matrix.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        ranked = sorted(zip(terms, scores), key=lambda item: item[1], reverse=True)[:top_n]
        frequencies = Counter(_tokenize(" ".join(chunks)))
        return [
            {
                "term": term,
                "score": float(score),
                "frequency": int(frequencies.get(term, 0)),
                "group": "",
            }
            for term, score in ranked
        ]
    except Exception:
        frequencies = Counter(_tokenize(" ".join(chunks)))
        return [
            {"term": term, "score": float(count), "frequency": int(count), "group": ""}
            for term, count in frequencies.most_common(top_n)
        ]


def keyword_agent(state: ESGState) -> ESGState:
    top_n = int(state.get("chart_data", {}).get("top_n", 40)) if isinstance(state.get("chart_data"), dict) else 40
    year = str(state.get("year", "2024"))
    keywords = _load_existing_tfidf(year, top_n)
    if not keywords:
        keywords = extract_keywords_from_chunks(state.get("chunks", []), top_n=top_n)
    return {"keywords": keywords}
