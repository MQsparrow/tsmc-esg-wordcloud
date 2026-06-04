from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .state import ESGState


def retrieve_chunks(chunks: list[str], question: str, top_k: int = 4) -> list[dict]:
    if not chunks or not question.strip():
        return []
    corpus = chunks + [question]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(corpus)
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
    except Exception:
        q_terms = {part.lower() for part in question.split() if len(part) > 2}
        ranked = []
        for idx, chunk in enumerate(chunks):
            score = sum(1 for term in q_terms if term in chunk.lower())
            ranked.append((idx, float(score)))
        ranked = sorted(ranked, key=lambda item: item[1], reverse=True)[:top_k]
    results = [
        {"chunk_id": idx, "score": round(float(score), 4), "text": chunks[idx][:1200]}
        for idx, score in ranked
    ]
    non_zero = [item for item in results if item["score"] > 0]
    return non_zero or results[: min(top_k, len(results))]


def retrieval_agent(state: ESGState) -> ESGState:
    question = state.get("user_question", "")
    retrieved = retrieve_chunks(state.get("chunks", []), question, top_k=4)
    return {"retrieved_chunks": retrieved}
