from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROCESSED_FILE = Path("data/processed/tsmc_report_processed.csv")
OUTPUT_DIR = Path("outputs/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_tfidf(texts: list[str]):
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),  # unigram + bigram
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    return tfidf_matrix, feature_names, vectorizer


def get_top_keywords(tfidf_matrix, feature_names, top_k=20):
    import numpy as np

    # 平均每個詞的重要性
    scores = tfidf_matrix.mean(axis=0).A1

    top_indices = scores.argsort()[::-1][:top_k]

    keywords = [(feature_names[i], scores[i]) for i in top_indices]
    return keywords


def main() -> None:
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError("Processed file not found. Run preprocess first.")

    df = pd.read_csv(PROCESSED_FILE)

    texts = df["processed_text"].fillna("").tolist()

    tfidf_matrix, feature_names, vectorizer = compute_tfidf(texts)

    keywords = get_top_keywords(tfidf_matrix, feature_names, top_k=30)

    print("\nTop Keywords:\n")
    for word, score in keywords:
        print(f"{word:<20} {score:.4f}")

    # 存 keyword
    keyword_df = pd.DataFrame(keywords, columns=["keyword", "score"])
    keyword_df.to_csv(OUTPUT_DIR / "top_keywords.csv", index=False)

    print("\nSaved to outputs/models/top_keywords.csv")


if __name__ == "__main__":
    main()