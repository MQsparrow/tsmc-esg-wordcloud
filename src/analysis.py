from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.pipeline import SDG_KEYWORDS


CHUNKS_CSV = Path("outputs/chunks_processed.csv")
FOCUS_SDGS = ["SDG9_innovation", "SDG7_energy"]


def parse_sdg_labels(label_text: object) -> list[str]:
    text = "" if pd.isna(label_text) else str(label_text).strip()
    if not text:
        return []
    return [label.strip() for label in text.split(",") if label.strip()]


def keyword_frequency_audit(df: pd.DataFrame, sdg: str) -> pd.DataFrame:
    """
    For each keyword in an SDG keyword list, compute:
      - chunk_hits: number of chunks containing the keyword
      - chunk_freq_%: share of chunks containing the keyword
      - verdict: whether the keyword looks too generic, too rare, or acceptable
    """
    total = len(df)
    all_text = df["raw_text"].fillna("").str.lower()
    keywords = SDG_KEYWORDS[sdg]

    records = []
    for keyword in keywords:
        hits = int(all_text.str.contains(re.escape(keyword.lower()), regex=True).sum())
        freq = hits / total if total else 0.0
        if freq > 0.40:
            verdict = "TOO GENERIC - remove"
        elif hits < 3:
            verdict = "RARE / missing from report"
        else:
            verdict = "OK"

        records.append(
            {
                "keyword": keyword,
                "chunk_hits": hits,
                "chunk_freq_%": round(freq * 100, 1),
                "verdict": verdict,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values("chunk_freq_%", ascending=False)
        .reset_index(drop=True)
    )


def sentence_cooccurrence(df: pd.DataFrame, sdg: str, top_n: int = 25) -> pd.DataFrame:
    """
    Run sentence-level TF-IDF on chunks associated with a target SDG versus all others.
    This helps inspect what those chunks are actually saying beyond the keyword rules.
    """
    label_lists = df["sdg_labels"].map(parse_sdg_labels)
    sdg_mask = label_lists.map(lambda labels: sdg in labels)
    sdg_text = df.loc[sdg_mask, "raw_text"]
    other_text = df.loc[~sdg_mask, "raw_text"]

    def to_sentences(series: pd.Series) -> list[str]:
        sentences: list[str] = []
        for text in series.fillna("").astype(str):
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                sentence = sentence.strip()
                if len(sentence.split()) >= 6:
                    sentences.append(sentence)
        return sentences

    sdg_sentences = to_sentences(sdg_text)
    other_sentences = to_sentences(other_text)

    if len(sdg_sentences) < 3:
        return pd.DataFrame(columns=["term", "sdg_tfidf", "other_tfidf", "lift"])

    all_sentences = sdg_sentences + other_sentences
    labels = ["sdg"] * len(sdg_sentences) + ["other"] * len(other_sentences)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(all_sentences)
    vocab = vectorizer.get_feature_names_out()

    sdg_idx = [idx for idx, label in enumerate(labels) if label == "sdg"]
    other_idx = [idx for idx, label in enumerate(labels) if label == "other"]

    sdg_mean = matrix[sdg_idx].mean(axis=0).A1
    other_mean = matrix[other_idx].mean(axis=0).A1 if other_idx else sdg_mean * 0
    lift = sdg_mean / (other_mean + 1e-9)

    return (
        pd.DataFrame(
            {
                "term": vocab,
                "sdg_tfidf": sdg_mean.round(4),
                "other_tfidf": other_mean.round(4),
                "lift": lift.round(2),
            }
        )
        .sort_values("lift", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def main() -> None:
    if not CHUNKS_CSV.exists():
        raise FileNotFoundError(f"Chunk file not found: {CHUNKS_CSV}")

    df = pd.read_csv(CHUNKS_CSV)
    print(f"Total chunks: {len(df)}\n")

    for sdg in FOCUS_SDGS:
        print(f"\n{'=' * 60}")
        print(f"  {sdg}")
        print(f"{'=' * 60}")

        print("\n--- Keyword frequency audit ---")
        print("(chunk_freq% = how often a keyword appears across all chunks)")
        audit = keyword_frequency_audit(df, sdg)
        print(audit.to_string(index=False))

        print(f"\n--- Sentence co-occurrence: what {sdg} chunks actually say ---")
        print("(lift = how much more a term appears in this SDG vs other SDGs)")
        cooccurrence = sentence_cooccurrence(df, sdg, top_n=20)
        print(cooccurrence.to_string(index=False))


if __name__ == "__main__":
    main()
