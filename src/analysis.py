from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.pipeline import SDG_KEYWORDS

CHUNKS_CSV = Path("outputs/chunks_processed.csv")


# =========================
# 1. KEYWORD FREQUENCY AUDIT
# =========================

def keyword_frequency_audit(df: pd.DataFrame, sdg: str) -> pd.DataFrame:
    """
    For each keyword in an SDG's list, compute:
      - chunk_hits      : how many chunks (out of all 262) contain this keyword
      - chunk_freq      : chunk_hits / total_chunks  (high = too generic, low = never appears)
      - verdict         : TOO GENERIC (>40%), RARE (<3 hits), or OK

    Use this to decide which keywords to remove from SDG_KEYWORDS.
    """
    total = len(df)
    all_text = df["raw_text"].str.lower()
    keywords = SDG_KEYWORDS[sdg]

    records = []
    for kw in keywords:
        hits = int(all_text.str.contains(re.escape(kw.lower()), regex=True).sum())
        freq = hits / total
        if freq > 0.40:
            verdict = "TOO GENERIC — remove"
        elif hits < 3:
            verdict = "RARE / missing from report"
        else:
            verdict = "OK"
        records.append({
            "keyword": kw,
            "chunk_hits": hits,
            "chunk_freq_%": round(freq * 100, 1),
            "verdict": verdict,
        })

    return (
        pd.DataFrame(records)
        .sort_values("chunk_freq_%", ascending=False)
        .reset_index(drop=True)
    )


# =========================
# 2. SENTENCE CO-OCCURRENCE
# =========================

def sentence_cooccurrence(df: pd.DataFrame, sdg: str, top_n: int = 25) -> pd.DataFrame:
    """
    Extract every sentence from chunks tagged with `sdg`.
    Run sentence-level TF-IDF to find words that are truly
    distinctive in the context of this SDG — i.e., what is
    actually being said in those sentences.
    """
    sdg_text = df[df["sdg_labels"].str.contains(sdg, na=False)]["raw_text"]
    other_text = df[~df["sdg_labels"].str.contains(sdg, na=False)]["raw_text"]

    def to_sentences(series):
        sents = []
        for text in series:
            for sent in re.split(r"(?<=[.!?])\s+", str(text)):
                sent = sent.strip()
                if len(sent.split()) >= 6:
                    sents.append(sent)
        return sents

    sdg_sents = to_sentences(sdg_text)
    other_sents = to_sentences(other_text)

    if len(sdg_sents) < 3:
        return pd.DataFrame(columns=["term", "sdg_tfidf", "other_tfidf", "lift"])

    all_sents = sdg_sents + other_sents
    labels = ["sdg"] * len(sdg_sents) + ["other"] * len(other_sents)

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        stop_words="english",
    )
    X = vec.fit_transform(all_sents)
    vocab = vec.get_feature_names_out()

    sdg_idx = [i for i, l in enumerate(labels) if l == "sdg"]
    other_idx = [i for i, l in enumerate(labels) if l == "other"]

    sdg_mean = X[sdg_idx].mean(axis=0).A1
    other_mean = X[other_idx].mean(axis=0).A1 if other_idx else sdg_mean * 0

    # lift = how much more prominent in SDG sentences vs non-SDG sentences
    lift = sdg_mean / (other_mean + 1e-9)

    return (
        pd.DataFrame({
            "term": vocab,
            "sdg_tfidf": sdg_mean.round(4),
            "other_tfidf": other_mean.round(4),
            "lift": lift.round(2),
        })
        .sort_values("lift", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# =========================
# 3. MAIN — DIAGNOSTIC REPORT
# =========================

def main():
    df = pd.read_csv(CHUNKS_CSV)
    total = len(df)
    print(f"Total chunks: {total}\n")

    focus_sdgs = ["SDG9_innovation", "SDG7_energy"]

    for sdg in focus_sdgs:
        print(f"\n{'='*60}")
        print(f"  {sdg}")
        print(f"{'='*60}")

        print("\n--- Keyword frequency audit ---")
        print("(chunk_freq% = how often keyword appears across ALL chunks)")
        audit = keyword_frequency_audit(df, sdg)
        print(audit.to_string(index=False))

        print(f"\n--- Sentence co-occurrence: what {sdg} chunks actually say ---")
        print("(lift = how much more this term appears in this SDG vs other SDGs)")
        cooc = sentence_cooccurrence(df, sdg, top_n=20)
        print(cooc.to_string(index=False))


if __name__ == "__main__":
    main()
