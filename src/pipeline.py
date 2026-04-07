# src/pipeline.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# 0. CONFIG
# =========================

CUSTOM_STOPWORDS = {
    "tsmc", "company", "companies", "report", "reports", "sustainability",
    "annual", "year", "years", "esg", "taiwan", "limited", "co", "ltd",
    "page", "figure", "table", "appendix", "chapter", "section",
    "including", "include", "also", "may", "can", "will", "shall",
    "within", "across", "among", "based", "according", "related", 
    
    "content", "overview", "index", "summary",
    "information", "data", "global", "group",
    "value", "business", "operation"
}

# 可依你們資料再加
ROLE_KEYWORDS = {
    "environment": [
        "climate", "carbon", "emission", "energy", "electricity", "renewable",
        "water", "waste", "recycling", "greenhouse", "biodiversity", "environment"
    ],
    "talent": [
        "employee", "employees", "talent", "training", "learning", "career",
        "workplace", "safety", "health", "well-being", "diversity", "inclusion"
    ],
    "supply_chain": [
        "supplier", "suppliers", "procurement", "supply chain", "vendor",
        "responsible sourcing", "audit", "compliance"
    ],
    "social": [
        "community", "education", "volunteer", "donation", "social", "inclusion",
        "public", "stakeholder", "society", "charity"
    ],
    "governance": [
        "governance", "board", "ethics", "integrity", "compliance", "risk",
        "policy", "management", "oversight", "internal control"
    ]
}

ACTION_WORDS = {
    "reduce", "improve", "enhance", "increase", "develop", "promote", "strengthen",
    "implement", "build", "support", "advance", "drive", "optimize", "expand",
    "achieve", "manage", "mitigate", "protect", 

    "reduce", "reduction", "improvement", "increase",
    "implementation", "development", "operation",
    "operational", "management", "strategy",
    "target", "goal", "plan", "initiative",
    "performance", "efficiency", "control"
}

PEOPLE_WORDS = {
    "employee", "employees", "people", "community", "communities", "customer",
    "customers", "stakeholder", "stakeholders", "supplier", "suppliers",
    "worker", "workers", "society", "public", "talent", 

    "health", "safety", "wellbeing", "well-being",
    "training", "education", "development",
    "diversity", "inclusion", "culture",
    "engagement", "human", "rights"
}

ALLOWED_POS = {"NOUN", "PROPN", "ADJ", "VERB"}


# =========================
# 1. DATA CLASS
# =========================

@dataclass
class ChunkRecord:
    chunk_id: int
    raw_text: str
    clean_text: str
    section_label: str
    orientation: str


# =========================
# 2. NLP LOADER
# =========================


def load_spacy_model(model_name: str = "en_core_web_sm"):
    import spacy
    try:
        return spacy.load(model_name, disable=["parser"])
    except OSError:
        from spacy.cli import download
        print(f"Downloading spaCy model: {model_name}")
        download(model_name)
        return spacy.load(model_name, disable=["parser"])

# =========================
# 3. CLEANUP
# =========================

def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_urls_emails(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    return text


def remove_page_artifacts(text: str) -> str:
    """
    粗清 PDF 噪音:
    - Page 12
    - standalone page numbers
    - repeated dotted TOC fragments
    """
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        s = line.strip()

        # 空行保留
        if not s:
            cleaned.append("")
            continue

        # 單獨頁碼
        if re.fullmatch(r"\d{1,4}", s):
            continue

        # Page 12 / page 12
        if re.fullmatch(r"[Pp]age\s+\d{1,4}", s):
            continue

        # 目錄點點點
        if re.search(r"\.{4,}", s):
            continue

        # 很短且像 header/footer
        if len(s) < 5 and re.search(r"\d", s):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def basic_pdf_cleanup(text: str) -> str:
    text = remove_urls_emails(text)
    text = remove_page_artifacts(text)
    text = normalize_whitespace(text)
    return text


# =========================
# 4. CHUNKING
# =========================

def split_into_paragraphs(text: str, min_len: int = 80) -> List[str]:
    """
    先以空行切段，再把太短的段落過濾掉。
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    paras = [p for p in paras if len(p) >= min_len]
    return paras


def merge_short_paragraphs(paragraphs: List[str], min_words: int = 40) -> List[str]:
    """
    太短的 paragraph 合併到下一段，避免 chunk 太碎。
    """
    merged = []
    buffer = ""

    for para in paragraphs:
        word_count = len(para.split())

        if word_count < min_words:
            buffer = f"{buffer} {para}".strip()
        else:
            if buffer:
                para = f"{buffer} {para}".strip()
                buffer = ""
            merged.append(para)

    if buffer:
        if merged:
            merged[-1] = f"{merged[-1]} {buffer}".strip()
        else:
            merged.append(buffer)

    return merged


# =========================
# 5. RULE-BASED CLASSIFIERS
# =========================

def count_keyword_hits(text: str, keywords: List[str]) -> int:
    text_lower = text.lower()
    score = 0
    for kw in keywords:
        # keyword phrase matching
        if kw in text_lower:
            score += 1
    return score


def classify_section(text: str) -> str:
    scores = {
        role: count_keyword_hits(text, kws)
        for role, kws in ROLE_KEYWORDS.items()
    }
    best_role = max(scores, key=scores.get)

    if scores[best_role] == 0:
        return "other"
    return best_role


def classify_orientation(tokens):
    if not tokens:
        return "mixed"

    action_hits = sum(1 for t in tokens if t in ACTION_WORDS)
    people_hits = sum(1 for t in tokens if t in PEOPLE_WORDS)

    total = len(tokens)

    action_ratio = action_hits / total
    people_ratio = people_hits / total

    if action_ratio > 0.03 and action_ratio > people_ratio:
        return "action_oriented"
    elif people_ratio > 0.02 and people_ratio > action_ratio:
        return "people_centric"
    else:
        return "mixed"


# =========================
# 6. TOKENIZATION / LEMMATIZATION / POS FILTER
# =========================

def preprocess_chunk(
    text: str,
    nlp,
    allowed_pos: set = ALLOWED_POS,
    custom_stopwords: set = CUSTOM_STOPWORDS,
) -> Tuple[str, List[str]]:
    """
    回傳:
    - clean_text: 給 TF-IDF 用
    - kept_tokens: 給其他 rule-based analysis 用
    """
    text = text.lower()

    # 保留字母與數字，其他大致清掉
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp(text)
    tokens = []

    for token in doc:
        lemma = token.lemma_.strip().lower()

        if not lemma:
            continue
        if token.is_stop:
            continue
        if lemma in custom_stopwords:
            continue
        if token.pos_ not in allowed_pos:
            continue
        if len(lemma) <= 2:
            continue
        if lemma.isdigit():
            continue

        tokens.append(lemma)

    clean_text = " ".join(tokens)
    return clean_text, tokens


# =========================
# 7. BIGRAM TF-IDF
# =========================

def build_tfidf(
    texts: List[str],
    min_df: int = 2,
    max_df: float = 0.85,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 3000,
):
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        max_features=max_features
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


def extract_top_terms_per_group(
    df: pd.DataFrame,
    group_col: str,
    text_col: str = "clean_text",
    top_k: int = 30,
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1, 2),
) -> pd.DataFrame:
    """
    把同 group 的 chunk 合併後做 TF-IDF。
    適合:
    - 每個 role 一組 top terms
    - 每個 orientation 一組 top terms
    """
    grouped = (
        df.groupby(group_col)[text_col]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )

    vectorizer = TfidfVectorizer(
        min_df=min_df,
        ngram_range=ngram_range,
        max_df=0.9
    )
    X = vectorizer.fit_transform(grouped[text_col])
    vocab = vectorizer.get_feature_names_out()

    records = []
    for i, row in grouped.iterrows():
        scores = X[i].toarray().flatten()
        top_idx = scores.argsort()[::-1][:top_k]

        for idx in top_idx:
            if scores[idx] <= 0:
                continue
            records.append({
                group_col: row[group_col],
                "term": vocab[idx],
                "tfidf_score": float(scores[idx])
            })

    return pd.DataFrame(records)


# =========================
# 8. FULL PIPELINE
# =========================

def run_pipeline(raw_text: str, nlp) -> pd.DataFrame:
    # 1) cleanup
    cleaned = basic_pdf_cleanup(raw_text)

    # 2) chunk
    paragraphs = split_into_paragraphs(cleaned, min_len=80)
    chunks = merge_short_paragraphs(paragraphs, min_words=40)

    records = []

    for i, chunk in enumerate(chunks):
        section_label = classify_section(chunk)
        clean_text, kept_tokens = preprocess_chunk(chunk, nlp)
        orientation = classify_orientation(kept_tokens)
        if len(clean_text.split()) < 20:
            continue
        
        if "content" in clean_text and "index" in clean_text:
            continue
        
        if "content" in chunk.lower() and "overview" in chunk.lower():
            continue
        
        records.append(
            ChunkRecord(
                chunk_id=i,
                raw_text=chunk,
                clean_text=clean_text,
                section_label=section_label,
                orientation=orientation
            ).__dict__
        )

    df = pd.DataFrame(records)

    # clean_text 空的要去掉
    df = df[df["clean_text"].str.strip().astype(bool)].reset_index(drop=True)
    return df


# =========================
# 9. EXPORT HELPERS
# =========================

def save_outputs(
    df_chunks: pd.DataFrame,
    output_dir: str = "outputs"
):
    import os
    os.makedirs(output_dir, exist_ok=True)

    df_chunks.to_csv(f"{output_dir}/chunks_processed.csv", index=False)

    # top terms by section
    df_section_terms = extract_top_terms_per_group(
        df_chunks, group_col="section_label", top_k=30
    )
    df_section_terms.to_csv(f"{output_dir}/tfidf_by_section.csv", index=False)

    # top terms by orientation
    df_orientation_terms = extract_top_terms_per_group(
        df_chunks, group_col="orientation", top_k=30
    )
    df_orientation_terms.to_csv(f"{output_dir}/tfidf_by_orientation.csv", index=False)

    return {
        "chunks": df_chunks,
        "section_terms": df_section_terms,
        "orientation_terms": df_orientation_terms
    }