# src/pipeline.py

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import networkx as nx
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations


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
    "value", "business", "operation",

    # generic corporate / noise terms
    "secret", "manager", "opera", "chairperson", "engineer",
    "wet", "nature", "ton", "metric ton", "scrubber",

    # TSMC / semiconductor industry — too obvious to be informative
    "fab", "semiconductor",

    # contact / address noise from PDF appendix
    "tel", "fax", "hsinchu", "science park", "eda", "iw",
    "currently serve", "currently serves",
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

ISSUE_FRAME_KEYWORDS = {
    "environment": [
        "climate", "carbon", "emission", "energy", "electricity", "renewable",
        "water", "waste", "recycling", "pollution", "wastewater", "resource",
        "conservation", "environment", "greenhouse", "decarbonization"
    ],
    "labor_talent": [
        "employee", "employees", "talent", "training", "learning", "career",
        "workplace", "safety", "health", "well-being", "wellbeing", "diversity",
        "inclusion", "human rights", "worker", "workers", "workforce", "recruitment"
    ],
    "supply_chain": [
        "supplier", "suppliers", "procurement", "supply chain", "vendor",
        "responsible sourcing", "audit", "compliance", "raw material", "material risk",
        "third party", "contractor"
    ],
    "innovation": [
        "patent", "trade secret", "intellectual property", "research", "development",
        "technology", "innovation", "process node", "automation", "digitalization",
        "artificial intelligence", "smart manufacturing", "breakthrough", "prototype"
    ],
    "governance_risk": [
        "governance", "board", "ethics", "integrity", "compliance", "risk",
        "policy", "management", "oversight", "internal control", "tax",
        "committee", "security", "regulation", "accountability"
    ],
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

# I added SDG keywords based on UN targets, GRI standards, and SASB topics. 
SDG_KEYWORDS = {

    "SDG3_health": [
        # UN target: ensure healthy lives, promote well-being
        # GRI: Occupational Health and Safety (GRI 403)
        # SASB: Employee Health & Safety
        "health", "safety", "occupational", "injury", "illness", "disease",
        "mental health", "wellbeing", "well-being", "medical", "ergonomic",
        "accident", "incident rate", "lost time", "fatality", "hazard",
        "health promotion", "employee assistance", "work-related",
        "psychological safety", "stress", "burnout"
    ],

    "SDG4_education": [
        # UN target: quality education, lifelong learning
        # GRI: Training and Education (GRI 404)
        # SASB: Workforce Development
        "training", "education", "learning", "skill", "competency",
        "upskilling", "reskilling", "career development", "scholarship",
        "internship", "apprenticeship", "STEAM", "university", "academic",
        "tuition", "e-learning", "knowledge transfer", "certification",
        "professional development", "talent pipeline", "mentoring"
    ],

    "SDG6_water": [
        # UN target: clean water, sanitation, water efficiency
        # GRI: Water and Effluents (GRI 303)
        # SASB: Water Management (semiconductor 高耗水產業重點)
        "water", "wastewater", "effluent", "water recycling", "water reuse",
        "water consumption", "water intensity", "water withdrawal",
        "water stewardship", "water stress", "groundwater", "discharge",
        "water treatment", "reclaimed water", "water reduction",
        "water footprint", "water risk", "sanitation"
    ],

    "SDG7_energy": [
        # UN target: affordable clean energy, energy efficiency, renewables
        # GRI: Energy (GRI 302)
        # SASB: Energy Management
        "energy", "electricity", "renewable", "solar", "wind",
        "energy consumption", "energy intensity", "energy efficiency",
        "clean energy", "power purchase agreement", "PPA", "RE100",
        "carbon free", "low carbon electricity", "energy reduction",
        "kilowatt", "megawatt", "energy mix", "fossil fuel"
    ],

    "SDG8_labor": [
        # UN target: decent work, economic growth, labor rights
        # GRI: Employment (GRI 401), Forced Labor (GRI 409), Child Labor (GRI 408)
        # SASB: Labor Practices
        "decent work", "labor", "labour", "employment", "wage", "compensation",
        "fair pay", "working hours", "overtime", "forced labor", "child labor",
        "human rights", "freedom of association", "collective bargaining",
        "worker", "workforce", "job creation", "economic growth",
        "living wage", "pay equity", "gender pay gap", "labor standard",
        "supply chain labor", "modern slavery"
    ],

    "SDG9_innovation": [
        # UN target: industry, innovation, infrastructure, R&D
        # GRI: (indirect, often reported as economic contribution)
        # SASB: Intellectual Property, Process Innovation
        "R&D", "patent", "trade secret",
        "infrastructure",
        "process node", "automation",
        "digitalization", "artificial intelligence", "smart manufacturing",
        "capital expenditure", "investment", "breakthrough", "prototype"
    ],

    "SDG12_consumption": [
        # UN target: responsible consumption and production, waste, chemicals
        # GRI: Waste (GRI 306), Materials (GRI 301), Supplier Environmental Assessment (GRI 308)
        # SASB: Hazardous Waste, Supply Chain Management
        "waste", "hazardous waste", "recycling", "circular economy",
        "chemical", "substance", "material", "packaging",
        "responsible sourcing", "supplier assessment", "procurement",
        "product stewardship", "end of life", "reuse", "reduce",
        "toxic", "restriction", "RoHS", "REACH", "chemical management",
        "waste reduction", "zero waste", "resource efficiency"
    ],

    "SDG13_climate": [
        # UN target: climate action, GHG reduction, resilience
        # GRI: Emissions (GRI 305)
        # SASB: GHG Emissions, Climate Risk
        "climate", "climate change", "carbon", "emission", "greenhouse gas",
        "GHG", "scope 1", "scope 2", "scope 3", "net zero", "carbon neutral",
        "carbon reduction", "decarbonization", "carbon footprint",
        "climate risk", "physical risk", "transition risk", "TCFD",
        "Paris Agreement", "1.5 degree", "carbon offset", "carbon credit",
        "climate resilience", "adaptation", "mitigation"
    ],

    "SDG17_partnership": [
        # UN target: partnerships, multi-stakeholder, finance, data
        # GRI: Stakeholder Engagement (GRI 2-29), Public Policy (GRI 415)
        # SASB: (cross-cutting)
        "partnership", "collaboration", "stakeholder", "engagement",
        "multi-stakeholder", "public private", "industry association",
        "coalition", "alliance", "initiative", "pledge", "commitment",
        "transparency", "disclosure", "reporting framework", "GRI", "SASB",
        "TCFD", "UN Global Compact", "standard", "certification",
        "third party", "assurance", "audit", "accountability"
    ],
}

ALLOWED_POS = {"NOUN", "PROPN", "ADJ", "VERB"}

# Multi-word phrases to protect from stopword splitting.
# They are joined with underscore before tokenization so they survive as one token.
PROTECTED_PHRASES = [
    "trade secret",
]


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
    sdg_labels: str
    sdg_confidence: float
    issue_frame: str
    issue_frame_confidence: float


# =========================
# 2. NLP LOADER
# =========================


def load_spacy_model(model_name: str = "en_core_web_sm"):
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
    skip_until_next_page_marker = False

    for line in lines:
        s = line.strip()

        if s == "Contents":
            skip_until_next_page_marker = True
            continue

        if skip_until_next_page_marker:
            if re.fullmatch(r"--- PAGE \d+ ---", s):
                skip_until_next_page_marker = False
            else:
                continue

        # 空行保留
        if not s:
            cleaned.append("")
            continue

        # 單獨頁碼
        if re.fullmatch(r"--- PAGE \d+ ---", s):
            continue

        if s == "Overview Sustainable Business Practices Operations and Governance Appendix":
            continue
        if s == "2024 Sustainability Report":
            continue
        if s == "An Innovation Pioneer A Responsible Purchaser A Practitioner of Green Power An Admired Employer Power to Change Society":
            continue
        if s.startswith("Please refer to the following instructions as a guide for reading this report"):
            continue
        if s.startswith("Click to proceed to the external hyperlink"):
            continue

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

        # contact info: Tel / Fax lines
        if re.match(r"(tel|fax|phone)[\.\:\s\+]", s, re.IGNORECASE):
            continue

        # phone number patterns (+886, international)
        if re.search(r"\+?\d[\d\s\-]{7,}", s) and len(s.split()) <= 8:
            continue

        # address lines: Science Park / Rd. / No. / zip code patterns
        if re.search(r"\bscience park\b|\brd\.\b|\bno\.\s*\d|\b\d{5,6}\b", s, re.IGNORECASE):
            continue

        # product codes / model numbers (e.g. EM9305, IW612)
        if re.fullmatch(r"[A-Z]{1,3}\d{3,}[\w\-]*", s):
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

def is_noisy_fragment(text: str) -> bool:
    s = " ".join(text.split())
    if not s:
        return True

    lower = s.lower()
    words = s.split()

    if len(words) <= 12:
        if re.fullmatch(r"(overview|appendix|contents?|index)", lower):
            return True
        if re.fullmatch(r"[a-z]", lower):
            return True
        if lower == "(continued from the previous page)":
            return True
        if s in {"●"}:
            return True

    if len(words) <= 30:
        if lower.startswith("please refer to the following instructions"):
            return True
        if lower.startswith("click to proceed to the external hyperlink"):
            return True
        if lower.startswith("click to send feedback"):
            return True
        if lower.startswith("click to return to the table of content"):
            return True

    if len(words) <= 20 and s == s.title():
        return True

    return False


def should_buffer_short_paragraph(text: str, min_words: int) -> bool:
    if is_noisy_fragment(text):
        return False
    return len(text.split()) < min_words


def is_header_like_line(line: str) -> bool:
    s = " ".join(line.split())
    if not s:
        return False

    words = s.split()
    if len(words) > 12:
        return False
    if re.search(r"[.!?]$", s):
        return False

    return s == s.title()


def looks_like_body_line(line: str) -> bool:
    s = " ".join(line.split())
    if not s:
        return False

    return len(s.split()) >= 12 or bool(re.search(r"[,:;.!?]", s))


def strip_leading_header_lines(paragraph: str) -> str:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if not lines:
        return ""

    body_idx = None
    for idx, line in enumerate(lines[:5]):
        if looks_like_body_line(line):
            body_idx = idx
            break

    if body_idx is not None and body_idx > 0:
        leading_block = lines[:body_idx]
        if all(is_header_like_line(line) or is_noisy_fragment(line) for line in leading_block):
            lines = lines[body_idx:]

    return "\n".join(lines).strip()


def split_into_paragraphs(text: str, min_len: int = 80) -> List[str]:
    """
    先以空行切段，再把太短的段落過濾掉。
    """
    paras = [strip_leading_header_lines(p.strip()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    paras = [p for p in paras if not is_noisy_fragment(p)]
    paras = [p for p in paras if len(p) >= min_len]
    return paras


def merge_short_paragraphs(paragraphs: List[str], min_words: int = 40) -> List[str]:
    """
    太短的 paragraph 合併到下一段，避免 chunk 太碎。
    """
    merged = []
    buffer = ""

    for para in paragraphs:
        if is_noisy_fragment(para):
            continue

        if should_buffer_short_paragraph(para, min_words):
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
        kw_lower = kw.lower()
        # use word boundary matching to avoid substring false positives (e.g. "ai" in "Taiwan")
        pattern = r"\b" + re.escape(kw_lower) + r"\b"
        if re.search(pattern, text_lower):
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


def get_issue_frame_scores(text: str) -> Dict[str, int]:
    return {
        issue_frame: count_keyword_hits(text, keywords)
        for issue_frame, keywords in ISSUE_FRAME_KEYWORDS.items()
    }


def classify_issue_frame(text: str, threshold: int = 2) -> Tuple[str, float]:
    scores = get_issue_frame_scores(text)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    if not ranked or ranked[0][1] < threshold:
        return "other", 0.0

    top_frame, top_score = ranked[0]
    total_signal = sum(score for _, score in ranked if score > 0)
    confidence = top_score / total_signal if total_signal else 0.0
    return top_frame, round(confidence, 4)

# Use global SDG_KEYWORDS to classify text into relevant SDGs.
def get_sdg_scores(text: str) -> Dict[str, int]:
    return {
        sdg: count_keyword_hits(text, kws)
        for sdg, kws in SDG_KEYWORDS.items()
    }

def classify_sdg(
    text: str,
    threshold: int = 3,
    max_labels: int = 2,
    secondary_ratio: float = 0.75,
) -> Tuple[List[str], float]:
    scores = get_sdg_scores(text)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    if not ranked or ranked[0][1] < threshold:
        return ["unclassified"], 0.0

    top_sdg, top_score = ranked[0]
    total_signal = sum(score for _, score in ranked if score > 0)
    labels = [top_sdg]

    if max_labels > 1 and len(ranked) > 1:
        second_sdg, second_score = ranked[1]
        if (
            second_score >= threshold
            and top_score > 0
            and (second_score / top_score) >= secondary_ratio
        ):
            labels.append(second_sdg)

    confidence = top_score / total_signal if total_signal else 0.0
    if len(labels) == 2:
        confidence = (top_score + ranked[1][1]) / total_signal if total_signal else 0.0

    return labels, round(confidence, 4)


def get_dominant_sdg(text: str) -> str:
    labels, _ = classify_sdg(text, max_labels=1)
    return labels[0]

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

    # protect multi-word phrases by joining with underscore before tokenization
    for phrase in PROTECTED_PHRASES:
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", phrase.replace(" ", "_"), text)

    # 保留字母與數字，其他大致清掉
    text = re.sub(r"[^a-z0-9\s\-_]", " ", text)
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

    # Adjust min_df if there are too few documents
    num_docs = len(grouped)
    if num_docs < min_df:
        min_df = 1
    max_df_val = 0.9 if num_docs > 1 else 1.0

    vectorizer = TfidfVectorizer(
        min_df=min_df,
        ngram_range=ngram_range,
        max_df=max_df_val
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


def build_cosine_similarity_by_group(
    df: pd.DataFrame,
    group_col: str,
    text_col: str = "clean_text",
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1, 2),
) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col)[text_col]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )

    if grouped.empty:
        return pd.DataFrame()

    num_docs = len(grouped)
    if num_docs < min_df:
        min_df = 1
    max_df_val = 0.9 if num_docs > 1 else 1.0

    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df_val,
        ngram_range=ngram_range,
    )
    matrix = vectorizer.fit_transform(grouped[text_col])
    similarity = cosine_similarity(matrix)

    return pd.DataFrame(
        similarity,
        index=grouped[group_col].tolist(),
        columns=grouped[group_col].tolist(),
    )


def build_issue_frame_distribution(
    df: pd.DataFrame,
    group_col: str | None = None,
    issue_col: str = "issue_frame",
) -> pd.DataFrame:
    if group_col is None:
        summary = df[issue_col].value_counts().reset_index()
        summary.columns = [issue_col, "count"]
        summary["share"] = (summary["count"] / summary["count"].sum()).round(4)
        return summary

    summary = (
        df.groupby([group_col, issue_col])
        .size()
        .reset_index(name="count")
    )
    group_totals = summary.groupby(group_col)["count"].transform("sum")
    summary["share"] = (summary["count"] / group_totals).round(4)
    return summary


def build_issue_frame_cooccurrence(
    df_chunks: pd.DataFrame,
    df_issue_frame_terms: pd.DataFrame,
    issue_col: str = "issue_frame",
    text_col: str = "clean_text",
    top_terms_per_frame: int = 15,
    top_nodes_per_frame: int = 10,
    top_edges_per_frame: int = 20,
    min_edge_weight: int = 2,
) -> pd.DataFrame:
    records = []

    for issue_frame in df_issue_frame_terms[issue_col].dropna().unique():
        frame_terms = (
            df_issue_frame_terms[df_issue_frame_terms[issue_col] == issue_frame]
            .sort_values("tfidf_score", ascending=False)
            .head(top_terms_per_frame)["term"]
            .astype(str)
            .tolist()
        )
        if len(frame_terms) < 2:
            continue

        graph = nx.Graph()
        frame_chunks = df_chunks[df_chunks[issue_col] == issue_frame]
        for text in frame_chunks[text_col].fillna("").astype(str):
            token_set = set(text.split())
            present_terms = sorted(term for term in frame_terms if term in token_set)
            for term_a, term_b in combinations(present_terms, 2):
                if graph.has_edge(term_a, term_b):
                    graph[term_a][term_b]["weight"] += 1
                else:
                    graph.add_edge(term_a, term_b, weight=1)

        if graph.number_of_edges() == 0:
            continue

        filtered_edges = [
            (term_a, term_b, attrs)
            for term_a, term_b, attrs in graph.edges(data=True)
            if attrs.get("weight", 0) >= min_edge_weight
        ]
        if not filtered_edges:
            continue

        filtered_graph = nx.Graph()
        filtered_graph.add_edges_from(filtered_edges)
        weighted_degree = dict(filtered_graph.degree(weight="weight"))
        top_nodes = {
            node
            for node, _ in sorted(
                weighted_degree.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:top_nodes_per_frame]
        }
        filtered_graph = filtered_graph.subgraph(top_nodes).copy()

        sorted_edges = sorted(
            filtered_graph.edges(data=True),
            key=lambda item: item[2].get("weight", 0),
            reverse=True,
        )[:top_edges_per_frame]
        for term_a, term_b, attrs in sorted_edges:
            weight = int(attrs.get("weight", 0))
            records.append(
                {
                    "issue_frame": issue_frame,
                    "term_a": term_a,
                    "term_b": term_b,
                    "weight": weight,
                    "degree_a": float(weighted_degree.get(term_a, 0.0)),
                    "degree_b": float(weighted_degree.get(term_b, 0.0)),
                }
            )

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
        sdg_labels, sdg_confidence = classify_sdg(chunk)
        issue_frame, issue_frame_confidence = classify_issue_frame(chunk)
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
                orientation=orientation,
                sdg_labels=",".join(sdg_labels),
                sdg_confidence=sdg_confidence,
                issue_frame=issue_frame,
                issue_frame_confidence=issue_frame_confidence,
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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df_chunks.to_csv(output_path / "chunks_processed.csv", index=False)

    # top terms by section
    df_section_terms = extract_top_terms_per_group(
        df_chunks, group_col="section_label", top_k=30
    )
    df_section_terms.to_csv(output_path / "tfidf_by_section.csv", index=False)

    # top terms by orientation
    df_orientation_terms = extract_top_terms_per_group(
        df_chunks, group_col="orientation", top_k=30
    )
    df_orientation_terms.to_csv(output_path / "tfidf_by_orientation.csv", index=False)
    df_issue_frame_terms = extract_top_terms_per_group(
        df_chunks, group_col="issue_frame", top_k=30
    )
    df_issue_frame_terms.to_csv(output_path / "tfidf_by_issue_frame.csv", index=False)
    issue_frame_cooccurrence = build_issue_frame_cooccurrence(df_chunks, df_issue_frame_terms)
    issue_frame_cooccurrence.to_csv(output_path / "issue_frame_cooccurrence.csv", index=False)
    similarity_section = build_cosine_similarity_by_group(df_chunks, group_col="section_label")
    similarity_section.to_csv(output_path / "similarity_by_section.csv", index=True)
    similarity_orientation = build_cosine_similarity_by_group(df_chunks, group_col="orientation")
    similarity_orientation.to_csv(output_path / "similarity_by_orientation.csv", index=True)

    # top terms by SDG (explode multi-labels so each chunk contributes to all its SDGs)
    df_exploded = (
        df_chunks.assign(sdg_labels=df_chunks["sdg_labels"].str.split(","))
        .explode("sdg_labels")
    )
    df_exploded["sdg_labels"] = df_exploded["sdg_labels"].str.strip()
    df_exploded = df_exploded[df_exploded["sdg_labels"] != "unclassified"]
    df_sdg_terms = extract_top_terms_per_group(
        df_exploded, group_col="sdg_labels", top_k=30
    )
    df_sdg_terms.to_csv(output_path / "tfidf_by_sdg.csv", index=False)
    similarity_sdg = build_cosine_similarity_by_group(df_exploded, group_col="sdg_labels")
    similarity_sdg.to_csv(output_path / "similarity_by_sdg.csv", index=True)
    issue_frame_distribution = build_issue_frame_distribution(df_chunks)
    issue_frame_distribution.to_csv(output_path / "issue_frame_distribution.csv", index=False)
    issue_frame_by_section = build_issue_frame_distribution(df_chunks, group_col="section_label")
    issue_frame_by_section.to_csv(output_path / "issue_frame_by_section.csv", index=False)

    return {
        "chunks": df_chunks,
        "section_terms": df_section_terms,
        "orientation_terms": df_orientation_terms,
        "sdg_terms": df_sdg_terms,
        "issue_frame_terms": df_issue_frame_terms,
        "section_similarity": similarity_section,
        "orientation_similarity": similarity_orientation,
        "sdg_similarity": similarity_sdg,
        "issue_frame_distribution": issue_frame_distribution,
        "issue_frame_by_section": issue_frame_by_section,
        "issue_frame_cooccurrence": issue_frame_cooccurrence,
    }


def process_all_years(output_base_dir: str = "outputs") -> dict[str, dict]:
    """
    Process all available TSMC reports (2022, 2023, 2024).
    Each year's outputs are saved to {output_base_dir}/{year}/ subdirectory.
    
    Returns:
        dict mapping year -> {chunks_df, outputs_dict, log_info}
    """
    from src.extract_pdf import extract_all_pdfs
    
    # Extract PDFs to text files
    print("=== Extracting PDFs ===")
    report_files = extract_all_pdfs()
    
    if not report_files:
        raise FileNotFoundError("No TSMC reports found. Ensure PDFs are in data/raw/")
    
    # Create output base directory
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Load spaCy model once (reuse for all years)
    print("Loading spaCy model...")
    nlp = load_spacy_model("en_core_web_sm")
    
    # Process each year
    all_results = {}
    for year in sorted(report_files.keys()):
        text_path = report_files[year]
        year_output_dir = output_base / year
        year_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Processing {year}: {text_path}")
        print(f"Output directory: {year_output_dir}")
        print(f"{'='*60}")
        
        raw_text = text_path.read_text(encoding="utf-8")
        df_chunks = run_pipeline(raw_text, nlp)
        outputs = save_outputs(df_chunks, output_dir=str(year_output_dir))
        
        # Generate sentence embeddings for each chunk
        from src.embeddings import EmbeddingPipeline
        embedding_pipeline = EmbeddingPipeline()
        clean_texts = df_chunks['clean_text'].fillna('').tolist()
        embeddings = embedding_pipeline.encode_texts(clean_texts)
        embedding_path = year_output_dir / 'chunk_embeddings.npz'
        embedding_pipeline.save_embeddings(embeddings, embedding_path)
        print(f"  - Saved {embeddings.shape[0]} chunk embeddings to {embedding_path}")
        
        all_results[year] = {
            "chunks": df_chunks,
            "outputs": outputs,
            "embeddings_path": str(embedding_path),
            "output_dir": str(year_output_dir),
            "text_path": str(text_path),
        }
        
        print(f"\n{year} Summary:")
        print(f"  - Chunks: {len(df_chunks)}")
        print(f"  - Section distribution: {dict(df_chunks['section_label'].value_counts())}")
        print(f"  - SDG count: {df_chunks['sdg_labels'].str.split(',').explode().str.strip().nunique()}")
    
    # Final summary
    print(f"\n\n{'='*60}")
    print("SUMMARY: All years processed successfully")
    print(f"{'='*60}")
    for year in sorted(all_results.keys()):
        result = all_results[year]
        print(f"{year}: {len(result['chunks'])} chunks → {result['output_dir']}/")
    
    return all_results
