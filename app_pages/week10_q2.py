from __future__ import annotations

import html
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as streamlit_cache
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

try:
    from gensim.models import Word2Vec
except ImportError:
    Word2Vec = None


CUSTOM_STOPWORDS = {
    "tsmc",
    "company",
    "report",
    "sustainability",
    "sustainable",
    "page",
    "section",
    "table",
    "figure",
    "including",
    "related",
    "based",
    "shall",
    "also",
    "year",
    "years",
    "and",
}
STOPWORDS = sorted(set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS))
TOKEN_RE = re.compile(r"[a-z][a-z]{2,}")
SPACED_LETTER_RE = re.compile(r"(?:\b[a-zA-Z]\s+){5,}\b[a-zA-Z]\b")
SHORT_FRAGMENT_RE = re.compile(r"(?:\b[a-zA-Z]{1,2}\b\s*){6,}")
MASHED_REPORT_FIELD_RE = re.compile(r"\b[A-Z]?[a-z]+(?:from|for|in|of|to|and)[A-Za-z0-9()]+")


def _parse_sdg_labels(value: str) -> list[str]:
    labels = [label.strip() for label in str(value).split(",") if label.strip()]
    return [label for label in labels if label != "unclassified"]


def _prettify(value: str, section_label_map: dict[str, str], sdg_label_map: dict[str, str]) -> str:
    if value in section_label_map:
        return section_label_map[value]
    if value in sdg_label_map:
        return sdg_label_map[value]
    return value.replace("_", " ").title()


def _tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(str(text).lower())
    return [token for token in tokens if token not in STOPWORDS]


def _filter_chunks(
    df_chunks: pd.DataFrame,
    selected_sections: list[str],
    selected_orientations: list[str],
    selected_sdgs: list[str],
    keyword: str,
) -> pd.DataFrame:
    df = df_chunks.copy()
    if selected_sections:
        df = df[df["section_label"].isin(selected_sections)]
    if selected_orientations:
        df = df[df["orientation"].isin(selected_orientations)]
    if selected_sdgs:
        selected = set(selected_sdgs)
        df = df[df["sdg_labels"].map(_parse_sdg_labels).map(lambda labels: bool(set(labels) & selected))]
    if keyword.strip():
        needle = keyword.strip().lower()
        df = df[df["clean_text"].str.lower().str.contains(needle, regex=False, na=False)]
    return df.copy()


@streamlit_cache.cache_resource(show_spinner="Training Word2Vec model...")
def _train_word2vec(token_rows: tuple[tuple[str, ...], ...], vector_size: int, window: int, min_count: int):
    if Word2Vec is None:
        return None
    usable_rows = [row for row in token_rows if len(row) >= 3]
    if not usable_rows:
        return None
    return Word2Vec(
        sentences=usable_rows,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        sg=1,
        epochs=30,
        seed=42,
    )


def _build_keyword_options(token_rows: list[list[str]], model, max_terms: int = 80) -> list[str]:
    counts = Counter(token for row in token_rows for token in row)
    return [term for term, _ in counts.most_common(max_terms) if term in model.wv]


def _build_similar_terms(model, selected_word: str, top_n: int) -> pd.DataFrame:
    if not selected_word or selected_word not in model.wv:
        return pd.DataFrame(columns=["term", "similarity"])
    rows = model.wv.most_similar(selected_word, topn=top_n)
    return pd.DataFrame(rows, columns=["term", "similarity"])


def _build_embedding_map(model, token_rows: list[list[str]], max_terms: int) -> pd.DataFrame:
    counts = Counter(token for row in token_rows for token in row)
    terms = [term for term, _ in counts.most_common(max_terms) if term in model.wv]
    if len(terms) < 3:
        return pd.DataFrame(columns=["term", "frequency", "x", "y"])

    vectors = np.array([model.wv[term] for term in terms])
    coords = PCA(n_components=2, random_state=42).fit_transform(vectors)
    return pd.DataFrame(
        {
            "term": terms,
            "frequency": [counts[term] for term in terms],
            "x": coords[:, 0],
            "y": coords[:, 1],
        }
    )


def _average_vector(tokens: list[str], model) -> np.ndarray | None:
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def _build_group_similarity(
    df: pd.DataFrame,
    token_rows: list[list[str]],
    model,
    group_col: str,
    section_label_map: dict[str, str],
    sdg_label_map: dict[str, str],
) -> pd.DataFrame:
    records = []
    for index, tokens in zip(df.index, token_rows):
        row = df.loc[index]
        vector = _average_vector(tokens, model)
        if vector is None:
            continue
        if group_col == "sdg_labels":
            groups = _parse_sdg_labels(row.get("sdg_labels", ""))
        else:
            groups = [row.get(group_col, "")]
        for group in groups:
            if str(group).strip():
                records.append({"group": group, "vector": vector})

    if not records:
        return pd.DataFrame()

    vectors_by_group = {}
    for group in sorted({record["group"] for record in records}):
        group_vectors = [record["vector"] for record in records if record["group"] == group]
        if group_vectors:
            vectors_by_group[group] = np.mean(group_vectors, axis=0)

    if len(vectors_by_group) < 2:
        return pd.DataFrame()

    labels = list(vectors_by_group.keys())
    matrix = np.array([vectors_by_group[label] for label in labels])
    similarity = cosine_similarity(matrix)
    pretty_labels = [_prettify(label, section_label_map, sdg_label_map) for label in labels]
    return pd.DataFrame(similarity, index=pretty_labels, columns=pretty_labels)


def _build_term_group_scores(
    df: pd.DataFrame,
    model,
    selected_word: str,
    group_col: str,
    section_label_map: dict[str, str],
    sdg_label_map: dict[str, str],
) -> pd.DataFrame:
    if not selected_word or selected_word not in model.wv:
        return pd.DataFrame(columns=["group", "semantic_score"])

    target_vector = model.wv[selected_word].reshape(1, -1)
    rows = []
    for group, group_df in df.groupby(group_col):
        token_rows = [_tokenize(text) for text in group_df["clean_text"].fillna("").astype(str)]
        vectors = [_average_vector(tokens, model) for tokens in token_rows]
        vectors = [vector for vector in vectors if vector is not None]
        if not vectors:
            continue
        group_vector = np.mean(vectors, axis=0).reshape(1, -1)
        rows.append(
            {
                "group": _prettify(str(group), section_label_map, sdg_label_map),
                "semantic_score": float(cosine_similarity(target_vector, group_vector)[0, 0]),
            }
        )
    return pd.DataFrame(rows).sort_values("semantic_score", ascending=False)


def _clean_evidence_text(raw_text: str, clean_text: str) -> str:
    text = str(raw_text or "").strip() or str(clean_text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u2022\u25cf]+", " ", text)
    text = SPACED_LETTER_RE.sub(" ", text)
    text = SHORT_FRAGMENT_RE.sub(" ", text)
    text = MASHED_REPORT_FIELD_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = text.strip(" -|")

    if not text or len(text.split()) < 12:
        text = str(clean_text or "").strip()
        text = SPACED_LETTER_RE.sub(" ", text)
        text = SHORT_FRAGMENT_RE.sub(" ", text)
        text = MASHED_REPORT_FIELD_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip(" -|")

    if len(text) > 520:
        text = text[:520].rsplit(" ", 1)[0] + "..."
    return text


def _artifact_score(text: str) -> float:
    text = str(text or "")
    if not text.strip():
        return 1.0
    spaced_chars = sum(len(match.group(0)) for match in SPACED_LETTER_RE.finditer(text))
    fragment_chars = sum(len(match.group(0)) for match in SHORT_FRAGMENT_RE.finditer(text))
    mashed_chars = sum(len(match.group(0)) for match in MASHED_REPORT_FIELD_RE.finditer(text))
    return (spaced_chars + fragment_chars + mashed_chars) / max(len(text), 1)


def _readability_score(raw_text: str, clean_text: str) -> float:
    text = _clean_evidence_text(raw_text, clean_text)
    words = re.findall(r"[A-Za-z][A-Za-z-]{2,}", text)
    if not words:
        return 0.0
    long_word_ratio = sum(len(word) >= 4 for word in words) / len(words)
    punctuation_bonus = min(text.count(".") + text.count(","), 5) * 0.06
    numeric_penalty = min(len(re.findall(r"\d", text)) / max(len(text), 1), 0.35)

    raw = str(raw_text or "")
    short_lines = [line.strip() for line in raw.splitlines() if 0 < len(line.strip()) <= 28]
    line_penalty = min(len(short_lines) * 0.035, 0.4)
    table_terms = re.findall(
        r"\b(input/output|outcome|metric tons|note \d+|valuation|dimensions|target:|scope \d)\b",
        text.lower(),
    )
    table_penalty = min(len(table_terms) * 0.12, 0.5)
    return long_word_ratio + punctuation_bonus - numeric_penalty - line_penalty - table_penalty


def _render_evidence_cards(st, df: pd.DataFrame, selected_word: str, max_cards: int = 3) -> None:
    if not selected_word:
        st.info("Choose a word to inspect supporting report chunks.")
        return

    evidence = df[df["clean_text"].str.lower().str.contains(selected_word.lower(), regex=False, na=False)].copy()
    if not evidence.empty:
        evidence["_artifact_score"] = evidence["raw_text"].map(_artifact_score)
        evidence["_readability_score"] = evidence.apply(
            lambda row: _readability_score(row.get("raw_text", ""), row.get("clean_text", "")),
            axis=1,
        )
        evidence["_evidence_quality"] = evidence["_readability_score"] - (10 * evidence["_artifact_score"])
        sort_cols = ["_evidence_quality", "_readability_score"]
        ascending = [False, False]
        if "sdg_confidence" in evidence.columns:
            sort_cols.append("sdg_confidence")
            ascending.append(False)
        evidence = evidence.sort_values(sort_cols, ascending=ascending).head(max_cards)
    if evidence.empty:
        st.info("No supporting chunks found for this word under the current filters.")
        return

    for row in evidence.itertuples(index=False):
        section = str(getattr(row, "section_label", "")).replace("_", " ").title()
        orientation = str(getattr(row, "orientation", "")).replace("_", " ").title()
        text = _clean_evidence_text(getattr(row, "raw_text", ""), getattr(row, "clean_text", ""))
        st.markdown(f"**{section} | {orientation}**")
        st.markdown(
            f"<p style='color:#475569;font-size:0.94rem;line-height:1.75;margin:0'>{html.escape(text)}</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")


def _plot_similar_terms(st, similar_df: pd.DataFrame, selected_word: str) -> None:
    if similar_df.empty:
        st.info("No similar words are available for the current model settings.")
        return
    fig = px.bar(
        similar_df.sort_values("similarity"),
        x="similarity",
        y="term",
        orientation="h",
        title=f"Words Used in Similar Contexts to '{selected_word}'",
        color="similarity",
        color_continuous_scale="Tealgrn",
        range_x=[0, 1],
    )
    fig.update_layout(height=420, yaxis_title="", xaxis_title="Word2Vec cosine similarity")
    st.plotly_chart(fig, use_container_width=True)


def _plot_embedding_map(st, embedding_df: pd.DataFrame, selected_word: str) -> None:
    if embedding_df.empty:
        st.info("Not enough vocabulary is available to draw the embedding map.")
        return
    embedding_df = embedding_df.copy()
    embedding_df["is_selected"] = embedding_df["term"].eq(selected_word)
    fig = px.scatter(
        embedding_df,
        x="x",
        y="y",
        text="term",
        size="frequency",
        color="is_selected",
        hover_data=["term", "frequency"],
        title="Word2Vec ESG Vocabulary Map",
        color_discrete_map={True: "#dc2626", False: "#0f766e"},
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=620, xaxis_title="PCA 1", yaxis_title="PCA 2", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _plot_similarity_heatmap(st, similarity_df: pd.DataFrame, title: str) -> None:
    if similarity_df.empty:
        st.info("Not enough groups are available to calculate semantic similarity.")
        return
    fig = px.imshow(
        similarity_df,
        text_auto=".2f",
        color_continuous_scale="Tealgrn",
        zmin=0,
        zmax=1,
        title=title,
    )
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)


def _plot_term_group_scores(st, term_group_df: pd.DataFrame, selected_word: str) -> None:
    if term_group_df.empty:
        return
    plot_df = term_group_df.sort_values("semantic_score", ascending=True)
    fig = px.bar(
        plot_df,
        x="semantic_score",
        y="group",
        orientation="h",
        text=plot_df["semantic_score"].map(lambda value: f"{value:.2f}"),
        title=f"Closest Strategic Frames to '{selected_word}'",
        color="semantic_score",
        color_continuous_scale="Tealgrn",
        range_x=[0, 1],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=340,
        margin=dict(l=8, r=42, t=52, b=28),
        xaxis_title="Semantic score",
        yaxis_title="",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_page(
    *,
    df_chunks: pd.DataFrame,
    render_page_header,
    render_metric_grid,
    render_card_grid,
    section_label_map: dict[str, str],
    sdg_label_map: dict[str, str],
    st,
) -> None:
    render_page_header(
        "Week 10 Q2",
        "Kiwi's Week10 Q2",
        "A Word2Vec page that learns ESG word embeddings from TSMC report chunks and turns semantic relationships into similar-word, map, and group-similarity views.",
    )

    if Word2Vec is None:
        st.error("Word2Vec is unavailable because gensim is not installed. Run `pip install -r requirements.txt` first.")
        return

    required_cols = {"clean_text", "raw_text", "section_label", "orientation", "sdg_labels"}
    missing_cols = sorted(required_cols - set(df_chunks.columns))
    if missing_cols:
        st.error(f"Missing required columns for Week 10 page: {', '.join(missing_cols)}")
        return

    all_sections = sorted([x for x in df_chunks["section_label"].dropna().unique() if str(x).strip()])
    all_orientations = sorted([x for x in df_chunks["orientation"].dropna().unique() if str(x).strip()])
    all_sdgs = sorted({label for labels in df_chunks["sdg_labels"].map(_parse_sdg_labels) for label in labels})

    st.markdown("### Controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        pretty_to_section = {_prettify(x, section_label_map, sdg_label_map): x for x in all_sections}
        selected_sections_pretty = st.multiselect("Strategic framing", options=list(pretty_to_section), default=[])
        selected_sections = [pretty_to_section[x] for x in selected_sections_pretty]
    with c2:
        selected_orientations = st.multiselect(
            "Orientation",
            options=all_orientations,
            default=[],
            format_func=lambda x: x.replace("_", " ").title(),
        )
    with c3:
        pretty_to_sdg = {_prettify(x, section_label_map, sdg_label_map): x for x in all_sdgs}
        selected_sdgs_pretty = st.multiselect("SDG theme", options=list(pretty_to_sdg), default=[])
        selected_sdgs = [pretty_to_sdg[x] for x in selected_sdgs_pretty]

    c4, c5, c6 = st.columns(3)
    with c4:
        vector_size = st.slider("Embedding dimension", min_value=40, max_value=160, value=80, step=20)
    with c5:
        min_count = st.slider("Minimum word frequency", min_value=1, max_value=8, value=2, step=1)
    with c6:
        keyword_filter = st.text_input("Optional keyword filter", "")

    filtered_df = _filter_chunks(
        df_chunks,
        selected_sections=selected_sections,
        selected_orientations=selected_orientations,
        selected_sdgs=selected_sdgs,
        keyword=keyword_filter,
    )
    if filtered_df.empty:
        st.warning("No chunks match the current filters. Widen the filters to continue.")
        return

    token_rows = tuple(
        tuple(_tokenize(text))
        for text in filtered_df["clean_text"].fillna("").astype(str)
    )
    model = _train_word2vec(token_rows, vector_size=vector_size, window=5, min_count=min_count)
    if model is None or len(model.wv) < 3:
        st.warning("Not enough clean tokens are available to train a useful Word2Vec model under these settings.")
        return

    keyword_options = _build_keyword_options(token_rows, model)
    if not keyword_options:
        st.warning("No vocabulary terms are available after filtering and Word2Vec training.")
        return

    selected_word = st.selectbox("Focus word", options=keyword_options, index=0)
    top_n = st.slider("Similar words to show", min_value=5, max_value=25, value=12, step=1)
    map_terms = st.slider("Words on embedding map", min_value=20, max_value=100, value=60, step=10)
    group_dimension = st.radio(
        "Semantic similarity group",
        ["Strategic framing", "SDG theme", "Orientation"],
        horizontal=True,
    )

    group_col = {
        "Strategic framing": "section_label",
        "SDG theme": "sdg_labels",
        "Orientation": "orientation",
    }[group_dimension]

    similar_df = _build_similar_terms(model, selected_word, top_n=top_n)
    embedding_df = _build_embedding_map(model, token_rows, max_terms=map_terms)
    similarity_df = _build_group_similarity(
        filtered_df,
        token_rows,
        model,
        group_col=group_col,
        section_label_map=section_label_map,
        sdg_label_map=sdg_label_map,
    )
    term_group_df = _build_term_group_scores(
        filtered_df,
        model,
        selected_word,
        group_col="section_label",
        section_label_map=section_label_map,
        sdg_label_map=sdg_label_map,
    )

    render_metric_grid(
        [
            {"label": "Filtered Chunks", "value": len(filtered_df), "note": "Report chunks used for this model"},
            {"label": "Vocabulary", "value": len(model.wv), "note": "Words retained by Word2Vec"},
            {"label": "Embedding Dim.", "value": vector_size, "note": "Numbers learned for each word"},
            {"label": "Focus Word", "value": selected_word, "note": "Anchor term for semantic exploration"},
        ],
        columns=4,
    )

    render_card_grid(
        [
            {
                "kicker": "Technique",
                "title": "Word2Vec",
                "text": "The model learns word vectors from neighboring words inside ESG report chunks, so words with similar contexts receive similar vectors.",
                "highlight": True,
            },
            {
                "kicker": "Interpretation",
                "title": "Semantic Similarity",
                "text": "Similar terms and group heatmaps show which ESG ideas are discussed in related language, even when the exact words are not identical.",
            },
        ],
        columns=2,
    )

    tabs = st.tabs(["Similar Terms", "Embedding Map", "Group Similarity", "Evidence"])
    with tabs[0]:
        st.markdown("### Similar ESG Terms")
        st.caption("Word2Vec cosine similarity finds words that appear in similar ESG contexts.")
        left, right = st.columns([1.2, 1])
        with left:
            _plot_similar_terms(st, similar_df, selected_word)
        with right:
            if not similar_df.empty:
                st.dataframe(similar_df, use_container_width=True, hide_index=True)
            if not term_group_df.empty:
                _plot_term_group_scores(st, term_group_df, selected_word)

    with tabs[1]:
        st.markdown("### Word Embedding Map")
        st.caption("PCA projects high-dimensional Word2Vec vectors into two dimensions for a presentation-friendly map.")
        _plot_embedding_map(st, embedding_df, selected_word)

    with tabs[2]:
        st.markdown("### Semantic Group Similarity")
        st.caption("Each group vector averages the Word2Vec vectors of its chunks; cosine similarity compares the resulting ESG language.")
        _plot_similarity_heatmap(st, similarity_df, f"Word2Vec Semantic Similarity by {group_dimension}")

    with tabs[3]:
        st.markdown("### Representative Chunks")
        st.caption("These examples connect the Word2Vec focus word back to actual report text.")
        _render_evidence_cards(st, filtered_df, selected_word)
