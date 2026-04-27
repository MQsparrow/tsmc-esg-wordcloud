from __future__ import annotations

import html
import itertools
import re
from collections import Counter

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    "year",
    "years",
    "including",
    "related",
    "based",
    "total",
    "new",
    "used",
    "use",
    "shall",
    "also",
    "and",
}
STOPWORDS = sorted(set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS))
TOKEN_RE = re.compile(r"[a-z][a-z]{2,}")
SPACED_LETTER_RE = re.compile(r"(?:\b[a-zA-Z]\s+){5,}\b[a-zA-Z]\b")
SHORT_FRAGMENT_RE = re.compile(r"(?:\b[a-zA-Z]{1,2}\b\s*){6,}")
MASHED_REPORT_FIELD_RE = re.compile(r"\b[A-Z]?[a-z]+(?:from|for|in|of|to|and)[A-Za-z0-9()]+")

ESG_TONE_LEXICON = {
    "Action": {
        "achieve",
        "advance",
        "build",
        "complete",
        "develop",
        "enable",
        "expand",
        "improve",
        "increase",
        "launch",
        "promote",
        "reduce",
        "strengthen",
    },
    "Risk": {
        "audit",
        "compliance",
        "control",
        "exposure",
        "hazard",
        "impact",
        "incident",
        "risk",
        "safety",
        "shortage",
        "supplier",
        "violation",
        "waste",
    },
    "People": {
        "care",
        "community",
        "education",
        "employee",
        "health",
        "human",
        "learning",
        "rights",
        "safety",
        "student",
        "talent",
        "training",
        "volunteer",
    },
    "Innovation": {
        "advanced",
        "ai",
        "chip",
        "innovation",
        "patent",
        "process",
        "research",
        "semiconductor",
        "technology",
        "wafer",
    },
    "Environment": {
        "carbon",
        "chemical",
        "climate",
        "electricity",
        "emission",
        "energy",
        "recycle",
        "renewable",
        "water",
        "waste",
    },
}


def _parse_sdg_labels(value: str) -> list[str]:
    labels = [x.strip() for x in str(value).split(",") if x.strip()]
    return [x for x in labels if x != "unclassified"]


def _prettify(value: str, section_label_map: dict[str, str], sdg_label_map: dict[str, str]) -> str:
    if value in section_label_map:
        return section_label_map[value]
    if value in sdg_label_map:
        return sdg_label_map[value]
    return value.replace("_", " ").title()


def _tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(str(text).lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


def _clean_evidence_text(raw_text: str, clean_text: str) -> str:
    text = str(raw_text or "").strip() or str(clean_text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u2022\u25cf]+", " ", text)
    text = SPACED_LETTER_RE.sub(" ", text)
    text = SHORT_FRAGMENT_RE.sub(" ", text)
    text = MASHED_REPORT_FIELD_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" -|")

    if not text or len(text.split()) < 12:
        text = str(clean_text or "").strip()
        text = re.sub(r"\s+", " ", text)

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
        df = df[df["sdg_labels"].map(_parse_sdg_labels).map(lambda labels: bool(set(labels) & set(selected_sdgs)))]
    if keyword.strip():
        needle = keyword.strip().lower()
        df = df[df["clean_text"].str.lower().str.contains(needle, regex=False, na=False)]
    return df.copy()


def _build_phrase_table(texts: pd.Series, ngram_size: int, top_n: int) -> pd.DataFrame:
    clean_texts = texts.fillna("").astype(str)
    if clean_texts.str.strip().eq("").all():
        return pd.DataFrame(columns=["phrase", "count"])

    vectorizer = CountVectorizer(
        stop_words=STOPWORDS,
        ngram_range=(ngram_size, ngram_size),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b",
        min_df=2,
        max_features=300,
    )
    try:
        matrix = vectorizer.fit_transform(clean_texts)
    except ValueError:
        return pd.DataFrame(columns=["phrase", "count"])

    counts = matrix.sum(axis=0).A1
    phrases = vectorizer.get_feature_names_out()
    phrase_df = pd.DataFrame({"phrase": phrases, "count": counts})
    return phrase_df.sort_values(["count", "phrase"], ascending=[False, True]).head(top_n).reset_index(drop=True)


def _build_keyword_stats(texts: pd.Series, top_terms: int, max_edges: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    token_rows = [_tokenize(text) for text in texts.fillna("").astype(str)]
    token_counter = Counter(token for row in token_rows for token in row)
    vocabulary = {term for term, _ in token_counter.most_common(top_terms)}

    pair_counter: Counter[tuple[str, str]] = Counter()
    for row in token_rows:
        unique_terms = sorted(set(row) & vocabulary)
        for term_a, term_b in itertools.combinations(unique_terms, 2):
            pair_counter[(term_a, term_b)] += 1

    term_df = pd.DataFrame(token_counter.most_common(top_terms), columns=["term", "frequency"])
    pair_df = pd.DataFrame(
        [{"source": a, "target": b, "weight": weight} for (a, b), weight in pair_counter.items()]
    )
    if pair_df.empty:
        pair_df = pd.DataFrame(columns=["source", "target", "weight"])
    else:
        pair_df = pair_df.sort_values(["weight", "source", "target"], ascending=[False, True, True]).head(max_edges)
    return term_df, pair_df.reset_index(drop=True)


def _build_group_documents(
    df: pd.DataFrame,
    dimension: str,
    section_label_map: dict[str, str],
    sdg_label_map: dict[str, str],
) -> pd.DataFrame:
    if dimension == "SDG theme":
        records = []
        for row in df.itertuples(index=False):
            labels = _parse_sdg_labels(getattr(row, "sdg_labels", ""))
            for label in labels:
                records.append(
                    {
                        "group": _prettify(label, section_label_map, sdg_label_map),
                        "clean_text": getattr(row, "clean_text", ""),
                    }
                )
        group_df = pd.DataFrame(records)
    else:
        source_col = "orientation" if dimension == "Orientation" else "section_label"
        group_df = df[[source_col, "clean_text"]].rename(columns={source_col: "group"}).copy()
        group_df["group"] = group_df["group"].map(lambda x: _prettify(str(x), section_label_map, sdg_label_map))

    if group_df.empty:
        return pd.DataFrame(columns=["group", "document"])

    group_docs = (
        group_df.dropna(subset=["group"])
        .assign(clean_text=lambda x: x["clean_text"].fillna("").astype(str))
        .groupby("group", as_index=False)["clean_text"]
        .agg(lambda rows: " ".join(row for row in rows if row.strip()))
        .rename(columns={"clean_text": "document"})
    )
    return group_docs[group_docs["document"].str.strip().ne("")].reset_index(drop=True)


def _build_tfidf_comparison(
    df: pd.DataFrame,
    dimension: str,
    top_terms_per_group: int,
    section_label_map: dict[str, str],
    sdg_label_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_docs = _build_group_documents(df, dimension, section_label_map, sdg_label_map)
    if len(group_docs) < 2:
        return (
            pd.DataFrame(columns=["group", "term", "tfidf_score"]),
            pd.DataFrame(),
        )

    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b",
        max_features=650,
        min_df=1,
        ngram_range=(1, 2),
    )
    try:
        matrix = vectorizer.fit_transform(group_docs["document"])
    except ValueError:
        return (
            pd.DataFrame(columns=["group", "term", "tfidf_score"]),
            pd.DataFrame(),
        )

    terms = vectorizer.get_feature_names_out()
    records = []
    dense_matrix = matrix.toarray()
    for row_idx, group in enumerate(group_docs["group"]):
        top_idx = dense_matrix[row_idx].argsort()[::-1][:top_terms_per_group]
        for term_idx in top_idx:
            score = dense_matrix[row_idx][term_idx]
            if score > 0:
                records.append(
                    {
                        "group": group,
                        "term": terms[term_idx],
                        "tfidf_score": round(float(score), 4),
                    }
                )

    similarity = cosine_similarity(matrix)
    similarity_df = pd.DataFrame(
        similarity,
        index=group_docs["group"],
        columns=group_docs["group"],
    )
    return pd.DataFrame(records), similarity_df


def _build_cluster_view(
    df: pd.DataFrame,
    cluster_count: int,
    section_label_map: dict[str, str],
    sdg_label_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 4:
        return (
            pd.DataFrame(columns=["x", "y", "cluster", "section", "orientation", "preview"]),
            pd.DataFrame(columns=["cluster", "top_terms", "representative_chunk"]),
        )

    clean_texts = df["clean_text"].fillna("").astype(str)
    max_clusters = min(cluster_count, len(df), 8)
    vectorizer = TfidfVectorizer(
        stop_words=STOPWORDS,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b",
        max_features=450,
        min_df=2,
        ngram_range=(1, 2),
    )
    try:
        matrix = vectorizer.fit_transform(clean_texts)
    except ValueError:
        return (
            pd.DataFrame(columns=["x", "y", "cluster", "section", "orientation", "preview"]),
            pd.DataFrame(columns=["cluster", "top_terms", "representative_chunk"]),
        )

    if matrix.shape[0] < 4 or matrix.shape[1] < 2:
        return (
            pd.DataFrame(columns=["x", "y", "cluster", "section", "orientation", "preview"]),
            pd.DataFrame(columns=["cluster", "top_terms", "representative_chunk"]),
        )

    clusterer = KMeans(n_clusters=max_clusters, random_state=9, n_init=10)
    labels = clusterer.fit_predict(matrix)
    coords = PCA(n_components=2, random_state=9).fit_transform(matrix.toarray())

    scatter_df = df.copy().reset_index(drop=True)
    scatter_df["x"] = coords[:, 0]
    scatter_df["y"] = coords[:, 1]
    scatter_df["cluster"] = [f"Cluster {label + 1}" for label in labels]
    scatter_df["section"] = scatter_df["section_label"].map(lambda x: _prettify(x, section_label_map, sdg_label_map))
    scatter_df["orientation"] = scatter_df["orientation"].str.replace("_", " ").str.title()
    scatter_df["preview"] = scatter_df.apply(
        lambda row: _clean_evidence_text(row.get("raw_text", ""), row.get("clean_text", ""))[:260],
        axis=1,
    )

    feature_names = vectorizer.get_feature_names_out()
    summary_records = []
    for cluster_id in range(max_clusters):
        member_idx = [idx for idx, label in enumerate(labels) if label == cluster_id]
        if not member_idx:
            continue

        centroid = clusterer.cluster_centers_[cluster_id]
        top_terms = [feature_names[idx] for idx in centroid.argsort()[::-1][:8]]
        distances = matrix[member_idx].dot(centroid)
        representative_local_idx = int(distances.argmax())
        representative_idx = member_idx[representative_local_idx]
        representative = _clean_evidence_text(
            df.iloc[representative_idx].get("raw_text", ""),
            df.iloc[representative_idx].get("clean_text", ""),
        )
        summary_records.append(
            {
                "cluster": f"Cluster {cluster_id + 1}",
                "top_terms": ", ".join(top_terms),
                "representative_chunk": representative[:360],
            }
        )

    return scatter_df, pd.DataFrame(summary_records)


def _build_tone_table(df: pd.DataFrame, group_col: str, section_label_map: dict[str, str], sdg_label_map: dict[str, str]) -> pd.DataFrame:
    records = []
    for group, sub_df in df.groupby(group_col):
        tokens = [token for text in sub_df["clean_text"].fillna("").astype(str) for token in _tokenize(text)]
        token_count = max(len(tokens), 1)
        token_counter = Counter(tokens)
        for tone, words in ESG_TONE_LEXICON.items():
            count = sum(token_counter[word] for word in words)
            records.append(
                {
                    "group": _prettify(str(group), section_label_map, sdg_label_map),
                    "tone": tone,
                    "score": round(count / token_count * 1000, 2),
                    "raw_count": count,
                }
            )
    return pd.DataFrame(records)


def _plot_phrase_bar(st, phrase_df: pd.DataFrame, title: str) -> None:
    if phrase_df.empty:
        st.info("No repeated phrases found for this filter. Try a broader section or a smaller n-gram size.")
        return

    plot_df = phrase_df.sort_values("count", ascending=True)
    fig = px.bar(
        plot_df,
        x="count",
        y="phrase",
        orientation="h",
        text="count",
        color="count",
        color_continuous_scale="Tealgrn",
        title=title,
    )
    fig.update_layout(
        height=max(390, 26 * len(plot_df)),
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis_title="Phrase frequency",
        yaxis_title="",
        coloraxis_showscale=False,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.92)",
    )
    fig.update_traces(hovertemplate="Phrase: %{y}<br>Count: %{x}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})


def _plot_pair_heatmap(st, pair_df: pd.DataFrame, top_pairs: int) -> None:
    if pair_df.empty:
        st.info("No keyword co-occurrence pairs found for this filter.")
        return

    top_df = pair_df.head(top_pairs).copy()
    label = top_df["source"] + " + " + top_df["target"]
    fig = px.bar(
        top_df.assign(pair=label).sort_values("weight", ascending=True),
        x="weight",
        y="pair",
        orientation="h",
        color="weight",
        color_continuous_scale="Blues",
        text="weight",
        title="Strongest Keyword Co-occurrence Pairs",
    )
    fig.update_layout(
        height=max(360, 26 * len(top_df)),
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis_title="Chunks where both terms appear",
        yaxis_title="",
        coloraxis_showscale=False,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.92)",
    )
    fig.update_traces(hovertemplate="Pair: %{y}<br>Shared chunks: %{x}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})


def _plot_tfidf_terms(st, tfidf_df: pd.DataFrame) -> None:
    if tfidf_df.empty:
        st.info("Not enough grouped text to calculate distinctive TF-IDF terms.")
        return

    group_count = tfidf_df["group"].nunique()
    plot_df = tfidf_df.sort_values(["group", "tfidf_score"], ascending=[True, True])
    fig = px.bar(
        plot_df,
        x="tfidf_score",
        y="term",
        color="group",
        facet_row="group",
        orientation="h",
        text="tfidf_score",
        custom_data=["group"],
        title="Most Distinctive TF-IDF Terms by Group",
    )
    fig.update_layout(
        height=max(460, min(900, 150 * group_count)),
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis_title="TF-IDF score",
        yaxis_title="",
        showlegend=False,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.92)",
    )
    fig.update_traces(
        texttemplate="%{x:.2f}",
        textposition="outside",
        hovertemplate="Group: %{customdata[0]}<br>Term: %{y}<br>TF-IDF: %{x:.3f}<extra></extra>",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_yaxes(matches=None)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})


def _plot_similarity_heatmap(st, similarity_df: pd.DataFrame) -> None:
    if similarity_df.empty:
        st.info("Similarity needs at least two groups with usable text.")
        return

    fig = px.imshow(
        similarity_df,
        text_auto=".2f",
        color_continuous_scale="Tealgrn",
        zmin=0,
        zmax=1,
        aspect="auto",
        title="Cosine Similarity Between Groups",
    )
    fig.update_layout(
        height=max(390, 48 * len(similarity_df)),
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis_title="",
        yaxis_title="",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.92)",
    )
    fig.update_traces(hovertemplate="%{y} vs %{x}<br>Similarity: %{z:.2f}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})


def _plot_network(st, pair_df: pd.DataFrame, term_df: pd.DataFrame) -> None:
    if pair_df.empty:
        st.info("The current filter is too narrow to draw a co-occurrence network.")
        return

    graph = nx.Graph()
    frequencies = dict(zip(term_df["term"], term_df["frequency"]))
    for row in pair_df.itertuples(index=False):
        graph.add_edge(row.source, row.target, weight=int(row.weight))

    if graph.number_of_nodes() < 2:
        st.info("The network needs at least two connected keywords.")
        return

    positions = nx.spring_layout(graph, seed=9, k=0.7)

    edge_x = []
    edge_y = []
    edge_text = []
    for source, target, data in graph.edges(data=True):
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{source} + {target}: {data['weight']}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.4, color="rgba(15, 118, 110, 0.35)"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    for node in graph.nodes():
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        freq = int(frequencies.get(node, 1))
        node_text.append(f"{node}<br>Frequency: {freq}<br>Connections: {graph.degree(node)}")
        node_size.append(12 + min(freq, 60) * 0.45)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(graph.nodes()),
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=[graph.degree(node) for node in graph.nodes()],
            colorscale="Tealgrn",
            line=dict(width=1, color="white"),
            showscale=False,
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Keyword Co-occurrence Network",
        height=520,
        margin=dict(l=10, r=10, t=55, b=10),
        showlegend=False,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.92)",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})


def _plot_cluster_scatter(st, scatter_df: pd.DataFrame) -> None:
    if scatter_df.empty:
        st.info("Not enough text variety to cluster this filtered set.")
        return

    fig = px.scatter(
        scatter_df,
        x="x",
        y="y",
        color="cluster",
        symbol="section",
        hover_data={
            "x": False,
            "y": False,
            "cluster": True,
            "section": True,
            "orientation": True,
            "preview": True,
        },
        title="TF-IDF Chunk Clusters",
    )
    fig.update_traces(marker=dict(size=11, opacity=0.78, line=dict(width=0.7, color="white")))
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.92)",
        legend_title_text="Cluster",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})


def _plot_tone_heatmap(st, tone_df: pd.DataFrame, title: str) -> None:
    if tone_df.empty:
        st.info("No tone scores are available for this filtered set.")
        return

    matrix = tone_df.pivot_table(index="group", columns="tone", values="score", fill_value=0)
    fig = px.imshow(
        matrix,
        text_auto=".1f",
        color_continuous_scale="Tealgrn",
        aspect="auto",
        title=title,
    )
    fig.update_layout(
        height=max(360, 46 * len(matrix)),
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis_title="ESG language frame",
        yaxis_title="",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0.92)",
    )
    fig.update_traces(hovertemplate="Group: %{y}<br>Frame: %{x}<br>Terms per 1k tokens: %{z:.1f}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})


def _render_summary_panel(st, summary_items: list[dict], method_items: list[dict]) -> None:
    metric_html = "".join(
        f"""
        <div class="week9-metric">
            <div class="week9-label">{html.escape(item["label"])}</div>
            <div class="week9-value">{html.escape(str(item["value"]))}</div>
            <div class="week9-note">{html.escape(item["note"])}</div>
        </div>
        """
        for item in summary_items
    )
    method_html = "".join(
        f"""
        <div class="week9-method">
            <div class="week9-method-kicker">{html.escape(item["kicker"])}</div>
            <div class="week9-method-title">{html.escape(item["title"])}</div>
            <div class="week9-method-text">{html.escape(item["text"])}</div>
        </div>
        """
        for item in method_items
    )
    st.markdown(
        f"""
        <style>
        .week9-summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 0.5rem 0 1.05rem 0;
        }}
        .week9-metric {{
            background: rgba(255,255,255,0.94);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 12px;
            padding: 0.85rem 0.95rem;
            min-height: 118px;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.045);
        }}
        .week9-label, .week9-method-kicker {{
            color: #0f766e;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }}
        .week9-value {{
            color: #0f172a;
            font-size: 1.38rem;
            line-height: 1.12;
            font-weight: 850;
            overflow-wrap: anywhere;
            margin-bottom: 0.35rem;
        }}
        .week9-note {{
            color: #64748b;
            font-size: 0.88rem;
            line-height: 1.45;
        }}
        .week9-method-strip {{
            display: grid;
            grid-template-columns: 1.2fr 1.2fr 1fr;
            gap: 0.9rem;
            margin: 0.3rem 0 1rem 0;
        }}
        .week9-method {{
            background: linear-gradient(180deg, #f8fffe 0%, #ffffff 100%);
            border: 1px solid rgba(14, 165, 164, 0.20);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            min-height: 132px;
        }}
        .week9-method-title {{
            color: #0f172a;
            font-size: 1.02rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }}
        .week9-method-text {{
            color: #475569;
            font-size: 0.9rem;
            line-height: 1.58;
        }}
        @media (max-width: 900px) {{
            .week9-summary-grid, .week9-method-strip {{
                grid-template-columns: 1fr;
            }}
            .week9-metric, .week9-method {{
                min-height: auto;
            }}
        }}
        </style>
        <div class="week9-summary-grid">{metric_html}</div>
        <div class="week9-method-strip">{method_html}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_context_cards(st, df: pd.DataFrame, selected_keyword: str, max_cards: int = 3) -> None:
    if not selected_keyword:
        st.info("Choose a keyword to inspect supporting chunks.")
        return

    keyword = selected_keyword.lower()
    evidence = df[df["clean_text"].str.lower().str.contains(keyword, regex=False, na=False)].copy()
    if not evidence.empty:
        evidence["_artifact_score"] = evidence["raw_text"].map(_artifact_score)
        evidence = evidence.sort_values(["_artifact_score", "sdg_confidence"], ascending=[True, False]).head(max_cards)
    if evidence.empty:
        st.info("No supporting chunks found for the selected keyword in the current filter.")
        return

    for row in evidence.itertuples(index=False):
        text = _clean_evidence_text(
            getattr(row, "raw_text", ""),
            getattr(row, "clean_text", ""),
        )
        section = getattr(row, "section_label", "")
        orientation = getattr(row, "orientation", "")
        st.markdown(f"**{section.replace('_', ' ').title()} | {orientation.replace('_', ' ').title()}**")
        st.markdown(
            f"<p style='color:#6b7280;font-size:0.92rem;line-height:1.75;margin:0'>{html.escape(text)}</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")


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
        "Week 9 Q1",
        "Kiwi's Week9 Q1",
        "A standalone text-mining app page using phrase mining and keyword co-occurrence to reveal how ESG ideas travel together in TSMC's sustainability language.",
    )

    required_cols = {"clean_text", "section_label", "orientation", "sdg_labels"}
    missing_cols = sorted(required_cols - set(df_chunks.columns))
    if missing_cols:
        st.error(f"Missing required columns for Week 9 page: {', '.join(missing_cols)}")
        return

    all_sections = sorted([x for x in df_chunks["section_label"].dropna().unique() if str(x).strip()])
    all_orientations = sorted([x for x in df_chunks["orientation"].dropna().unique() if str(x).strip()])
    all_sdgs = sorted({label for labels in df_chunks["sdg_labels"].map(_parse_sdg_labels) for label in labels})

    st.markdown("### Controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        selected_sections_pretty = st.multiselect(
            "Strategic framing",
            options=[_prettify(x, section_label_map, sdg_label_map) for x in all_sections],
            default=[],
        )
        pretty_to_section = {_prettify(x, section_label_map, sdg_label_map): x for x in all_sections}
        selected_sections = [pretty_to_section[x] for x in selected_sections_pretty]
    with c2:
        selected_orientations = st.multiselect(
            "Orientation",
            options=all_orientations,
            default=[],
            format_func=lambda x: x.replace("_", " ").title(),
        )
    with c3:
        selected_sdgs_pretty = st.multiselect(
            "SDG theme",
            options=[_prettify(x, section_label_map, sdg_label_map) for x in all_sdgs],
            default=[],
        )
        pretty_to_sdg = {_prettify(x, section_label_map, sdg_label_map): x for x in all_sdgs}
        selected_sdgs = [pretty_to_sdg[x] for x in selected_sdgs_pretty]

    c4, c5, c6 = st.columns(3)
    with c4:
        ngram_label = st.radio("Phrase type", ["Bigrams", "Trigrams"], horizontal=True)
        ngram_size = 2 if ngram_label == "Bigrams" else 3
    with c5:
        top_n = st.slider("Top phrases / pairs", min_value=8, max_value=30, value=15, step=1)
    with c6:
        keyword_filter = st.text_input("Optional keyword filter", "")

    c7, c8 = st.columns([1, 1])
    with c7:
        cluster_count = st.slider("Chunk clusters", min_value=3, max_value=8, value=5, step=1)
    with c8:
        comparison_dimension = st.selectbox(
            "TF-IDF comparison group",
            ["Strategic framing", "SDG theme", "Orientation"],
        )

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

    phrase_df = _build_phrase_table(filtered_df["clean_text"], ngram_size=ngram_size, top_n=top_n)
    term_df, pair_df = _build_keyword_stats(filtered_df["clean_text"], top_terms=35, max_edges=45)
    tfidf_df, similarity_df = _build_tfidf_comparison(
        filtered_df,
        dimension=comparison_dimension,
        top_terms_per_group=6,
        section_label_map=section_label_map,
        sdg_label_map=sdg_label_map,
    )
    cluster_df, cluster_summary_df = _build_cluster_view(
        filtered_df,
        cluster_count=cluster_count,
        section_label_map=section_label_map,
        sdg_label_map=sdg_label_map,
    )
    tone_group_col = "section_label" if not selected_sdgs else "orientation"
    tone_df = _build_tone_table(
        filtered_df,
        group_col=tone_group_col,
        section_label_map=section_label_map,
        sdg_label_map=sdg_label_map,
    )

    top_phrase = phrase_df.iloc[0]["phrase"] if not phrase_df.empty else "N/A"
    strongest_pair = (
        f"{pair_df.iloc[0]['source']} + {pair_df.iloc[0]['target']}" if not pair_df.empty else "N/A"
    )
    most_connected = "N/A"
    if not pair_df.empty:
        degree_counts = Counter(pair_df["source"].tolist() + pair_df["target"].tolist())
        most_connected = degree_counts.most_common(1)[0][0]

    _render_summary_panel(
        st,
        [
            {"label": "Filtered Chunks", "value": len(filtered_df), "note": "Evidence units used in this page"},
            {"label": "Top Phrase", "value": top_phrase, "note": "Most repeated phrase after stopword removal"},
            {"label": "Strongest Pair", "value": strongest_pair, "note": "Keywords most often appearing together"},
            {"label": "TF-IDF Groups", "value": max(similarity_df.shape[0], 0), "note": "Comparison groups available in the current filter"},
        ],
        [
            {
                "kicker": "Technique",
                "title": "Phrase + Co-occurrence",
                "text": "Repeated phrases show the vocabulary surface; keyword pairs show which ESG ideas travel together.",
            },
            {
                "kicker": "Technique",
                "title": "TF-IDF + Similarity",
                "text": "Distinctive terms show what makes each group different; cosine similarity shows which groups speak alike.",
            },
            {
                "kicker": "Finding",
                "title": f"Network Hub: {most_connected}",
                "text": "The most connected keyword is a useful anchor for explaining the current filter's ESG language.",
            },
        ],
    )

    view_mode = st.selectbox(
        "Visualization",
        [
            "Phrase Mining",
            "Keyword Co-occurrence",
            "TF-IDF Terms",
            "Cosine Similarity",
            "Chunk Clusters",
            "ESG Tone",
            "Keyword Context",
        ],
        index=0,
    )

    if view_mode == "Phrase Mining":
        st.markdown("### Phrase Mining")
        st.caption("This view counts repeated two-word or three-word expressions in the filtered chunks.")
        _plot_phrase_bar(st, phrase_df, f"Top {ngram_label} in Filtered ESG Text")

    elif view_mode == "Keyword Co-occurrence":
        st.markdown("### Keyword Co-occurrence")
        left, right = st.columns([1.1, 1])
        with left:
            _plot_network(st, pair_df, term_df)
        with right:
            _plot_pair_heatmap(st, pair_df, top_pairs=top_n)

    elif view_mode == "TF-IDF Terms":
        st.markdown("### TF-IDF Group Comparison")
        st.caption(
            "This view highlights terms that are distinctive for each selected grouping, rather than merely frequent overall."
        )
        _plot_tfidf_terms(st, tfidf_df)
        if not tfidf_df.empty:
            st.markdown("**Top Terms Table**")
            st.dataframe(tfidf_df, use_container_width=True, hide_index=True)

    elif view_mode == "Cosine Similarity":
        st.markdown("### Cosine Similarity")
        st.caption("Groups with similar TF-IDF vocabularies receive higher similarity scores.")
        _plot_similarity_heatmap(st, similarity_df)

    elif view_mode == "Chunk Clusters":
        st.markdown("### TF-IDF Chunk Clustering")
        st.caption("Each point is one report chunk. Nearby points use similar vocabulary after TF-IDF vectorization.")
        _plot_cluster_scatter(st, cluster_df)
        if not cluster_summary_df.empty:
            st.markdown("**Cluster Interpretation Table**")
            st.dataframe(cluster_summary_df, use_container_width=True, hide_index=True)

    elif view_mode == "ESG Tone":
        st.markdown("### ESG Tone Heatmap")
        st.caption("Scores are lexicon hits per 1,000 tokens, grouped by section unless an SDG filter is active.")
        _plot_tone_heatmap(st, tone_df, "ESG Language Frames by Group")
        if not tone_df.empty:
            strongest_tone = tone_df.sort_values("score", ascending=False).head(1).iloc[0]
            st.info(
                f"Strongest frame under the current filters: {strongest_tone['group']} leans toward "
                f"{strongest_tone['tone']} language ({strongest_tone['score']} terms per 1,000 tokens)."
            )

    elif view_mode == "Keyword Context":
        st.markdown("### Keyword Context Explorer")
        keyword_options = term_df["term"].tolist()
        if not keyword_options:
            st.info("No keywords are available for this filter.")
            return

        selected_keyword = st.selectbox("Keyword", options=keyword_options, index=0)
        _render_context_cards(st, filtered_df, selected_keyword)
