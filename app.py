from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

try:
    from pyvis.network import Network
except ImportError:
    Network = None

try:
    from streamlit_echarts import st_echarts
except ImportError:
    st_echarts = None


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="TSMC ESG Word Cloud Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("TSMC Sustainability Direction Analysis")
st.caption("Interactive dashboard for section-based and orientation-based ESG language analysis")


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

CHUNKS_PATH = OUTPUT_DIR / "chunks_processed.csv"
SECTION_TFIDF_PATH = OUTPUT_DIR / "tfidf_by_section.csv"
ORIENTATION_TFIDF_PATH = OUTPUT_DIR / "tfidf_by_orientation.csv"
SDG_TFIDF_PATH = OUTPUT_DIR / "tfidf_by_sdg.csv"
ISSUE_FRAME_TFIDF_PATH = OUTPUT_DIR / "tfidf_by_issue_frame.csv"
SECTION_SIMILARITY_PATH = OUTPUT_DIR / "similarity_by_section.csv"
ORIENTATION_SIMILARITY_PATH = OUTPUT_DIR / "similarity_by_orientation.csv"
SDG_SIMILARITY_PATH = OUTPUT_DIR / "similarity_by_sdg.csv"
ISSUE_FRAME_DISTRIBUTION_PATH = OUTPUT_DIR / "issue_frame_distribution.csv"
ISSUE_FRAME_BY_SECTION_PATH = OUTPUT_DIR / "issue_frame_by_section.csv"
ISSUE_FRAME_COOCCURRENCE_PATH = OUTPUT_DIR / "issue_frame_cooccurrence.csv"


# =========================
# Constants
# =========================
SECTION_ORDER = ["environment", "talent", "supply_chain", "social", "governance"]
ORIENTATION_ORDER = ["action_oriented", "people_centric", "mixed"]
SDG_ORDER = ["SDG3_health", "SDG4_education", "SDG6_water", "SDG7_energy", "SDG8_labor", "SDG9_innovation", "SDG12_consumption", "SDG13_climate", "SDG17_partnership"]
ISSUE_FRAME_ORDER = ["environment", "labor_talent", "supply_chain", "innovation", "governance_risk", "other"]
PLOTLY_INTERACTIVE_CONFIG = {"scrollZoom": True, "displayModeBar": True}
WORDCLOUD_COLORS = ["#0f766e", "#0891b2", "#2563eb", "#059669", "#0ea5e9", "#14b8a6"]

SECTION_LABEL_MAP = {
    "environment": "Environment",
    "talent": "Talent",
    "supply_chain": "Supply Chain",
    "social": "Social",
    "governance": "Governance",
}

ORIENTATION_LABEL_MAP = {
    "action_oriented": "Action-Oriented",
    "people_centric": "People-Centric",
    "mixed": "Mixed",
}

SDG_LABEL_MAP = {
    "SDG3_health": "SDG 3: Health",
    "SDG4_education": "SDG 4: Education",
    "SDG6_water": "SDG 6: Water",
    "SDG7_energy": "SDG 7: Energy",
    "SDG8_labor": "SDG 8: Labor",
    "SDG9_innovation": "SDG 9: Innovation",
    "SDG12_consumption": "SDG 12: Consumption",
    "SDG13_climate": "SDG 13: Climate",
    "SDG17_partnership": "SDG 17: Partnership",
}

ISSUE_FRAME_LABEL_MAP = {
    "environment": "Environment",
    "labor_talent": "Labor / Talent",
    "supply_chain": "Supply Chain",
    "innovation": "Innovation",
    "governance_risk": "Governance / Risk",
    "other": "Other",
}

SDG_NARRATIVE = {
    "SDG17_partnership": (
        "**Partnership dominates because it has to.**\n\n"
        "SDG 17 has the highest chunk count and its top terms are *committee, carbon reduction, chemical* — "
        "this isn't goodwill language, it's compliance infrastructure. TSMC is building governance structures "
        "to push Scope 3 emissions and chemical accountability down to suppliers, driven by CSRD exposure "
        "from the Dresden fab."
    ),
    "SDG4_education": (
        "**Education coverage is inflated.**\n\n"
        "SDG 4 ranks highly but top terms *learning, senior, care* suggest significant overlap with talent "
        "retention and social welfare content. The keyword boundary is leaking — not all of these chunks "
        "are genuinely about education."
    ),
    "SDG12_consumption": (
        "**Consumption reflects manufacturing reality.**\n\n"
        "Top terms *chemical, recycle, acid* point to TSMC's fab-level material management: "
        "acid recovery, chemical reuse loops, and hazardous waste reduction — core to responsible "
        "semiconductor production at scale."
    ),
    "SDG13_climate": (
        "**Climate is the backbone of ESG reporting.**\n\n"
        "Top terms *carbon reduction, equivalent, electricity* reflect TSMC's RE100 commitment and "
        "Scope 2 decarbonization strategy. The high chunk count signals that climate disclosure "
        "is embedded across sections, not siloed."
    ),
    "SDG8_labor": (
        "**Labor language is precise and formal.**\n\n"
        "Top terms *permanent employee, intern, recruitment* indicate structured HR reporting. "
        "Coverage is driven by headcount disclosures and employment contract data, not narrative."
    ),
    "SDG9_innovation": (
        "**Innovation is IP-driven.**\n\n"
        "Top terms *patent, trade secret, patent application* show that TSMC frames innovation "
        "through intellectual property — not just R&D spend. This reflects a defensive IP strategy "
        "alongside process node advancement."
    ),
    "SDG3_health": (
        "**Health focus is safety-led.**\n\n"
        "Top terms *safety health, chemical, injury* indicate occupational health dominates over "
        "community health. Chemical exposure and injury prevention are the primary concerns in a fab environment."
    ),
    "SDG6_water": (
        "**Water and chemical management are intertwined.**\n\n"
        "Top terms *chemical, indicator, acid* reflect that water reporting in semiconductor fabs "
        "is inseparable from chemical discharge management — acidic wastewater treatment is the core challenge."
    ),
    "SDG7_energy": (
        "**Energy reporting centers on carbon.**\n\n"
        "Top terms *carbon reduction, electricity, charity* show decarbonization of electricity "
        "consumption is the primary energy narrative, aligned with TSMC's RE100 and net-zero targets."
    ),
}

SECTION_NARRATIVE = {
    "environment": "Environment language centers on operational resource management, especially water, energy, waste, and carbon.",
    "talent": "Talent language emphasizes employees, health, safety, and development, showing a people-and-workforce framing.",
    "supply_chain": "Supply chain language is compliance-heavy, with supplier controls, audits, and material-risk management standing out.",
    "social": "Social language highlights foundations, education, volunteering, and public-facing community programs.",
    "governance": "Governance language is structured around oversight, directors, committees, tax, and risk management.",
}

ORIENTATION_NARRATIVE = {
    "action_oriented": "Action-oriented chunks emphasize execution, controls, and measurable implementation through operational verbs and targets.",
    "people_centric": "People-centric chunks emphasize workforce safety, care, learning, and stakeholder-facing human impacts.",
    "mixed": "Mixed chunks combine implementation language with stakeholder or workforce language, often in cross-functional reporting sections.",
}

ISSUE_FRAME_NARRATIVE = {
    "environment": "This issue frame captures resource, emissions, water, and waste language that reflects the operational footprint of semiconductor production.",
    "labor_talent": "This issue frame captures workforce, safety, development, and human-rights language around employees and labor conditions.",
    "supply_chain": "This issue frame captures supplier oversight, procurement controls, audits, and third-party compliance language.",
    "innovation": "This issue frame captures patents, R&D, process technology, and other language about technical leadership and innovation capability.",
    "governance_risk": "This issue frame captures risk, board oversight, policy, tax, security, and accountability language.",
    "other": "This issue frame groups chunks with weak or mixed signals that do not clearly fit one of the main issue categories.",
}


# =========================
# Load data
# =========================
@st.cache_data
def load_data():
    df_chunks = pd.read_csv(CHUNKS_PATH)
    df_section = pd.read_csv(SECTION_TFIDF_PATH)
    df_orientation = pd.read_csv(ORIENTATION_TFIDF_PATH)
    df_sdg = pd.read_csv(SDG_TFIDF_PATH)
    df_issue_frame = pd.read_csv(ISSUE_FRAME_TFIDF_PATH)
    df_section_similarity = pd.read_csv(SECTION_SIMILARITY_PATH, index_col=0)
    df_orientation_similarity = pd.read_csv(ORIENTATION_SIMILARITY_PATH, index_col=0)
    df_sdg_similarity = pd.read_csv(SDG_SIMILARITY_PATH, index_col=0)
    df_issue_frame_distribution = pd.read_csv(ISSUE_FRAME_DISTRIBUTION_PATH)
    df_issue_frame_by_section = pd.read_csv(ISSUE_FRAME_BY_SECTION_PATH)
    df_issue_frame_cooccurrence = pd.read_csv(ISSUE_FRAME_COOCCURRENCE_PATH)

    for col in ["raw_text", "clean_text", "section_label", "orientation", "sdg_labels", "issue_frame"]:
        if col in df_chunks.columns:
            df_chunks[col] = df_chunks[col].fillna("").astype(str)
    if "sdg_confidence" in df_chunks.columns:
        df_chunks["sdg_confidence"] = pd.to_numeric(df_chunks["sdg_confidence"], errors="coerce").fillna(0.0)
    if "issue_frame_confidence" in df_chunks.columns:
        df_chunks["issue_frame_confidence"] = pd.to_numeric(df_chunks["issue_frame_confidence"], errors="coerce").fillna(0.0)

    return (
        df_chunks,
        df_section,
        df_orientation,
        df_sdg,
        df_issue_frame,
        df_section_similarity,
        df_orientation_similarity,
        df_sdg_similarity,
        df_issue_frame_distribution,
        df_issue_frame_by_section,
        df_issue_frame_cooccurrence,
    )


(
    df_chunks,
    df_section_tfidf,
    df_orientation_tfidf,
    df_sdg_tfidf,
    df_issue_frame_tfidf,
    df_section_similarity,
    df_orientation_similarity,
    df_sdg_similarity,
    df_issue_frame_distribution,
    df_issue_frame_by_section,
    df_issue_frame_cooccurrence,
) = load_data()


# =========================
# Helper functions
# =========================
def prettify_label(x: str) -> str:
    if x in SECTION_LABEL_MAP:
        return SECTION_LABEL_MAP[x]
    if x in ORIENTATION_LABEL_MAP:
        return ORIENTATION_LABEL_MAP[x]
    if x in SDG_LABEL_MAP:
        return SDG_LABEL_MAP[x]
    if x in ISSUE_FRAME_LABEL_MAP:
        return ISSUE_FRAME_LABEL_MAP[x]
    return x.replace("_", " ").title()


def parse_sdg_labels(label_text: str) -> list[str]:
    if not label_text:
        return []
    return [label.strip() for label in str(label_text).split(",") if label.strip() and label.strip() != "unclassified"]


def explode_sdg_chunks(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(sdg_labels=df["sdg_labels"].map(parse_sdg_labels))
        .explode("sdg_labels")
        .dropna(subset=["sdg_labels"])
    )


def filter_chunks_by_sdg(df: pd.DataFrame, sdg: str) -> pd.DataFrame:
    return df[df["sdg_labels"].map(parse_sdg_labels).map(lambda labels: sdg in labels)].copy()


def build_issue_frame_section_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    heatmap_df = (
        df.pivot(index="issue_frame", columns="section_label", values="share")
        .fillna(0.0)
    )
    ordered_index = [label for label in ISSUE_FRAME_ORDER if label in heatmap_df.index]
    ordered_cols = [label for label in SECTION_ORDER if label in heatmap_df.columns]
    return heatmap_df.reindex(index=ordered_index, columns=ordered_cols).fillna(0.0)


def make_wordcloud_from_tfidf(
    df: pd.DataFrame,
    term_col: str = "term",
    score_col: str = "tfidf_score",
):
    freq = {row[term_col]: float(row[score_col]) for _, row in df.iterrows()}
    if not freq:
        return None

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        collocations=False,
        max_words=100
    ).generate_from_frequencies(freq)
    return wc


def plot_wordcloud(wc: WordCloud):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)


def render_interactive_wordcloud(df: pd.DataFrame, top_k: int = 60):
    if df.empty:
        st.info("No terms available.")
        return

    if st_echarts is None:
        wc = make_wordcloud_from_tfidf(df.head(80))
        if wc:
            plot_wordcloud(wc)
        else:
            st.info("No terms available.")
        return

    data = (
        df.sort_values("tfidf_score", ascending=False)
        .head(top_k)[["term", "tfidf_score"]]
        .copy()
    )
    max_score = float(data["tfidf_score"].max()) if not data.empty else 1.0
    min_score = float(data["tfidf_score"].min()) if not data.empty else 0.0
    score_span = max(max_score - min_score, 1e-9)
    word_data = [
        {
            "name": row["term"],
            "value": round(28 + (float(row["tfidf_score"]) - min_score) * 92 / score_span, 2),
            "textStyle": {"color": color},
        }
        for (_, row), color in zip(
            data.iterrows(),
            WORDCLOUD_COLORS * 20,
        )
    ]

    options = {
        "tooltip": {"show": True},
        "series": [
            {
                "type": "wordCloud",
                "shape": "circle",
                "keepAspect": True,
                "left": "center",
                "top": "center",
                "width": "100%",
                "height": 420,
                "sizeRange": [18, 72],
                "rotationRange": [-45, 45],
                "rotationStep": 15,
                "gridSize": 10,
                "drawOutOfBound": False,
                "textStyle": {
                    "fontFamily": "Arial",
                    "fontWeight": "bold",
                },
                "emphasis": {
                    "focus": "self",
                    "textStyle": {
                        "shadowBlur": 8,
                        "shadowColor": "#94a3b8",
                    },
                },
                "data": word_data,
            }
        ],
    }
    st_echarts(options=options, height="440px")


def plot_top_terms(df: pd.DataFrame, title: str, top_k: int = 15):
    df_plot = df.sort_values("tfidf_score", ascending=False).head(top_k).copy()

    fig = px.bar(
        df_plot.iloc[::-1],
        x="tfidf_score",
        y="term",
        orientation="h",
        title=title
    )
    fig.update_layout(
        height=500,
        xaxis_title="TF-IDF Score",
        yaxis_title="Term",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_traces(
        hovertemplate="Term: %{y}<br>TF-IDF: %{x:.4f}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_INTERACTIVE_CONFIG)


def get_top_terms_table(df: pd.DataFrame, top_k: int = 20):
    out = (
        df.sort_values("tfidf_score", ascending=False)
        .head(top_k)[["term", "tfidf_score"]]
        .reset_index(drop=True)
        .copy()
    )
    out["tfidf_score"] = out["tfidf_score"].round(4)
    return out


def get_top_terms_list(df: pd.DataFrame, top_k: int = 8) -> list[str]:
    if df.empty:
        return []
    return (
        df.sort_values("tfidf_score", ascending=False)
        .head(top_k)["term"]
        .astype(str)
        .tolist()
    )


def aggregate_terms_for_overview(*dfs: pd.DataFrame, top_k: int = 60) -> pd.DataFrame:
    valid_frames = [df[["term", "tfidf_score"]].copy() for df in dfs if not df.empty]
    if not valid_frames:
        return pd.DataFrame(columns=["term", "tfidf_score"])

    combined = pd.concat(valid_frames, ignore_index=True)
    aggregated = (
        combined.groupby("term", as_index=False)["tfidf_score"]
        .mean()
        .sort_values("tfidf_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    return aggregated


def chunk_match_count(raw_text: str, terms: list[str]) -> int:
    text = str(raw_text).lower()
    return sum(1 for term in terms if term.lower() in text)


def rank_representative_chunks(
    df: pd.DataFrame,
    df_terms: pd.DataFrame,
    max_chunks: int,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    ranked = df.copy()
    top_terms = get_top_terms_list(df_terms, top_k=8)
    ranked["token_count"] = ranked["clean_text"].fillna("").str.split().str.len()
    ranked["term_matches"] = ranked["raw_text"].fillna("").map(lambda text: chunk_match_count(text, top_terms))
    if "sdg_confidence" not in ranked.columns:
        ranked["sdg_confidence"] = 0.0

    ranked = ranked.sort_values(
        ["term_matches", "sdg_confidence", "token_count"],
        ascending=[False, False, False],
    )
    return ranked.head(max_chunks).copy()


def render_interpretation(title: str, text: str):
    st.markdown(f"**Interpretation: {title}**")
    st.caption(text)


def render_chunk_cards(df: pd.DataFrame, df_terms: pd.DataFrame, max_chunks: int = 5):
    view_df = df.copy()
    view_df = rank_representative_chunks(view_df, df_terms, max_chunks=max_chunks)
    top_terms = get_top_terms_list(df_terms, top_k=8)

    for i, (_, row) in enumerate(view_df.iterrows(), start=1):
        label_section = prettify_label(row["section_label"])
        label_orientation = prettify_label(row["orientation"])
        label_SDGs = ", ".join(prettify_label(label) for label in parse_sdg_labels(row.get("sdg_labels", "")))
        confidence = float(row.get("sdg_confidence", 0.0))
        confidence_text = f" | SDG confidence {confidence:.2f}" if confidence > 0 else ""
        token_count = int(row.get("token_count", 0))
        term_matches = int(row.get("term_matches", 0))
        matched_terms = [term for term in top_terms if term.lower() in str(row["raw_text"]).lower()][:4]
        evidence_text = ", ".join(matched_terms) if matched_terms else "No top-term overlap"

        with st.expander(
            f"Chunk {i} | {label_section} | {label_orientation} | {label_SDGs}{confidence_text}",
            expanded=False
        ):
            st.caption(f"Evidence: {term_matches} top-term matches | {token_count} clean tokens | {evidence_text}")
            st.write(row["raw_text"])


def make_frequency_dict(df: pd.DataFrame, group_col: str, ordered_groups: list[str], top_n: int = 20):
    out = {}
    for group in ordered_groups:
        sub = (
            df[df[group_col] == group]
            .sort_values("tfidf_score", ascending=False)
            .head(top_n)
        )
        if len(sub) > 0:
            out[group] = dict(zip(sub["term"], sub["tfidf_score"]))
    return out


def build_overlap_heatmap(df: pd.DataFrame, group_col: str, ordered_groups: list[str], top_n: int = 20):
    freq_dict = make_frequency_dict(df, group_col=group_col, ordered_groups=ordered_groups, top_n=top_n)
    groups = list(freq_dict.keys())

    matrix = []
    for g1 in groups:
        row = []
        set1 = set(freq_dict[g1].keys())
        for g2 in groups:
            set2 = set(freq_dict[g2].keys())
            overlap = len(set1 & set2)
            row.append(overlap)
        matrix.append(row)

    return pd.DataFrame(matrix, index=groups, columns=groups)


def plot_heatmap(heatmap_df: pd.DataFrame, title: str):
    pretty_index = [prettify_label(x) for x in heatmap_df.index]
    pretty_cols = [prettify_label(x) for x in heatmap_df.columns]

    fig = px.imshow(
        heatmap_df.values,
        x=pretty_cols,
        y=pretty_index,
        text_auto=True,
        aspect="auto",
        title=title,
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        height=480,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="",
        yaxis_title=""
    )
    fig.update_traces(
        hovertemplate="X: %{x}<br>Y: %{y}<br>Value: %{z}<extra></extra>"
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOTLY_INTERACTIVE_CONFIG,
    )


def summarize_similarity_pairs(similarity_df: pd.DataFrame, top_n: int = 3, ascending: bool = False) -> pd.DataFrame:
    if similarity_df.empty:
        return pd.DataFrame(columns=["group_a", "group_b", "similarity"])

    records = []
    labels = similarity_df.index.tolist()
    for i, label_a in enumerate(labels):
        for j, label_b in enumerate(labels):
            if j <= i:
                continue
            records.append(
                {
                    "group_a": label_a,
                    "group_b": label_b,
                    "similarity": float(similarity_df.iloc[i, j]),
                }
            )

    if not records:
        return pd.DataFrame(columns=["group_a", "group_b", "similarity"])

    out = pd.DataFrame(records).sort_values("similarity", ascending=ascending).head(top_n).copy()
    out["group_a"] = out["group_a"].map(prettify_label)
    out["group_b"] = out["group_b"].map(prettify_label)
    out["similarity"] = out["similarity"].round(3)
    return out.reset_index(drop=True)


def plot_distribution_bar(df: pd.DataFrame, col: str, title: str):
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    counts[col] = counts[col].map(prettify_label)
    counts = counts.sort_values("count", ascending=False).reset_index(drop=True)

    fig = px.bar(
        counts,
        x=col,
        y="count",
        title=title,
        text="count",
        color="count",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="",
        yaxis_title="Count",
        coloraxis_showscale=False,
    )
    fig.update_traces(
        hovertemplate="Label: %{x}<br>Count: %{y}<extra></extra>"
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOTLY_INTERACTIVE_CONFIG,
    )


def build_issue_frame_graph(network_df: pd.DataFrame) -> nx.Graph:
    return nx.from_pandas_edgelist(
        network_df,
        source="term_a",
        target="term_b",
        edge_attr="weight",
        create_using=nx.Graph(),
    )


def render_issue_frame_network_tables(graph: nx.Graph, network_df: pd.DataFrame) -> None:
    weighted_degree = dict(graph.degree(weight="weight"))
    degree_centrality = nx.degree_centrality(graph)

    edge_table = network_df.sort_values("weight", ascending=False).copy()
    edge_table["term_a"] = edge_table["term_a"].astype(str)
    edge_table["term_b"] = edge_table["term_b"].astype(str)
    st.caption("Top co-occurring keyword pairs within this issue frame.")
    st.dataframe(edge_table.head(10), use_container_width=True, hide_index=True)

    centrality_table = (
        pd.DataFrame(
            {
                "term": list(graph.nodes()),
                "connected_weight": [round(weighted_degree[node], 2) for node in graph.nodes()],
                "degree_centrality": [round(degree_centrality[node], 3) for node in graph.nodes()],
            }
        )
        .sort_values(["connected_weight", "degree_centrality"], ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    st.caption("Most central terms in this issue-frame network.")
    st.dataframe(centrality_table, use_container_width=True, hide_index=True)


def plot_issue_frame_cooccurrence_network(
    df_edges: pd.DataFrame,
    issue_frame: str,
    title: str,
):
    network_df = df_edges[df_edges["issue_frame"] == issue_frame].copy()
    if network_df.empty:
        st.info("No co-occurrence edges available for this issue frame.")
        return

    graph = build_issue_frame_graph(network_df)
    if graph.number_of_nodes() < 2:
        st.info("Not enough connected terms to render a network.")
        return

    positions = nx.spring_layout(graph, weight="weight", seed=42, k=0.9 / math.sqrt(graph.number_of_nodes()))
    weighted_degree = dict(graph.degree(weight="weight"))
    degree_centrality = nx.degree_centrality(graph)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_text_x: list[float] = []
    edge_text_y: list[float] = []
    edge_text: list[str] = []

    max_weight = max(float(attrs.get("weight", 0.0)) for _, _, attrs in graph.edges(data=True))
    for term_a, term_b, attrs in graph.edges(data=True):
        weight = float(attrs.get("weight", 0.0))
        x0, y0 = positions[term_a]
        x1, y1 = positions[term_b]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text_x.append((x0 + x1) / 2)
        edge_text_y.append((y0 + y1) / 2)
        edge_text.append(f"{term_a} ↔ {term_b} | weight {int(weight)}")

    fig = go.Figure()
    for term_a, term_b, attrs in graph.edges(data=True):
        weight = float(attrs.get("weight", 0.0))
        x0, y0 = positions[term_a]
        x1, y1 = positions[term_b]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="#94a3b8", width=1.5 + (4.0 * weight / max_weight)),
                hovertemplate=f"{term_a} ↔ {term_b} | weight {int(weight)}<extra></extra>",
                showlegend=False,
            )
        )

    edge_label_trace = go.Scatter(
        x=edge_text_x,
        y=edge_text_y,
        mode="markers",
        marker=dict(size=8, color="rgba(0,0,0,0)"),
        text=edge_text,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )
    fig.add_trace(edge_label_trace)

    nodes = sorted(graph.nodes())
    node_x = [positions[node][0] for node in nodes]
    node_y = [positions[node][1] for node in nodes]
    node_sizes = [24 + (weighted_degree[node] * 3.0) for node in nodes]
    node_labels = [
        f"{node}<br>connected weight {weighted_degree[node]:.0f}<br>centrality {degree_centrality[node]:.2f}"
        for node in nodes
    ]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=nodes,
        textposition="top center",
        hovertext=node_labels,
        hovertemplate="%{hovertext}<extra></extra>",
        textfont=dict(size=14, color="#0f172a", family="Arial Black, Arial, sans-serif"),
        marker=dict(
            size=node_sizes,
            color="#a7f3d0",
            line=dict(color="#0f766e", width=2),
            opacity=0.98,
        ),
        showlegend=False,
    )
    fig.add_trace(node_trace)

    fig.update_layout(
        title=title,
        height=560,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        dragmode="pan",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
        },
    )

    edge_table = network_df.sort_values("weight", ascending=False).copy()
    edge_table["term_a"] = edge_table["term_a"].astype(str)
    edge_table["term_b"] = edge_table["term_b"].astype(str)
    st.caption("Top co-occurring keyword pairs within this issue frame.")
    st.dataframe(edge_table.head(10), use_container_width=True, hide_index=True)

    centrality_table = (
        pd.DataFrame(
            {
                "term": list(graph.nodes()),
                "connected_weight": [round(weighted_degree[node], 2) for node in graph.nodes()],
                "degree_centrality": [round(degree_centrality[node], 3) for node in graph.nodes()],
            }
        )
        .sort_values(["connected_weight", "degree_centrality"], ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    st.caption("Most central terms in this issue-frame network.")
    st.dataframe(centrality_table, use_container_width=True, hide_index=True)


def render_draggable_issue_frame_network(
    df_edges: pd.DataFrame,
    issue_frame: str,
    title: str,
):
    network_df = df_edges[df_edges["issue_frame"] == issue_frame].copy()
    if network_df.empty:
        st.info("No co-occurrence edges available for this issue frame.")
        return

    graph = build_issue_frame_graph(network_df)
    if graph.number_of_nodes() < 2:
        st.info("Not enough connected terms to render a network.")
        return

    if Network is None:
        st.info("Install `pyvis` for draggable nodes. Showing the Plotly fallback for now.")
        plot_issue_frame_cooccurrence_network(df_edges, issue_frame, title)
        return

    weighted_degree = dict(graph.degree(weight="weight"))
    degree_centrality = nx.degree_centrality(graph)
    max_weight = max(float(attrs.get("weight", 0.0)) for _, _, attrs in graph.edges(data=True))

    net = Network(
        height="620px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#0f172a",
        notebook=False,
    )
    net.barnes_hut(
        gravity=-2500,
        central_gravity=0.2,
        spring_length=180,
        spring_strength=0.04,
        damping=0.9,
    )
    net.set_options(
        """
        const options = {
          "nodes": {
            "shape": "dot",
            "font": {
              "size": 18,
              "face": "Arial",
              "color": "#0f172a",
              "strokeWidth": 0
            },
            "borderWidth": 2,
            "borderWidthSelected": 3
          },
          "edges": {
            "color": {
              "color": "#94a3b8",
              "highlight": "#475569"
            },
            "smooth": {
              "type": "dynamic"
            }
          },
          "interaction": {
            "hover": true,
            "dragNodes": true,
            "dragView": true,
            "zoomView": true,
            "navigationButtons": true
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "iterations": 150
            }
          }
        }
        """
    )

    for node in graph.nodes():
        connected_weight = float(weighted_degree[node])
        centrality = float(degree_centrality[node])
        net.add_node(
            node,
            label=node,
            title=(
                f"{node}<br>"
                f"connected weight: {connected_weight:.0f}<br>"
                f"centrality: {centrality:.2f}"
            ),
            size=16 + (connected_weight * 1.8),
            color={
                "background": "#a7f3d0",
                "border": "#0f766e",
                "highlight": {"background": "#d1fae5", "border": "#115e59"},
            },
        )

    for term_a, term_b, attrs in graph.edges(data=True):
        weight = float(attrs.get("weight", 0.0))
        net.add_edge(
            term_a,
            term_b,
            value=weight,
            width=1.5 + (4.0 * weight / max_weight),
            title=f"{term_a} ↔ {term_b} | weight {int(weight)}",
        )

    st.markdown(f"**{title}**")
    components.html(net.generate_html(), height=640, scrolling=False)

    edge_table = network_df.sort_values("weight", ascending=False).copy()
    edge_table["term_a"] = edge_table["term_a"].astype(str)
    edge_table["term_b"] = edge_table["term_b"].astype(str)
    st.caption("Top co-occurring keyword pairs within this issue frame.")
    st.dataframe(edge_table.head(10), use_container_width=True, hide_index=True)

    centrality_table = (
        pd.DataFrame(
            {
                "term": list(graph.nodes()),
                "connected_weight": [round(weighted_degree[node], 2) for node in graph.nodes()],
                "degree_centrality": [round(degree_centrality[node], 3) for node in graph.nodes()],
            }
        )
        .sort_values(["connected_weight", "degree_centrality"], ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    st.caption("Most central terms in this issue-frame network.")
    st.dataframe(centrality_table, use_container_width=True, hide_index=True)


_SDG9_EXCLUSIVE_TERMS = {"trade_secret", "trade_secret registration", "patent application", "patent"}

def get_top_insight(df_terms: pd.DataFrame, label_col: str, top_k: int = 3):
    insights = []
    for label in df_terms[label_col].unique():
        sub = df_terms[df_terms[label_col] == label].sort_values("tfidf_score", ascending=False)
        if label_col == "sdg_labels" and label != "SDG9_innovation":
            sub = sub[~sub["term"].isin(_SDG9_EXCLUSIVE_TERMS)]
        sub = sub.head(top_k)
        terms = ", ".join(sub["term"].tolist())
        insights.append((prettify_label(label), terms))
    return insights


def render_hero_summary():
    st.markdown(
        """
        ### Executive Summary
        This dashboard reads TSMC's sustainability report as a language system.
        It highlights how ESG communication differs by **section** and by **narrative orientation**,
        especially between **Action-Oriented** and **People-Centric** language.
        """
    )

    sec_insights = get_top_insight(df_section_tfidf, "section_label", top_k=3)
    ori_insights = get_top_insight(df_orientation_tfidf, "orientation", top_k=3)
    SDG_insights = get_top_insight(df_sdg_tfidf, "sdg_labels", top_k=3)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Section highlights**")
        for name, terms in sec_insights:
            st.markdown(f"- **{name}**: {terms}")

    with col2:
        st.markdown("**Orientation highlights**")
        for name, terms in ori_insights:
            st.markdown(f"- **{name}**: {terms}")

    with col3:
        st.markdown("**SDG highlights**")
        for name, terms in SDG_insights:
            st.markdown(f"- **{name}**: {terms}")


def filter_chunks_by_keyword(df: pd.DataFrame, keyword: str):
    if not keyword.strip():
        return df.copy()

    kw = keyword.strip().lower()
    return df[df["raw_text"].str.lower().str.contains(kw, na=False, regex=False)].copy()


# =========================
# Sidebar
# =========================
st.sidebar.header("Controls")

page_mode = st.sidebar.selectbox(
    "Page",
    ["Overview", "Explorer"]
)

view_mode = st.sidebar.radio(
    "Explorer mode",
    ["Section view", "Orientation view", "SDG view", "Issue Frame view"]
)

top_k = st.sidebar.slider("Top keywords", min_value=10, max_value=30, value=15, step=5)
show_chunks = st.sidebar.checkbox("Show example chunks", value=True)
max_chunks = st.sidebar.slider("Number of chunks to show", min_value=3, max_value=10, value=5)

st.sidebar.markdown("---")
keyword_search = st.sidebar.text_input("Keyword search in raw text", "")

filtered_chunks = filter_chunks_by_keyword(df_chunks, keyword_search)


# =========================
# Overview Page
# =========================
if page_mode == "Overview":
    render_hero_summary()
    st.markdown("")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Chunks", len(df_chunks))
    c2.metric("Sections", df_chunks["section_label"].nunique())
    c3.metric("Orientations", df_chunks["orientation"].nunique())
    c4.metric("SDG Themes", df_chunks["sdg_labels"].str.split(",").explode().str.strip().pipe(lambda s: s[s != "unclassified"]).nunique())
    c5.metric("Filtered Chunks", len(filtered_chunks))

    st.markdown("### Quick Insights")

    insight_col1, insight_col2, insight_col3 = st.columns(3)

    top_env = (
        df_section_tfidf[df_section_tfidf["section_label"] == "environment"]
        .sort_values("tfidf_score", ascending=False)
        .head(3)["term"]
        .tolist()
    )
    top_talent = (
        df_section_tfidf[df_section_tfidf["section_label"] == "talent"]
        .sort_values("tfidf_score", ascending=False)
        .head(3)["term"]
        .tolist()
    )
    top_action = (
        df_orientation_tfidf[df_orientation_tfidf["orientation"] == "action_oriented"]
        .sort_values("tfidf_score", ascending=False)
        .head(3)["term"]
        .tolist()
    )

    with insight_col1:
        st.info("**Environment focus**  \n" + ", ".join(top_env))

    with insight_col2:
        st.info("**Talent focus**  \n" + ", ".join(top_talent))

    with insight_col3:
        st.info("**Action-oriented focus**  \n" + ", ".join(top_action))

    # Top 2 SDGs by chunk count
    df_chunks_sdg_exploded = explode_sdg_chunks(df_chunks)
    sdg_counts = df_chunks_sdg_exploded["sdg_labels"].value_counts()
    top2_sdgs = sdg_counts.head(2).index.tolist()

    sdg_focus_col1, sdg_focus_col2 = st.columns(2)
    for col, sdg in zip([sdg_focus_col1, sdg_focus_col2], top2_sdgs):
        top_terms = (
            df_sdg_tfidf[df_sdg_tfidf["sdg_labels"] == sdg]
            .sort_values("tfidf_score", ascending=False)
            .head(3)["term"]
            .tolist()
        )
        label = SDG_LABEL_MAP.get(sdg, sdg)
        count = int(sdg_counts.get(sdg, 0))
        narrative = SDG_NARRATIVE.get(sdg, "")
        with col:
            with st.expander(f"**{label} focus** — {', '.join(top_terms)} ({count} chunks)"):
                if narrative:
                    st.markdown(narrative)

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        plot_distribution_bar(df_chunks, "section_label", "Section Distribution")

    with row1_col2:
        plot_distribution_bar(df_chunks, "orientation", "Orientation Distribution")

    plot_distribution_bar(df_chunks_sdg_exploded, "sdg_labels", "SDG Distribution")

    st.markdown("### Overview Word Cloud")
    overview_wordcloud_terms = aggregate_terms_for_overview(
        df_section_tfidf,
        df_orientation_tfidf,
        df_sdg_tfidf,
        top_k=70,
    )
    st.caption("Interactive overview cloud built from the strongest terms across sections, orientations, and SDGs.")
    render_interactive_wordcloud(overview_wordcloud_terms, top_k=70)

    st.markdown("### Evidence Snapshot")
    overview_terms = df_orientation_tfidf[df_orientation_tfidf["orientation"] == "action_oriented"].copy()
    render_interpretation(
        "Overview",
        "These example chunks are ranked by overlap with top action-oriented terms, then by SDG confidence and chunk richness.",
    )
    render_chunk_cards(filtered_chunks, overview_terms, max_chunks=min(max_chunks, 3))

    st.markdown("### Issue Framing Layer")
    render_interpretation(
        "Issue Framing",
        "This layer groups report language into broad strategic issue categories so you can explain whether TSMC emphasizes environment, labor and talent, supply chain controls, innovation, or governance and risk.",
    )

    issue_col1, issue_col2 = st.columns(2)
    with issue_col1:
        plot_distribution_bar(df_chunks, "issue_frame", "Dominant Issue Frame Distribution")
    with issue_col2:
        top_issue_frames = (
            df_issue_frame_distribution.sort_values("count", ascending=False)
            .head(3)
            .copy()
        )
        top_issue_frames["issue_frame"] = top_issue_frames["issue_frame"].map(prettify_label)
        top_issue_frames["share"] = (top_issue_frames["share"] * 100).round(1)
        st.markdown("**Top issue frames**")
        st.dataframe(top_issue_frames, use_container_width=True, hide_index=True)

    issue_heatmap = build_issue_frame_section_heatmap(df_issue_frame_by_section)
    plot_heatmap(issue_heatmap.round(3), "Issue Frame Share Across ESG Sections")

    st.markdown("### Cross-Section Theme Overlap")
    heatmap_section = build_overlap_heatmap(
        df_section_tfidf,
        group_col="section_label",
        ordered_groups=SECTION_ORDER,
        top_n=20
    )
    plot_heatmap(heatmap_section, "Top-Term Overlap Across ESG Sections")

    st.markdown("### Cross-Orientation Theme Overlap")
    heatmap_orientation = build_overlap_heatmap(
        df_orientation_tfidf,
        group_col="orientation",
        ordered_groups=ORIENTATION_ORDER,
        top_n=20
    )
    plot_heatmap(heatmap_orientation, "Top-Term Overlap Across Narrative Orientations")

    st.markdown("### Cross-SDG Theme Overlap")
    heatmap_sdg = build_overlap_heatmap(
        df_sdg_tfidf,
        group_col="sdg_labels",
        ordered_groups=SDG_ORDER,
        top_n=20
    )
    plot_heatmap(heatmap_sdg, "Top-Term Overlap Across SDGs")

    st.markdown("### Cosine Similarity")
    render_interpretation(
        "Cosine Similarity",
        "These heatmaps compare grouped language using TF-IDF vectors. Higher similarity means two categories rely on more similar vocabulary patterns overall.",
    )

    similarity_col1, similarity_col2 = st.columns(2)
    with similarity_col1:
        plot_heatmap(df_section_similarity, "Cosine Similarity Across ESG Sections")
    with similarity_col2:
        st.markdown("**Most similar section pairs**")
        st.dataframe(
            summarize_similarity_pairs(df_section_similarity, top_n=3, ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Most distinct section pairs**")
        st.dataframe(
            summarize_similarity_pairs(df_section_similarity, top_n=3, ascending=True),
            use_container_width=True,
            hide_index=True,
        )

    similarity_col3, similarity_col4 = st.columns(2)
    with similarity_col3:
        plot_heatmap(df_orientation_similarity, "Cosine Similarity Across Orientations")
    with similarity_col4:
        st.markdown("**Most similar orientation pairs**")
        st.dataframe(
            summarize_similarity_pairs(df_orientation_similarity, top_n=3, ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Most distinct orientation pairs**")
        st.dataframe(
            summarize_similarity_pairs(df_orientation_similarity, top_n=3, ascending=True),
            use_container_width=True,
            hide_index=True,
        )

    similarity_col5, similarity_col6 = st.columns(2)
    with similarity_col5:
        plot_heatmap(df_sdg_similarity, "Cosine Similarity Across SDGs")
    with similarity_col6:
        st.markdown("**Most similar SDG pairs**")
        st.dataframe(
            summarize_similarity_pairs(df_sdg_similarity, top_n=3, ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**Most distinct SDG pairs**")
        st.dataframe(
            summarize_similarity_pairs(df_sdg_similarity, top_n=3, ascending=True),
            use_container_width=True,
            hide_index=True,
        )

# =========================
# Explorer Page
# =========================
else:
    st.markdown("## Explorer")
    st.caption("Browse ESG language by section or by narrative orientation.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Chunks", len(df_chunks))
    c2.metric("Sections", df_chunks["section_label"].nunique())
    c3.metric("Orientations", df_chunks["orientation"].nunique())
    c4.metric("Filtered Chunks", len(filtered_chunks))

    if view_mode == "Section view":
        tabs = st.tabs([SECTION_LABEL_MAP.get(x, x) for x in SECTION_ORDER])

        for tab, section in zip(tabs, SECTION_ORDER):
            with tab:
                df_terms = df_section_tfidf[df_section_tfidf["section_label"] == section].copy()
                df_chunk_section = filtered_chunks[filtered_chunks["section_label"] == section].copy()

                count_section = len(df_chunks[df_chunks["section_label"] == section])
                st.subheader(f"{SECTION_LABEL_MAP.get(section, section)} Analysis ({count_section} chunks)")
                render_interpretation(
                    SECTION_LABEL_MAP.get(section, section),
                    SECTION_NARRATIVE.get(section, "This section summarizes the dominant language and evidence for this category."),
                )

                left, right = st.columns([1.2, 1])

                with left:
                    st.markdown("**Word Cloud**")
                    render_interactive_wordcloud(df_terms.head(80))

                with right:
                    st.markdown("**Top Keywords**")
                    plot_top_terms(
                        df_terms,
                        title=f"Top TF-IDF Terms: {SECTION_LABEL_MAP.get(section, section)}",
                        top_k=top_k
                    )

                st.markdown("**Top Keywords Table**")
                st.dataframe(
                    get_top_terms_table(df_terms, top_k=top_k),
                    use_container_width=True,
                    hide_index=True
                )

                if show_chunks:
                    st.markdown("**Representative Chunks**")
                    if len(df_chunk_section) == 0:
                        st.info("No chunks match the current filter.")
                    else:
                        render_chunk_cards(df_chunk_section, df_terms, max_chunks=max_chunks)

    elif view_mode == "Orientation view":
        tabs = st.tabs([ORIENTATION_LABEL_MAP.get(x, x) for x in ORIENTATION_ORDER])

        for tab, orientation in zip(tabs, ORIENTATION_ORDER):
            with tab:
                df_terms = df_orientation_tfidf[df_orientation_tfidf["orientation"] == orientation].copy()
                df_chunk_ori = filtered_chunks[filtered_chunks["orientation"] == orientation].copy()

                count_orientation = len(df_chunks[df_chunks["orientation"] == orientation])
                st.subheader(f"{ORIENTATION_LABEL_MAP.get(orientation, orientation)} Analysis ({count_orientation} chunks)")
                render_interpretation(
                    ORIENTATION_LABEL_MAP.get(orientation, orientation),
                    ORIENTATION_NARRATIVE.get(orientation, "This view compares how implementation language and stakeholder language are framed."),
                )

                left, right = st.columns([1.2, 1])

                with left:
                    st.markdown("**Word Cloud**")
                    render_interactive_wordcloud(df_terms.head(80))

                with right:
                    st.markdown("**Top Keywords**")
                    plot_top_terms(
                        df_terms,
                        title=f"Top TF-IDF Terms: {ORIENTATION_LABEL_MAP.get(orientation, orientation)}",
                        top_k=top_k
                    )

                st.markdown("**Top Keywords Table**")
                st.dataframe(
                    get_top_terms_table(df_terms, top_k=top_k),
                    use_container_width=True,
                    hide_index=True
                )

                if show_chunks:
                    st.markdown("**Representative Chunks**")
                    if len(df_chunk_ori) == 0:
                        st.info("No chunks match the current filter.")
                    else:
                        render_chunk_cards(df_chunk_ori, df_terms, max_chunks=max_chunks)

    elif view_mode == "SDG view":
        tabs = st.tabs([SDG_LABEL_MAP.get(x, x) for x in SDG_ORDER])

        for tab, sdg in zip(tabs, SDG_ORDER):
            with tab:
                df_terms = df_sdg_tfidf[df_sdg_tfidf["sdg_labels"] == sdg].copy()
                df_chunk_sdg = filter_chunks_by_sdg(filtered_chunks, sdg)

                count_sdg = len(filter_chunks_by_sdg(df_chunks, sdg))
                st.subheader(f"{SDG_LABEL_MAP.get(sdg, sdg)} Analysis ({count_sdg} chunks)")
                render_interpretation(
                    SDG_LABEL_MAP.get(sdg, sdg),
                    SDG_NARRATIVE.get(sdg, "This SDG view shows the most distinctive terms and representative supporting chunks."),
                )

                left, right = st.columns([1.2, 1])

                with left:
                    st.markdown("**Word Cloud**")
                    render_interactive_wordcloud(df_terms.head(80))

                with right:
                    st.markdown("**Top Keywords**")
                    plot_top_terms(
                        df_terms,
                        title=f"Top TF-IDF Terms: {SDG_LABEL_MAP.get(sdg, sdg)}",
                        top_k=top_k
                    )

                st.markdown("**Top Keywords Table**")
                st.dataframe(
                    get_top_terms_table(df_terms, top_k=top_k),
                    use_container_width=True,
                    hide_index=True
                )

                if show_chunks:
                    st.markdown("**Representative Chunks**")
                    if len(df_chunk_sdg) == 0:
                        st.info("No chunks match the current filter.")
                    else:
                        render_chunk_cards(df_chunk_sdg, df_terms, max_chunks=max_chunks)

    else:  # Issue Frame view
        tabs = st.tabs([ISSUE_FRAME_LABEL_MAP.get(x, x) for x in ISSUE_FRAME_ORDER])

        for tab, issue_frame in zip(tabs, ISSUE_FRAME_ORDER):
            with tab:
                df_terms = df_issue_frame_tfidf[df_issue_frame_tfidf["issue_frame"] == issue_frame].copy()
                df_chunk_issue = filtered_chunks[filtered_chunks["issue_frame"] == issue_frame].copy()

                count_issue = len(df_chunks[df_chunks["issue_frame"] == issue_frame])
                st.subheader(f"{ISSUE_FRAME_LABEL_MAP.get(issue_frame, issue_frame)} Analysis ({count_issue} chunks)")
                render_interpretation(
                    ISSUE_FRAME_LABEL_MAP.get(issue_frame, issue_frame),
                    ISSUE_FRAME_NARRATIVE.get(issue_frame, "This issue frame summarizes a major strategic theme in the report."),
                )

                left, right = st.columns([1.2, 1])

                with left:
                    st.markdown("**Word Cloud**")
                    render_interactive_wordcloud(df_terms.head(80))

                with right:
                    st.markdown("**Top Keywords**")
                    plot_top_terms(
                        df_terms,
                        title=f"Top TF-IDF Terms: {ISSUE_FRAME_LABEL_MAP.get(issue_frame, issue_frame)}",
                        top_k=top_k
                    )

                st.markdown("**Top Keywords Table**")
                st.dataframe(
                    get_top_terms_table(df_terms, top_k=top_k),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("**Co-occurrence Network**")
                st.caption(
                    "This network links high-signal terms that appear together inside the same chunks for this issue frame."
                )
                render_draggable_issue_frame_network(
                    df_issue_frame_cooccurrence,
                    issue_frame=issue_frame,
                    title=f"Co-occurrence Network: {ISSUE_FRAME_LABEL_MAP.get(issue_frame, issue_frame)}",
                )

                if show_chunks:
                    st.markdown("**Representative Chunks**")
                    if len(df_chunk_issue) == 0:
                        st.info("No chunks match the current filter.")
                    else:
                        render_chunk_cards(df_chunk_issue, df_terms, max_chunks=max_chunks)


# =========================
# Footer
# =========================
st.markdown("---")
st.caption("Built from processed chunks and TF-IDF outputs generated from the TSMC sustainability report.")
