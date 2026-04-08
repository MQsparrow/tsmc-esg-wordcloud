from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px


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


# =========================
# Constants
# =========================
SECTION_ORDER = ["environment", "talent", "supply_chain", "social", "governance"]
ORIENTATION_ORDER = ["action_oriented", "people_centric", "mixed"]
SDG_ORDER = ["SDG3_health", "SDG4_education", "SDG6_water", "SDG7_energy", "SDG8_labor", "SDG9_innovation", "SDG12_consumption", "SDG13_climate", "SDG17_partnership"]

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


# =========================
# Load data
# =========================
@st.cache_data
def load_data():
    df_chunks = pd.read_csv(CHUNKS_PATH)
    df_section = pd.read_csv(SECTION_TFIDF_PATH)
    df_orientation = pd.read_csv(ORIENTATION_TFIDF_PATH)
    df_sdg = pd.read_csv(SDG_TFIDF_PATH)

    for col in ["raw_text", "clean_text", "section_label", "orientation", "sdg_labels"]:
        if col in df_chunks.columns:
            df_chunks[col] = df_chunks[col].fillna("").astype(str)

    return df_chunks, df_section, df_orientation, df_sdg


df_chunks, df_section_tfidf, df_orientation_tfidf, df_sdg_tfidf = load_data()


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
    return x.replace("_", " ").title()


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
    st.plotly_chart(fig, use_container_width=True)


def get_top_terms_table(df: pd.DataFrame, top_k: int = 20):
    out = (
        df.sort_values("tfidf_score", ascending=False)
        .head(top_k)[["term", "tfidf_score"]]
        .reset_index(drop=True)
        .copy()
    )
    out["tfidf_score"] = out["tfidf_score"].round(4)
    return out


def render_chunk_cards(df: pd.DataFrame, max_chunks: int = 5):
    view_df = df.copy()
    if "clean_text" in view_df.columns:
        view_df["token_count"] = view_df["clean_text"].str.split().str.len()
        view_df = view_df.sort_values("token_count", ascending=False)

    for i, (_, row) in enumerate(view_df.head(max_chunks).iterrows(), start=1):
        label_section = prettify_label(row["section_label"])
        label_orientation = prettify_label(row["orientation"])
        label_SDGs = ", ".join([prettify_label(s.strip()) for s in row.get("sdg_labels", "").split(",") if s.strip() and s.strip() != "unclassified"])

        with st.expander(
            f"Chunk {i} | {label_section} | {label_orientation} | {label_SDGs}",
            expanded=False
        ):
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
    st.plotly_chart(fig, use_container_width=True)


def plot_distribution_bar(df: pd.DataFrame, col: str, title: str):
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    counts[col] = counts[col].map(prettify_label)

    fig = px.bar(
        counts,
        x=col,
        y="count",
        title=title,
        text="count"
    )
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="",
        yaxis_title="Count"
    )
    st.plotly_chart(fig, use_container_width=True)


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
    return df[df["raw_text"].str.lower().str.contains(kw, na=False)].copy()


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
    ["Section view", "Orientation view", "SDG view"]
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
    sdg_counts = (
        df_chunks["sdg_labels"].str.split(",").explode().str.strip()
        .loc[lambda s: s != "unclassified"]
        .value_counts()
    )
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

    df_chunks_sdg_exploded = (
        df_chunks.assign(sdg_labels=df_chunks["sdg_labels"].str.split(","))
        .explode("sdg_labels")
        .assign(sdg_labels=lambda d: d["sdg_labels"].str.strip())
        .loc[lambda d: d["sdg_labels"] != "unclassified"]
    )
    plot_distribution_bar(df_chunks_sdg_exploded, "sdg_labels", "SDG Distribution")

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

                left, right = st.columns([1.2, 1])

                with left:
                    st.markdown("**Word Cloud**")
                    wc = make_wordcloud_from_tfidf(df_terms.head(80))
                    if wc:
                        plot_wordcloud(wc)
                    else:
                        st.info("No terms available.")

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
                        render_chunk_cards(df_chunk_section, max_chunks=max_chunks)

    elif view_mode == "Orientation view":
        tabs = st.tabs([ORIENTATION_LABEL_MAP.get(x, x) for x in ORIENTATION_ORDER])

        for tab, orientation in zip(tabs, ORIENTATION_ORDER):
            with tab:
                df_terms = df_orientation_tfidf[df_orientation_tfidf["orientation"] == orientation].copy()
                df_chunk_ori = filtered_chunks[filtered_chunks["orientation"] == orientation].copy()

                count_orientation = len(df_chunks[df_chunks["orientation"] == orientation])
                st.subheader(f"{ORIENTATION_LABEL_MAP.get(orientation, orientation)} Analysis ({count_orientation} chunks)")

                left, right = st.columns([1.2, 1])

                with left:
                    st.markdown("**Word Cloud**")
                    wc = make_wordcloud_from_tfidf(df_terms.head(80))
                    if wc:
                        plot_wordcloud(wc)
                    else:
                        st.info("No terms available.")

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
                        render_chunk_cards(df_chunk_ori, max_chunks=max_chunks)

    else:  # SDG view
        tabs = st.tabs([SDG_LABEL_MAP.get(x, x) for x in SDG_ORDER])

        for tab, sdg in zip(tabs, SDG_ORDER):
            with tab:
                df_terms = df_sdg_tfidf[df_sdg_tfidf["sdg_labels"] == sdg].copy()
                df_chunk_sdg = filtered_chunks[filtered_chunks["sdg_labels"].str.contains(sdg, na=False)].copy()

                count_sdg = df_chunks["sdg_labels"].str.contains(sdg, na=False).sum()
                st.subheader(f"{SDG_LABEL_MAP.get(sdg, sdg)} Analysis ({count_sdg} chunks)")

                left, right = st.columns([1.2, 1])

                with left:
                    st.markdown("**Word Cloud**")
                    wc = make_wordcloud_from_tfidf(df_terms.head(80))
                    if wc:
                        plot_wordcloud(wc)
                    else:
                        st.info("No terms available.")

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
                        render_chunk_cards(df_chunk_sdg, max_chunks=max_chunks)


# =========================
# Footer
# =========================
st.markdown("---")
st.caption("Built from processed chunks and TF-IDF outputs generated from the TSMC sustainability report.")