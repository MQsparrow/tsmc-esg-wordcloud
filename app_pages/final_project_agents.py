from __future__ import annotations

import io
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud

from agents import answer_question, run_analysis


SAMPLE_QUESTIONS = [
    "What are the main environmental topics in the report?",
    "How does TSMC discuss water management?",
    "What does the report say about carbon reduction?",
    "What social responsibility topics appear most often?",
    "What governance risks are mentioned?",
]


def _get_configured_api_key(st: Any) -> str:
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""
    return str(secret_key or "")


def _extract_uploaded_text(uploaded_file: Any) -> tuple[str, str]:
    if uploaded_file is None:
        return "", ""
    name = uploaded_file.name
    suffix = name.lower().split(".")[-1]
    if suffix in {"txt", "md", "csv"}:
        return uploaded_file.getvalue().decode("utf-8", errors="ignore"), name
    if suffix == "pdf":
        try:
            import pdfplumber

            text_parts: list[str] = []
            with pdfplumber.open(io.BytesIO(uploaded_file.getvalue())) as pdf:
                for page in pdf.pages[:80]:
                    text_parts.append(page.extract_text() or "")
            return "\n\n".join(text_parts), name
        except Exception as exc:
            return f"PDF extraction failed: {exc}", name
    return uploaded_file.getvalue().decode("utf-8", errors="ignore"), name


def _render_agent_styles(st: Any) -> None:
    st.markdown(
        """
        <style>
        .fp-hero {
            background: linear-gradient(135deg, #07111f 0%, #0f766e 52%, #1e1b4b 100%);
            border-radius: 18px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1.1rem;
            color: white;
            box-shadow: 0 20px 48px rgba(15, 23, 42, 0.18);
        }
        .fp-kicker {
            color: #99f6e4;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }
        .fp-hero h2 {
            color: white;
            font-size: 2rem;
            line-height: 1.08;
            margin: 0 0 0.55rem 0;
        }
        .fp-hero p {
            color: #e0f2fe;
            max-width: 820px;
            margin: 0;
            line-height: 1.55;
        }
        .fp-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }
        .fp-chip {
            border: 1px solid rgba(255,255,255,0.24);
            background: rgba(255,255,255,0.10);
            border-radius: 999px;
            padding: 0.42rem 0.65rem;
            color: #ecfeff;
            font-size: 0.84rem;
            font-weight: 700;
        }
        .fp-panel {
            background: rgba(255,255,255,0.94);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 14px;
            padding: 1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            min-height: 100%;
        }
        .fp-panel h3 {
            color: #0f172a;
            margin: 0 0 0.65rem 0;
            font-size: 1.12rem;
        }
        .fp-agent-flow {
            background: #0f172a;
            color: #ccfbf1;
            border-radius: 14px;
            padding: 0.85rem 1rem;
            margin: 0.8rem 0 1.1rem 0;
            overflow-x: auto;
            font-weight: 800;
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_wordcloud(st: Any, frequencies: dict[str, float]) -> None:
    if not frequencies:
        st.info("No keyword frequencies available yet.")
        return
    wc = WordCloud(
        width=1100,
        height=520,
        background_color="#07111f",
        colormap="winter",
        max_words=90,
        prefer_horizontal=0.9,
        contour_width=1,
        contour_color="#14b8a6",
    ).generate_from_frequencies(frequencies)
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor="#07111f")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _keyword_dataframe(state: dict[str, Any]) -> pd.DataFrame:
    rows = state.get("keywords", []) or []
    if not rows:
        return pd.DataFrame(columns=["term", "score", "frequency", "group"])
    df = pd.DataFrame(rows)
    for col in ["term", "score", "frequency", "group"]:
        if col not in df.columns:
            df[col] = ""
    return df[["term", "score", "frequency", "group"]]


def _classification_dataframe(state: dict[str, Any]) -> pd.DataFrame:
    rows = state.get("esg_classifications", []) or []
    if not rows:
        return pd.DataFrame(columns=["label", "confidence", "reason", "text"])
    return pd.DataFrame(rows)


def render_page(render_page_header, render_metric_grid, render_card_grid, st) -> None:
    _render_agent_styles(st)
    render_page_header(
        "Final Project Agents",
        "TSMC ESG Intelligence Dashboard",
        "A LangGraph-based agent workflow for ESG text mining, visual exploration, and report Q&A.",
    )

    st.markdown(
        """
        <div class="fp-hero">
            <div class="fp-kicker">LangGraph multi-agent workflow</div>
            <h2>TSMC ESG analysis dashboard</h2>
            <p>Five agents turn one sustainability report into keywords, ESG categories, charts, a summary, and source-grounded Q&A.</p>
            <div class="fp-chip-row">
                <div class="fp-chip">Preprocessing</div>
                <div class="fp-chip">Keywords</div>
                <div class="fp-chip">ESG Classifier</div>
                <div class="fp-chip">Visualization</div>
                <div class="fp-chip">Q&A Retrieval</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="fp-agent-flow">Input report -> Preprocess -> Keywords -> ESG classify -> Visualize -> Summarize -> Q&A</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Final Project Agents")
        year = st.selectbox("Report year", ["2024", "2023", "2022"], index=0, key="fp_year")
        top_n = st.slider("Keywords to analyze", 15, 80, 40, 5, key="fp_top_n")
        summary_mode = st.selectbox(
            "Summary style",
            ["executive", "investor-style", "classroom presentation", "short demo script"],
            key="fp_summary_mode",
        )
        model = st.selectbox(
            "OpenAI model",
            ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1"],
            index=0,
            key="fp_model",
        )
        session_api_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="Optional for local testing. On Streamlit Cloud, use app secrets instead.",
            key="fp_api_key",
        )
        uploaded_file = st.file_uploader("Optional report upload", type=["txt", "md", "csv", "pdf"], key="fp_upload")
        run_clicked = st.button("Run LangGraph analysis", type="primary", key="fp_run")

    raw_text, source_name = _extract_uploaded_text(uploaded_file)
    if uploaded_file and raw_text.startswith("PDF extraction failed"):
        st.warning(raw_text)
        raw_text = ""

    # Effective key for this rerun: session text box first, then configured secret.
    api_key = session_api_key.strip() or _get_configured_api_key(st)
    if run_clicked or "fp_agent_state" not in st.session_state:
        with st.spinner("Running ESG agents..."):
            st.session_state.fp_agent_state = run_analysis(
                raw_text=raw_text,
                year=year,
                top_n=top_n,
                summary_mode=summary_mode,
                source=source_name,
                api_key=api_key,
                model=model,
            )
    state = st.session_state.fp_agent_state
    # Keep the cached agent state in sync with the current key/model so the Q&A tab
    # uses a freshly typed key without needing to re-run the whole analysis.
    state["api_key"] = api_key
    state["model"] = model
    if api_key:
        st.sidebar.caption("OpenAI key detected — LLM answers/summaries active (requires the `openai` package).")
    else:
        st.sidebar.caption("No OpenAI key — running deterministic fallback mode.")

    errors = state.get("errors", [])
    if errors:
        with st.expander("Runtime notes"):
            for error in errors:
                st.caption(error)

    chunks = state.get("chunks", []) or []
    keywords = state.get("keywords", []) or []
    counts = state.get("esg_counts", {}) or {}
    dominant = max(counts.items(), key=lambda item: item[1])[0] if counts else "N/A"

    render_metric_grid(
        [
            {"label": "Source", "value": str(state.get("year", year)), "note": str(state.get("source", "processed data"))[:80]},
            {"label": "Chunks", "value": f"{len(chunks):,}", "note": "report segments analyzed"},
            {"label": "Keywords", "value": f"{len(keywords):,}", "note": "ranked terms available"},
            {"label": "Top ESG theme", "value": dominant, "note": "chunk-level classification"},
        ]
    )

    chart_data = state.get("chart_data", {}) or {}
    esg_df = pd.DataFrame(
        [{"category": label, "chunks": value} for label, value in counts.items()]
    )
    col_left, col_right = st.columns([1.25, 1])
    with col_left:
        st.markdown('<div class="fp-panel"><h3>Keyword word cloud</h3>', unsafe_allow_html=True)
        _render_wordcloud(st, chart_data.get("wordcloud_freq", {}))
        st.markdown("</div>", unsafe_allow_html=True)
    with col_right:
        st.markdown('<div class="fp-panel"><h3>ESG distribution</h3>', unsafe_allow_html=True)
        if not esg_df.empty:
            color_map = chart_data.get("esg_colors", {})
            fig = px.pie(
                esg_df,
                names="category",
                values="chunks",
                hole=0.52,
                color="category",
                color_discrete_map=color_map,
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("E / S / G / Other from the ESG classifier (one label per chunk) — a different scheme from the keyword 'group' (report sections) below.")
        else:
            st.info("No ESG classification data yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- AI insight summary, promoted above the tabs as the headline AI output ---
    with st.container(border=True):
        st.markdown("#### ✦ AI insight summary")
        st.caption(
            f"Written by the Summary agent · style: {summary_mode} · "
            + ("LLM (OpenAI)" if api_key else "deterministic fallback (no API key)")
        )
        st.markdown(state.get("summary") or "Run the analysis to generate a summary.")

    # --- Ask the report: the interactive AI feature, kept prominent ---
    st.subheader("Ask the report")
    question_choice = st.selectbox("Sample questions", SAMPLE_QUESTIONS)
    question = st.text_area("Ask a question about the report", value=question_choice, height=90)
    st.caption(
        f"Answers are grounded in the {year} report (set via **Report year** in the sidebar). "
        "For three-year comparison, use the **Cross-Year Compare** page."
    )
    if st.button("Ask report", key="fp_ask"):
        with st.spinner("Retrieving evidence and generating answer..."):
            st.session_state.fp_agent_state = answer_question(state, question, scope="single")
            state = st.session_state.fp_agent_state
    if state.get("qa_answer"):
        st.markdown("#### Answer")
        st.markdown(state["qa_answer"])
    if state.get("retrieved_chunks"):
        st.markdown("#### Source chunks")
        for item in state["retrieved_chunks"]:
            year_tag = f"{item.get('year', '')} | " if item.get("year") else ""
            with st.expander(f"{year_tag}Chunk {item['chunk_id']} | score {item['score']}"):
                st.write(item["text"])

    # --- Supporting deterministic detail, collapsed so the AI stays the focus ---
    with st.expander("Evidence & analysis details (keyword scores + ESG classification)", expanded=False):
        ev_keywords, ev_evidence = st.tabs(["Keywords", "ESG evidence"])
        with ev_keywords:
            keyword_df = _keyword_dataframe(state)
            group_options = ["All"] + sorted([value for value in keyword_df["group"].dropna().astype(str).unique() if value])
            selected_group = st.selectbox("Keyword group", group_options)
            filtered_keywords = keyword_df if selected_group == "All" else keyword_df[keyword_df["group"] == selected_group]
            # Drop the always-empty frequency column (the section TF-IDF source has no raw counts).
            show_cols = [c for c in ["term", "score", "group"] if c in filtered_keywords.columns]
            st.dataframe(filtered_keywords[show_cols].head(top_n), use_container_width=True, hide_index=True)
            st.caption("score = TF-IDF importance (higher = more characteristic). group = report section, not the E/S/G label. The word cloud above is the visual version of this table.")
        with ev_evidence:
            class_df = _classification_dataframe(state)
            label_options = ["All"] + sorted([value for value in class_df["label"].dropna().unique() if value])
            selected_label = st.selectbox("ESG category", label_options)
            evidence_df = class_df if selected_label == "All" else class_df[class_df["label"] == selected_label]
            display_cols = [col for col in ["label", "confidence", "reason", "section_label", "sdg_labels", "text"] if col in evidence_df.columns]
            st.dataframe(evidence_df[display_cols].head(20), use_container_width=True, hide_index=True)
            st.caption("Each row = one report chunk and why the classifier tagged it E / S / G / Other.")
