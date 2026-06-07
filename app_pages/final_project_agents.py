from __future__ import annotations

import html
import time
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud

from agents import answer_question, run_analysis
from agents.classify import classification_agent
from agents.graph import qa_agent, summary_agent
from agents.keywords import keyword_agent
from agents.preprocess import preprocess_agent
from agents.retrieve import retrieval_agent
from agents.visualize import visualization_agent


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
        .fp-trace-shell {
            border: 1px solid rgba(20, 184, 166, 0.22);
            background:
                linear-gradient(135deg, rgba(240,253,250,0.96) 0%, rgba(255,255,255,0.98) 46%, rgba(239,246,255,0.96) 100%);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.08);
            margin: 1rem 0;
        }
        .fp-trace-head {
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            align-items: flex-start;
            margin-bottom: 0.9rem;
        }
        .fp-trace-title {
            color: #0f172a;
            font-size: 1.08rem;
            font-weight: 850;
            margin: 0 0 0.15rem 0;
        }
        .fp-trace-subtitle {
            color: #475569;
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 0;
        }
        .fp-trace-badge {
            flex: 0 0 auto;
            border-radius: 999px;
            padding: 0.42rem 0.65rem;
            background: #0f172a;
            color: #ccfbf1;
            font-size: 0.78rem;
            font-weight: 800;
            white-space: nowrap;
        }
        .fp-trace-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
        }
        .fp-trace-step {
            position: relative;
            overflow: hidden;
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 12px;
            padding: 0.78rem 0.82rem;
            min-height: 118px;
        }
        .fp-trace-step::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 4px;
            background: linear-gradient(90deg, #14b8a6, #2563eb);
        }
        .fp-trace-kicker {
            color: #0f766e;
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.34rem;
        }
        .fp-trace-step h4 {
            color: #0f172a;
            font-size: 0.96rem;
            line-height: 1.25;
            margin: 0 0 0.35rem 0;
        }
        .fp-trace-step p {
            color: #475569;
            font-size: 0.84rem;
            line-height: 1.45;
            margin: 0;
        }
        .fp-trace-proof {
            color: #0f766e;
            font-size: 0.78rem;
            font-weight: 750;
            margin-top: 0.45rem;
        }
        .fp-live-shell {
            border: 1px solid rgba(20, 184, 166, 0.22);
            background: linear-gradient(135deg, #07111f 0%, #0f172a 52%, #172554 100%);
            border-radius: 16px;
            padding: 1rem;
            margin: 0.9rem 0 1.1rem 0;
            box-shadow: 0 18px 38px rgba(15, 23, 42, 0.18);
            color: #e0f2fe;
        }
        .fp-live-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.8rem;
        }
        .fp-live-title {
            color: white;
            font-size: 1rem;
            font-weight: 850;
            margin: 0;
        }
        .fp-live-badge {
            border: 1px solid rgba(153, 246, 228, 0.34);
            background: rgba(20, 184, 166, 0.14);
            color: #ccfbf1;
            border-radius: 999px;
            padding: 0.34rem 0.58rem;
            font-size: 0.76rem;
            font-weight: 800;
            white-space: nowrap;
        }
        .fp-live-events {
            display: grid;
            gap: 0.55rem;
        }
        .fp-live-event {
            display: grid;
            grid-template-columns: 88px 1fr;
            gap: 0.7rem;
            align-items: start;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(255, 255, 255, 0.07);
            border-radius: 12px;
            padding: 0.62rem 0.7rem;
        }
        .fp-live-event.latest {
            border-color: rgba(45, 212, 191, 0.68);
            background: rgba(20, 184, 166, 0.16);
        }
        .fp-live-time {
            color: #99f6e4;
            font-size: 0.74rem;
            font-weight: 850;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .fp-live-agent {
            color: #ffffff;
            font-size: 0.88rem;
            font-weight: 850;
            margin-bottom: 0.16rem;
        }
        .fp-live-message {
            color: #cbd5e1;
            font-size: 0.84rem;
            line-height: 1.45;
        }
        .fp-chart-title {
            color: #1f2937;
            font-size: 1.45rem;
            line-height: 1.1;
            font-weight: 850;
            margin: 0.35rem 0 0.9rem 0;
        }
        @media (max-width: 900px) {
            .fp-trace-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 620px) {
            .fp-trace-head {
                flex-direction: column;
            }
            .fp-trace-grid {
                grid-template-columns: 1fr;
            }
            .fp-live-event {
                grid-template-columns: 1fr;
                gap: 0.25rem;
            }
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
        height=590,
        background_color="#07111f",
        colormap="winter",
        max_words=90,
        prefer_horizontal=0.9,
        contour_width=1,
        contour_color="#14b8a6",
    ).generate_from_frequencies(frequencies)
    fig, ax = plt.subplots(figsize=(11, 5.9), facecolor="#07111f")
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


def _safe_text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _trace_terms(state: dict[str, Any], limit: int = 5) -> str:
    terms = [str(item.get("term", "")) for item in state.get("keywords", []) if item.get("term")]
    return ", ".join(terms[:limit]) if terms else "waiting for keywords"


def _trace_counts(counts: dict[str, Any]) -> str:
    if not counts:
        return "waiting for classifier output"
    return ", ".join(f"{label}: {value}" for label, value in counts.items())


def _compact_source_label(source: Any, year: str) -> str:
    text = str(source or "").strip()
    if not text:
        return "processed data"
    text = text.replace("\\", "/")
    text = text.replace("outputs/", "")
    text = text.replace("_processed.csv", "")
    if len(text) > 30:
        return f"{year} processed chunks"
    return text


def _merge_state(state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(state)
    merged.update(update)
    return merged


def _render_live_log(placeholder: Any, events: list[dict[str, str]], title: str = "Live agent run") -> None:
    event_html = []
    for index, event in enumerate(events):
        latest = " latest" if index == len(events) - 1 else ""
        event_html.append(
            f"<div class=\"fp-live-event{latest}\">"
            f"<div class=\"fp-live-time\">{_safe_text(event.get('time', 'now'))}</div>"
            "<div>"
            f"<div class=\"fp-live-agent\">{_safe_text(event.get('agent', 'Agent'))}</div>"
            f"<div class=\"fp-live-message\">{_safe_text(event.get('message', 'Working...'))}</div>"
            "</div>"
            "</div>"
        )
    placeholder.markdown(
        "<div class=\"fp-live-shell\">"
        "<div class=\"fp-live-head\">"
        f"<div class=\"fp-live-title\">{_safe_text(title)}</div>"
        "<div class=\"fp-live-badge\">streaming execution</div>"
        "</div>"
        f"<div class=\"fp-live-events\">{''.join(event_html)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _event(agent: str, message: str) -> dict[str, str]:
    return {"time": time.strftime("%H:%M:%S"), "agent": agent, "message": message}


def _run_node_live(
    placeholder: Any,
    messages: list[str],
    state: dict[str, Any],
    label: str,
    action: str,
    node: Any,
    describe: Any,
) -> dict[str, Any]:
    messages.append(_event(label, action))
    _render_live_log(placeholder, messages)
    time.sleep(0.9)
    try:
        update = node(state)
        state = _merge_state(state, update)
        messages.append(_event(label, describe(state)))
    except Exception as exc:
        errors = list(state.get("errors", [])) + [f"{label}: {exc}"]
        state["errors"] = errors
        messages.append(_event(label, f"{type(exc).__name__}: {str(exc)[:120]}"))
    _render_live_log(placeholder, messages)
    time.sleep(1.0)
    return state


def _run_analysis_live(
    st: Any,
    raw_text: str,
    year: str,
    top_n: int,
    summary_mode: str,
    source: str,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    placeholder = st.empty()
    messages: list[dict[str, str]] = []
    state: dict[str, Any] = {
        "raw_text": raw_text,
        "year": str(year),
        "source": source,
        "mode": summary_mode,
        "api_key": api_key,
        "model": model,
        "chart_data": {"top_n": top_n},
        "qa_scope": "cross_year",
        "errors": [],
    }
    plan = [
        (
            "Preprocess agent",
            "loading the selected report and splitting it into usable chunks",
            preprocess_agent,
            lambda s: f"{len(s.get('chunks', []) or []):,} chunks ready from {s.get('source', 'the report')}",
        ),
        (
            "Keyword agent",
            "ranking terms that best describe the report language",
            keyword_agent,
            lambda s: f"{len(s.get('keywords', []) or []):,} keywords ranked; top terms: {_trace_terms(s, 4)}",
        ),
        (
            "ESG classifier",
            "assigning E / S / G / Other labels to chunks",
            classification_agent,
            lambda s: _trace_counts(s.get("esg_counts", {}) or {}),
        ),
        (
            "Visualization agent",
            "preparing chart payloads for the dashboard",
            visualization_agent,
            lambda s: f"{len((s.get('chart_data', {}) or {}).get('wordcloud_freq', {}) or {}):,} word-cloud terms prepared",
        ),
        (
            "Summary agent",
            "writing the presentation summary from the current analysis state",
            summary_agent,
            lambda s: "LLM summary returned" if api_key else "fallback summary returned",
        ),
    ]
    for label, action, node, describe in plan:
        state = _run_node_live(placeholder, messages, state, label, action, node, describe)
    messages.append(_event("Coordinator", "Dashboard state updated. The visual report is ready."))
    _render_live_log(placeholder, messages)
    return state


def _answer_question_live(
    st: Any,
    state: dict[str, Any],
    question: str,
    scope: str,
    api_key: str,
) -> dict[str, Any]:
    placeholder = st.empty()
    messages: list[dict[str, str]] = []
    qa_state = dict(state)
    qa_state["user_question"] = question
    qa_state["qa_scope"] = scope
    qa_state = _run_node_live(
        placeholder,
        messages,
        qa_state,
        "Retrieval agent",
        f"searching evidence with `{scope}` scope",
        retrieval_agent,
        lambda s: (
            f"{len(s.get('retrieved_chunks', []) or []):,} chunks retrieved from "
            f"{', '.join(sorted({str(x.get('year', '')) for x in s.get('retrieved_chunks', []) if x.get('year')})) or 'current report'}"
        ),
    )
    qa_state = _run_node_live(
        placeholder,
        messages,
        qa_state,
        "Q&A agent",
        "turning retrieved evidence into a grounded answer",
        qa_agent,
        lambda s: "LLM answer returned" if api_key else "fallback evidence answer returned",
    )
    messages.append(_event("Coordinator", "Answer and source chunks updated."))
    _render_live_log(placeholder, messages)
    return qa_state


def _render_agent_trace(
    st: Any,
    state: dict[str, Any],
    year: str,
    summary_mode: str,
    model: str,
    api_key: str,
    qa_scope: str,
) -> None:
    chunks = state.get("chunks", []) or []
    keywords = state.get("keywords", []) or []
    counts = state.get("esg_counts", {}) or {}
    retrieved = state.get("retrieved_chunks", []) or []
    chart_data = state.get("chart_data", {}) or {}
    years = sorted({str(item.get("year", "")).strip() for item in retrieved if str(item.get("year", "")).strip()})
    retrieved_label = ", ".join(
        f"{item.get('year', year)} / {item.get('chunk_id')} / {item.get('score')}"
        for item in retrieved[:4]
    )
    scope_label = "All years 2022-2024" if qa_scope == "cross_year" else f"Current year {year}"
    llm_label = "LLM active" if api_key else "Fallback mode"

    steps = [
        {
            "kicker": "01 Input",
            "title": "Report context locked",
            "body": f"Source: {state.get('source', 'processed data') or 'processed data'} | Year setting: {year}.",
            "proof": f"{len(chunks):,} chunks ready",
        },
        {
            "kicker": "02 Keywords",
            "title": "Keyword agent ranked signals",
            "body": _trace_terms(state),
            "proof": f"{len(keywords):,} terms in state",
        },
        {
            "kicker": "03 Classifier",
            "title": "ESG classifier assigned labels",
            "body": _trace_counts(counts),
            "proof": "Dominant theme calculated",
        },
        {
            "kicker": "04 Visuals",
            "title": "Visualization payload prepared",
            "body": "Word cloud frequencies and ESG chart data are available for the dashboard.",
            "proof": f"{len(chart_data.get('wordcloud_freq', {}) or {}):,} cloud terms",
        },
        {
            "kicker": "05 Summary",
            "title": "Summary agent generated narrative",
            "body": f"Mode: {summary_mode} | Model: {model} | Runtime: {llm_label}.",
            "proof": "Presentation summary ready",
        },
        {
            "kicker": "06 Retrieval + Q&A",
            "title": "Evidence trail for the answer",
            "body": (
                f"Scope: {scope_label}. Years found: {', '.join(years) or 'waiting for question'}."
                if not retrieved_label
                else f"Scope: {scope_label}. Retrieved: {retrieved_label}."
            ),
            "proof": f"{len(retrieved):,} source chunks returned",
        },
    ]

    cards = []
    for step in steps:
        cards.append(
            "<div class=\"fp-trace-step\">"
            f"<div class=\"fp-trace-kicker\">{_safe_text(step['kicker'])}</div>"
            f"<h4>{_safe_text(step['title'])}</h4>"
            f"<p>{_safe_text(step['body'])}</p>"
            f"<div class=\"fp-trace-proof\">{_safe_text(step['proof'])}</div>"
            "</div>"
        )

    st.markdown(
        "<div class=\"fp-trace-shell\">"
        "<div class=\"fp-trace-head\">"
        "<div>"
        "<div class=\"fp-trace-title\">Agent execution trace</div>"
        "<p class=\"fp-trace-subtitle\">A presentation-friendly view of what each agent did, using observable inputs, outputs, and evidence.</p>"
        "</div>"
        f"<div class=\"fp-trace-badge\">{_safe_text(scope_label)} | {_safe_text(llm_label)}</div>"
        "</div>"
        f"<div class=\"fp-trace-grid\">{''.join(cards)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


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
        retrieval_scope = st.selectbox(
            "Q&A retrieval scope",
            ["Current report year", "All years (2022-2024)"],
            index=0,
            key="fp_retrieval_scope",
        )
        run_clicked = st.button("Run LangGraph analysis", type="primary", key="fp_run")

    # Effective key for this rerun: session text box first, then configured secret.
    api_key = session_api_key.strip() or _get_configured_api_key(st)
    if run_clicked or "fp_agent_state" not in st.session_state:
        st.session_state.fp_agent_state = _run_analysis_live(
            st=st,
            raw_text="",
            year=year,
            top_n=top_n,
            summary_mode=summary_mode,
            source="",
            api_key=api_key,
            model=model,
        )
    state = st.session_state.fp_agent_state
    # Keep the cached agent state in sync with the current key/model so the Q&A tab
    # uses a freshly typed key without needing to re-run the whole analysis.
    state["api_key"] = api_key
    state["model"] = model
    if api_key:
        st.sidebar.caption("OpenAI key detected. LLM summaries and answers are enabled.")
    else:
        st.sidebar.caption("No OpenAI key detected. Running deterministic fallback mode.")

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
            {"label": "Source", "value": str(state.get("year", year)), "note": _compact_source_label(state.get("source", ""), year)},
            {"label": "Chunks", "value": f"{len(chunks):,}", "note": "report segments analyzed"},
            {"label": "Keywords", "value": f"{len(keywords):,}", "note": "ranked terms available"},
            {"label": "Top ESG theme", "value": dominant, "note": "chunk-level classification"},
        ]
    )
    st.markdown("<div style='height:1.15rem'></div>", unsafe_allow_html=True)

    chart_data = state.get("chart_data", {}) or {}
    esg_df = pd.DataFrame(
        [{"category": label, "chunks": value} for label, value in counts.items()]
    )
    col_left, col_right = st.columns([1.12, 1], gap="large")
    with col_left:
        with st.container(border=True):
            st.markdown('<div class="fp-chart-title">Keyword word cloud</div>', unsafe_allow_html=True)
            _render_wordcloud(st, chart_data.get("wordcloud_freq", {}))
    with col_right:
        with st.container(border=True):
            st.markdown('<div class="fp-chart-title">ESG distribution</div>', unsafe_allow_html=True)
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
                fig.update_layout(
                    height=300,
                    margin=dict(l=4, r=4, t=4, b=4),
                    legend=dict(orientation="v", y=0.5, yanchor="middle", x=1.02, xanchor="left"),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            if esg_df.empty:
                st.info("No ESG classification data yet.")

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
    qa_scope = "cross_year" if retrieval_scope == "All years (2022-2024)" else "single"
    scope_note = (
        "Answers search across 2022, 2023, and 2024 reports."
        if qa_scope == "cross_year"
        else f"Answers are grounded in the {year} report."
    )
    st.caption(
        f"{scope_note} For metrics-based three-year comparison, use the **Cross-Year Compare** page."
    )
    if st.button("Ask report", key="fp_ask"):
        st.session_state.fp_agent_state = _answer_question_live(st, state, question, qa_scope, api_key)
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

    _render_agent_trace(st, state, year, summary_mode, model, api_key, qa_scope)

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
