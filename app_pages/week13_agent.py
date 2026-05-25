from __future__ import annotations

import contextlib
import html
import io
import os
from typing import Any

import pandas as pd
import plotly.express as px

from src.agent_tools import (
    AgentToolError,
    BERT_ASSETS,
    PROJECT_ROOT,
    compare_groups,
    get_bert_assets,
    get_representative_chunks,
    get_similarity_neighbors,
    get_top_terms,
    build_structured_answer,
    normalize_group_name,
    run_fallback_agent,
)


EXAMPLE_QUERIES = [
    "What are the key terms for SDG13 in 2024?",
    "Show evidence chunks for Talent section in 2024.",
    "Compare SDG12 and SDG13 in 2024.",
    "Which SDGs are semantically similar according to BERT?",
]


def _get_secret_api_key(st: Any) -> str:
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""
    return os.getenv("OPENAI_API_KEY", "") or secret_key


def _ag2_available() -> tuple[bool, str]:
    try:
        import autogen  # noqa: F401
    except ImportError:
        return False, "AG2 is not installed. Install `requirements_agent.txt` to enable AG2 mode."
    return True, "AG2 is installed."


def _table_to_text(rows: pd.DataFrame, max_rows: int = 8) -> str:
    if rows.empty:
        return "No rows returned."
    return rows.head(max_rows).to_markdown(index=False)


def _run_ag2_agent(query: str, year: str, model: str, api_key: str) -> dict[str, Any]:
    from autogen import ConversableAgent, register_function

    if not api_key:
        raise AgentToolError("OPENAI_API_KEY is required for AG2 mode.")

    llm_config = {
        "config_list": [
            {
                "model": model,
                "api_key": api_key,
            }
        ],
    }

    analyst = ConversableAgent(
        name="esg_analyst_agent",
        system_message=(
            "You are an ESG text-mining analyst for a TSMC sustainability report dashboard. "
            "Use the registered tools to answer with concise evidence. Do not invent data. "
            "Valid group_type values are 'sdg' and 'section'. Valid SDG examples include "
            "SDG13_climate, SDG12_consumption, SDG17_partnership. Valid section examples "
            "include environment, talent, supply_chain, social, governance. "
            "After tool output is returned, write one final answer and end with TERMINATE."
        ),
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    executor = ConversableAgent(
        name="project_tool_executor",
        llm_config=False,
        human_input_mode="NEVER",
    )
    executed_results: list[dict[str, Any]] = []

    def top_terms_tool(group_type: str, group_name: str, top_n: int = 8) -> str:
        group_name = normalize_group_name(group_type, group_name)
        result = get_top_terms(year, group_type, group_name, top_n)
        executed_results.append(result)
        return _table_to_text(result["rows"])

    def evidence_tool(group_type: str, group_name: str, top_n: int = 3) -> str:
        group_name = normalize_group_name(group_type, group_name)
        result = get_representative_chunks(year, group_type, group_name, top_n)
        executed_results.append(result)
        return _table_to_text(result["rows"], max_rows=top_n)

    def similarity_tool(group_type: str, group_name: str, top_n: int = 5) -> str:
        group_name = normalize_group_name(group_type, group_name)
        result = get_similarity_neighbors(year, group_type, group_name, top_n)
        executed_results.append(result)
        return _table_to_text(result["rows"])

    def compare_tool(group_type: str, group_a: str, group_b: str, top_n: int = 6) -> str:
        group_a = normalize_group_name(group_type, group_a)
        group_b = normalize_group_name(group_type, group_b)
        result = compare_groups(year, group_type, group_a, group_b, top_n)
        executed_results.append(result)
        return _table_to_text(result["rows"], max_rows=top_n * 2)

    def bert_assets_tool() -> str:
        result = get_bert_assets()
        executed_results.append(result)
        return _table_to_text(result["rows"])

    register_function(
        top_terms_tool,
        caller=analyst,
        executor=executor,
        description="Get top TF-IDF terms. group_type must be 'sdg' or 'section'.",
    )
    register_function(
        evidence_tool,
        caller=analyst,
        executor=executor,
        description="Get representative report chunks for an SDG or section.",
    )
    register_function(
        similarity_tool,
        caller=analyst,
        executor=executor,
        description="Get most similar SDG or section groups from cosine similarity output.",
    )
    register_function(
        compare_tool,
        caller=analyst,
        executor=executor,
        description="Compare top TF-IDF terms for two SDGs or two sections.",
    )
    register_function(
        bert_assets_tool,
        caller=analyst,
        executor=executor,
        description="List available BERT semantic figures for the project.",
    )

    message = (
        f"Year: {year}\n"
        f"Question: {query}\n"
        "Answer with: tools used, key evidence, and one short interpretation."
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        chat_result = executor.initiate_chat(analyst, message=message, max_turns=6, silent=True)
    history = getattr(chat_result, "chat_history", []) or []
    answer = ""
    for item in reversed(history):
        content = str(item.get("content", "") or "").strip()
        if content and "TERMINATE" in content:
            answer = content.replace("TERMINATE", "").strip()
            break
        if content and ("Tools used" in content or "Key evidence" in content or "Interpretation" in content):
            answer = content.replace("TERMINATE", "").strip()
            break
    if not answer:
        answer = history[-1].get("content", "") if history else str(chat_result)
    if executed_results:
        answer = build_structured_answer(query, executed_results)
    else:
        unreliable_answer = (
            not answer
            or "empty message" in answer.lower()
            or "which would you like" in answer.lower()
            or "didn't receive a request" in answer.lower()
            or "didn't receive any parameters" in answer.lower()
            or "did not receive any parameters" in answer.lower()
            or "didn't receive a task" in answer.lower()
            or "did not receive a task" in answer.lower()
            or "didn’t get" in answer.lower()
            or "request parameters" in answer.lower()
            or "tell me what you want" in answer.lower()
            or "which action and group" in answer.lower()
            or "please pick one action" in answer.lower()
        )
        if unreliable_answer:
            answer = build_structured_answer(query, executed_results)
    return {
        "mode": "AG2 mode",
        "year": year,
        "answer": answer,
        "tool_results": executed_results,
    }


def _render_tool_result(st: Any, result: dict[str, Any]) -> None:
    st.markdown(f"**{result['tool']}**")
    st.caption(f"{result['group_type']} | {result['group_name']} | {result['year']}")
    rows = result.get("rows")
    if isinstance(rows, pd.DataFrame) and not rows.empty:
        if {"term", "tfidf_score"}.issubset(rows.columns):
            chart_df = rows.copy()
            color_col = "group" if "group" in chart_df.columns else None
            fig = px.bar(
                chart_df,
                x="tfidf_score",
                y="term",
                color=color_col,
                orientation="h",
                title="Top TF-IDF terms",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=360)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No table output for this tool.")


def _render_bert_assets(st: Any) -> None:
    available_assets = [asset for asset in BERT_ASSETS if asset["path"].exists()]
    if not available_assets:
        st.warning("No BERT figure files were found in `pic/`.")
        return
    cols = st.columns(2)
    for index, asset in enumerate(available_assets):
        with cols[index % 2]:
            st.image(str(asset["path"]), caption=f"{asset['name']} - {asset['use']}")


def _render_compact_overview(st: Any, status_cards: list[dict[str, str]], info_cards: list[dict[str, str]]) -> None:
    status_html = "".join(
        f"""
        <div class="week13-status-card">
            <div class="week13-label">{html.escape(card["label"])}</div>
            <div class="week13-value">{html.escape(card["value"])}</div>
            <div class="week13-note">{html.escape(card["note"])}</div>
        </div>
        """
        for card in status_cards
    )
    info_html = "".join(
        f"""
        <div class="week13-info-card {'week13-info-highlight' if card.get("highlight") else ''}">
            <div class="week13-kicker">{html.escape(card["kicker"])}</div>
            <h3>{html.escape(card["title"])}</h3>
            <p>{html.escape(card["text"])}</p>
        </div>
        """
        for card in info_cards
    )
    st.markdown(
        f"""
        <style>
        .week13-status-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
            margin: 1rem 0 1.25rem 0;
        }}
        .week13-status-card {{
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            min-height: 116px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
        }}
        .week13-label {{
            color: #64748b;
            font-size: 0.86rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }}
        .week13-value {{
            color: #0f172a;
            font-size: 1.65rem;
            line-height: 1.05;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }}
        .week13-note {{
            color: #64748b;
            font-size: 0.92rem;
            line-height: 1.35;
        }}
        .week13-info-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin: 0 0 1.5rem 0;
        }}
        .week13-info-card {{
            background: rgba(255,255,255,0.94);
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 16px;
            padding: 1.15rem 1.25rem;
            min-height: 156px;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
        }}
        .week13-info-highlight {{
            background: linear-gradient(145deg, #ecfeff 0%, #f8fafc 100%);
            border-color: rgba(14, 165, 164, 0.22);
        }}
        .week13-kicker {{
            color: #0f766e;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }}
        .week13-info-card h3 {{
            color: #0f172a;
            font-size: 1.35rem;
            line-height: 1.2;
            margin: 0 0 0.9rem 0;
        }}
        .week13-info-card p {{
            color: #475569;
            font-size: 0.98rem;
            line-height: 1.55;
            margin: 0;
        }}
        @media (max-width: 900px) {{
            .week13-status-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
            .week13-info-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>
        <div class="week13-status-grid">{status_html}</div>
        <div class="week13-info-grid">{info_html}</div>
        """,
        unsafe_allow_html=True,
    )


def render_page(
    render_page_header,
    render_metric_grid,
    render_card_grid,
    st,
) -> None:
    render_page_header(
        "Week 13 Agent",
        "TSMC ESG Agent Assistant",
        (
            "A small AG2-style tool agent that answers questions using existing TF-IDF, "
            "evidence chunk, similarity, and BERT semantic outputs."
        ),
    )

    ag2_ok, ag2_message = _ag2_available()
    secret_api_key = _get_secret_api_key(st)
    session_api_key = st.text_input(
        "OpenAI API key for this browser session",
        type="password",
        help=(
            "Optional. Used only for this Streamlit session. "
            "If left blank, the app uses configured Streamlit secrets when available, "
            "otherwise it falls back to deterministic demo mode."
        ),
    )
    api_key = session_api_key.strip() or secret_api_key
    _render_compact_overview(
        st,
        [
            {"label": "Agent Framework", "value": "AG2" if ag2_ok else "Fallback", "note": ag2_message},
            {
                "label": "LLM API",
                "value": "Ready" if api_key else "No key",
                "note": "Session key" if session_api_key.strip() else ("Streamlit secret" if secret_api_key else "Fallback mode remains available"),
            },
            {"label": "Project Tools", "value": "5", "note": "TF-IDF, chunks, similarity, BERT"},
            {"label": "Data Source", "value": "2022-2024", "note": "Existing processed report outputs"},
        ],
        [
            {
                "kicker": "Agent Role",
                "title": "ESG analyst",
                "text": "Interprets a user question and grounds the answer in project outputs.",
                "highlight": True,
            },
            {
                "kicker": "Tools",
                "title": "CSV + BERT evidence",
                "text": "Looks up keywords, report chunks, group comparisons, similarity scores, and semantic figures.",
            },
        ],
    )

    st.subheader("Agent query")
    query_choice = st.selectbox("Example queries", EXAMPLE_QUERIES)
    query = st.text_area("Ask a project question", value=query_choice, height=90)

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        year = st.selectbox("Year", ["2024", "2023", "2022"], index=0)
    with col_b:
        mode = st.selectbox("Run mode", ["Auto", "Fallback demo", "AG2"], index=0)
    with col_c:
        model = st.text_input("OpenAI model for AG2 mode", value="gpt-5-mini")

    run_clicked = st.button("Run agent", type="primary")

    if not run_clicked:
        st.info("Run one of the example questions.")
        st.subheader("BERT semantic tool preview")
        _render_bert_assets(st)
        return

    use_ag2 = mode == "AG2" or (mode == "Auto" and ag2_ok and bool(api_key))
    try:
        if use_ag2:
            result = _run_ag2_agent(query, year, model, api_key)
            st.success("AG2 mode completed.")
        else:
            result = run_fallback_agent(query, year)
            st.warning("Fallback demo mode used. Install AG2 and set OPENAI_API_KEY for full AG2 mode.")
    except Exception as exc:
        st.error(f"Agent mode failed, using fallback demo mode. Reason: {exc}")
        result = run_fallback_agent(query, year)

    st.subheader("Agent answer")
    st.markdown(result["answer"])

    tool_results = result.get("tool_results", [])
    if tool_results:
        st.subheader("Tool trace and evidence")
        for tool_result in tool_results:
            _render_tool_result(st, tool_result)

    if "bert" in query.lower() or "semantic" in query.lower() or "embedding" in query.lower():
        st.subheader("BERT semantic figures")
        _render_bert_assets(st)
