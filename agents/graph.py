from __future__ import annotations

import os
from typing import Any, Callable

from .classify import classification_agent
from .keywords import keyword_agent
from .llm import answer_with_optional_llm, summarize_cross_year, summarize_with_optional_llm
from .preprocess import preprocess_agent
from .retrieve import retrieval_agent
from .state import ESGState
from .visualize import visualization_agent


def summary_agent(state: ESGState) -> ESGState:
    mode = str(state.get("mode", "executive"))
    # mode "cross_year:<SDG>" -> grounded three-year narrative for that SDG.
    if mode.startswith("cross_year:"):
        sdg = mode.split(":", 1)[1].strip()
        try:
            from cross_year_analysis import get_cross_year_metrics

            metrics = get_cross_year_metrics(sdg)
            summary = summarize_cross_year(
                metrics,
                mode="executive",
                model=str(state.get("model", "gpt-4.1-mini")),
                api_key=state.get("api_key", ""),
            )
            return {"summary": summary}
        except Exception as exc:
            errors = list(state.get("errors", [])) + [f"cross_year summary: {exc}"]
            return {"summary": summarize_with_optional_llm(state, mode="executive"), "errors": errors}
    summary = summarize_with_optional_llm(state, mode=mode)
    return {"summary": summary}


def qa_agent(state: ESGState) -> ESGState:
    answer = answer_with_optional_llm(
        state.get("user_question", ""),
        state.get("retrieved_chunks", []),
        model=str(state.get("model", "gpt-4.1-mini")),
        api_key=state.get("api_key", ""),
    )
    return {"qa_answer": answer}


PIPELINE: list[Callable[[ESGState], ESGState]] = [
    preprocess_agent,
    keyword_agent,
    classification_agent,
    visualization_agent,
    summary_agent,
]


def _merge_state(state: ESGState, update: ESGState) -> ESGState:
    merged = dict(state)
    merged.update(update)
    return merged


def _run_sequential(initial_state: ESGState, nodes: list[Callable[[ESGState], ESGState]]) -> ESGState:
    state = dict(initial_state)
    errors = list(state.get("errors", []))
    for node in nodes:
        try:
            state = _merge_state(state, node(state))
        except Exception as exc:
            errors.append(f"{node.__name__}: {exc}")
            state["errors"] = errors
    return state


def _build_langgraph():
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(ESGState)
    builder.add_node("preprocess", preprocess_agent)
    builder.add_node("extract_keywords", keyword_agent)
    builder.add_node("classify_esg", classification_agent)
    builder.add_node("build_visualizations", visualization_agent)
    builder.add_node("summarize_insights", summary_agent)
    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "extract_keywords")
    builder.add_edge("extract_keywords", "classify_esg")
    builder.add_edge("classify_esg", "build_visualizations")
    builder.add_edge("build_visualizations", "summarize_insights")
    builder.add_edge("summarize_insights", END)
    return builder.compile()


def run_analysis(
    raw_text: str = "",
    year: str = "2024",
    top_n: int = 40,
    summary_mode: str = "executive",
    source: str = "",
    api_key: str = "",
    model: str = "",
) -> ESGState:
    initial_state: ESGState = {
        "raw_text": raw_text,
        "year": str(year),
        "source": source,
        "mode": summary_mode,
        "api_key": api_key,
        "model": model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "chart_data": {"top_n": top_n},
        "qa_scope": "cross_year",
        "errors": [],
    }
    try:
        graph = _build_langgraph()
        return graph.invoke(initial_state)
    except Exception as exc:
        state = _run_sequential(initial_state, PIPELINE)
        state["errors"] = list(state.get("errors", [])) + [f"LangGraph fallback mode: {exc}"]
        return state


def answer_question(
    state: ESGState,
    question: str,
    scope: str = "cross_year",
    years: list[str] | None = None,
) -> ESGState:
    qa_state = dict(state)
    qa_state["user_question"] = question
    qa_state["qa_scope"] = scope
    if years:
        qa_state["qa_years"] = [str(y) for y in years]
    return _run_sequential(qa_state, [retrieval_agent, qa_agent])
