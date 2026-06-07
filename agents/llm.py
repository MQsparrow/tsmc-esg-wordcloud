from __future__ import annotations

import os
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _resolve_api_key(api_key: str | None = None) -> str:
    return (api_key or os.getenv("OPENAI_API_KEY") or "").strip()


def has_openai_key(api_key: str | None = None) -> bool:
    return bool(_resolve_api_key(api_key))


def _try_langchain(prompt: str, model: str, api_key: str | None = None) -> str | None:
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, temperature=0.2, api_key=_resolve_api_key(api_key))
        response = llm.invoke(prompt)
        return str(getattr(response, "content", response)).strip()
    except Exception:
        return None


def _try_openai(prompt: str, model: str, api_key: str | None = None) -> str | None:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=_resolve_api_key(api_key))
        response = client.responses.create(model=model, input=prompt)
        return str(response.output_text).strip()
    except Exception:
        return None


def generate_text(prompt: str, model: str = "gpt-4.1-mini", api_key: str | None = None) -> str | None:
    if not has_openai_key(api_key):
        return None
    return _try_langchain(prompt, model, api_key=api_key) or _try_openai(prompt, model, api_key=api_key)


def fallback_summary(state: dict[str, Any]) -> str:
    counts = state.get("esg_counts", {}) or {}
    keywords = state.get("keywords", [])[:8]
    top_terms = ", ".join(str(item.get("term", "")) for item in keywords if item.get("term"))
    dominant = max(counts.items(), key=lambda item: item[1])[0] if counts else "Environmental"
    return (
        f"The report is most strongly represented by {dominant} content in the current chunk-level analysis. "
        f"Important terms include {top_terms or 'ESG-related operational and sustainability terms'}. "
        "The dashboard combines deterministic text mining with optional LLM summarization so the demo remains stable "
        "even when an API key is not available."
    )


def summarize_with_optional_llm(state: dict[str, Any], mode: str = "executive", model: str = "gpt-4.1-mini") -> str:
    api_key = state.get("api_key", "")
    model = str(state.get("model", model) or model)
    counts = state.get("esg_counts", {}) or {}
    keywords = state.get("keywords", [])[:15]
    prompt = f"""
You are helping present a text mining final project about TSMC ESG reports.

Write a concise {mode} summary based only on these analysis results.

ESG counts:
{counts}

Top keywords:
{keywords}

Requirements:
- Mention the strongest ESG theme.
- Mention 3-5 notable keywords.
- Keep it presentation-ready.
- Do not invent facts outside the provided analysis.
"""
    return generate_text(prompt, model=model, api_key=api_key) or fallback_summary(state)


def _chunk_tag(item: dict[str, Any]) -> str:
    year = str(item.get("year", "")).strip()
    cid = item.get("chunk_id")
    return f"Year {year} | Chunk {cid}" if year else f"Chunk {cid}"


def answer_with_optional_llm(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    model: str = "gpt-4.1-mini",
    api_key: str | None = None,
) -> str:
    if not retrieved_chunks:
        return "I could not find a strong matching section. Try asking about climate, water, suppliers, employees, governance, or risk."
    years_present = sorted({str(item.get("year", "")).strip() for item in retrieved_chunks if str(item.get("year", "")).strip()})
    multi_year = len(years_present) > 1
    context = "\n\n".join(f"[{_chunk_tag(item)}]\n{item['text']}" for item in retrieved_chunks)
    cross_year_guidance = (
        "- The evidence spans multiple report years ("
        + ", ".join(years_present)
        + "). When the question is about change over time, compare the years explicitly and attribute each point to its year.\n"
        if multi_year else ""
    )
    prompt = f"""
Answer the question using ONLY the provided TSMC ESG report chunks. Each chunk is tagged with the report year it came from.

Question:
{question}

Context:
{context}

Rules:
- Use only the information in the chunks above. Do not invent facts or motivations not present in the text.
{cross_year_guidance}- If the chunks do not contain enough to answer, say so plainly.

Return:
1. A short answer.
2. 2-3 evidence points, each citing its year and chunk id.
3. List the (year, chunk id) pairs you used.
"""
    llm_answer = generate_text(prompt, model=model, api_key=api_key)
    if llm_answer:
        return llm_answer
    evidence = "\n\n".join(f"- [{_chunk_tag(item)}] (score {item['score']}): {item['text'][:260]}..." for item in retrieved_chunks[:4])
    span = f" across {', '.join(years_present)}" if multi_year else (f" ({years_present[0]})" if years_present else "")
    return (
        f"Relevant evidence was found for this question{span}. "
        f"Without an API key, here are the closest report chunks (each tagged with its year):\n\n{evidence}"
    )


def _fmt_terms(terms: list[str], limit: int = 8) -> str:
    return ", ".join(terms[:limit]) if terms else "(none)"


def cross_year_fallback_summary(metrics: dict[str, Any]) -> str:
    """Deterministic, no-LLM narrative built strictly from the cross-year metrics dict."""
    sdg = metrics.get("sdg", "")
    cov = metrics.get("coverage") or {}
    ks = metrics.get("keyword_shift") or {}
    cs = (metrics.get("centroid_shift") or {}).get("euclid") or {}
    count = cov.get("count", {})
    share = cov.get("share_pct", {})
    years = metrics.get("years", ["2022", "2023", "2024"])
    parts = [f"**{sdg} — cross-year comparison (metrics only; no motive interpretation)**", ""]
    if count and share:
        parts.append(
            "Coverage: " + " -> ".join(f"{y} {count.get(y, 0)} chunks ({share.get(y, 0)}%)" for y in years)
            + f"; 2022->2024 change {cov.get('delta_abs_22_24', 0):+d} chunks, share {cov.get('delta_share_pp_22_24', 0):+.1f}pp."
        )
    parts.append(f"Persistent keywords: {_fmt_terms(ks.get('persistent', []))}.")
    parts.append(f"Newly appeared: {_fmt_terms(ks.get('new', []))}.")
    parts.append(f"Dropped: {_fmt_terms(ks.get('dropped', []))}.")
    if cs:
        parts.append(
            f"Semantic centroid shift: 22->23 {cs.get('d_22_23')}, 23->24 {cs.get('d_23_24')}, "
            f"path total {cs.get('path_total')} (Euclidean)."
        )
    return "\n".join(parts)


def summarize_cross_year(
    metrics: dict[str, Any],
    mode: str = "executive",
    model: str = "gpt-4.1-mini",
    api_key: str | None = None,
) -> str:
    """Generate a cross-year narrative for one SDG, grounded strictly in `metrics`.

    `metrics` is the dict returned by cross_year_analysis.get_cross_year_metrics(sdg).
    The LLM is instructed to describe only the supplied numbers, not to speculate.
    """
    sdg = metrics.get("sdg", "")
    prompt = f"""
You are reporting cross-year (2022/2023/2024) text-mining results for the TSMC ESG topic {sdg}.

Write a concise {mode} narrative describing ONLY the metrics below. These are deterministic
text-mining outputs (TF-IDF keyword sets, chunk-coverage shares, embedding centroid distances).

Metrics (JSON):
{metrics}

Strict rules:
- Describe only what the numbers show: how the coverage share changed, which keywords are
  persistent / newly appeared / dropped, and the size of the semantic centroid shift.
- Do NOT speculate about the company's motivations, strategy, or causes.
- Do NOT introduce facts, events, or keywords that are not in the metrics.
- Attribute every quantitative claim to its year(s).
- Keep it presentation-ready.
"""
    return generate_text(prompt, model=model, api_key=api_key) or cross_year_fallback_summary(metrics)
