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


def _try_langchain(prompt: str, model: str, api_key: str | None = None) -> tuple[str | None, str]:
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=model, temperature=0.2, api_key=_resolve_api_key(api_key))
        response = llm.invoke(prompt)
        return str(getattr(response, "content", response)).strip(), ""
    except Exception as exc:
        return None, f"langchain_openai: {type(exc).__name__}: {str(exc)[:180]}"


def _try_openai(prompt: str, model: str, api_key: str | None = None) -> tuple[str | None, str]:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=_resolve_api_key(api_key))
        response = client.responses.create(model=model, input=prompt)
        return str(response.output_text).strip(), ""
    except Exception as exc:
        return None, f"openai: {type(exc).__name__}: {str(exc)[:180]}"


def generate_text_ex(prompt: str, model: str = "gpt-4.1-mini", api_key: str | None = None) -> tuple[str | None, str]:
    """Return (text, error). error is '' on success, 'no_key' when no key is configured,
    otherwise the underlying LLM error string. Tries langchain first, then the openai SDK."""
    if not has_openai_key(api_key):
        return None, "no_key"
    text, err1 = _try_langchain(prompt, model, api_key=api_key)
    if text:
        return text, ""
    text, err2 = _try_openai(prompt, model, api_key=api_key)
    if text:
        return text, ""
    return None, " | ".join(e for e in (err1, err2) if e)


def generate_text(prompt: str, model: str = "gpt-4.1-mini", api_key: str | None = None) -> str | None:
    text, _ = generate_text_ex(prompt, model, api_key=api_key)
    return text


def fallback_summary(state: dict[str, Any]) -> str:
    counts = state.get("esg_counts", {}) or {}
    keywords = state.get("keywords", [])[:8]
    top_terms = ", ".join(str(item.get("term", "")) for item in keywords if item.get("term"))
    dominant = max(counts.items(), key=lambda item: item[1])[0] if counts else "Environmental"
    return (
        f"Based on chunk-level analysis, this report is most strongly represented by {dominant} content. "
        f"Frequently emphasized terms include {top_terms or 'ESG-related operational and sustainability terms'}. "
        "(Generated without an API key — add one for a fuller AI-written summary.)"
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
    llm_answer, err = generate_text_ex(prompt, model=model, api_key=api_key)
    if llm_answer:
        return llm_answer
    evidence = "\n\n".join(f"- [{_chunk_tag(item)}] (score {item['score']}): {item['text'][:260]}..." for item in retrieved_chunks[:4])
    span = f" across {', '.join(years_present)}" if multi_year else (f" ({years_present[0]})" if years_present else "")
    if err == "no_key":
        reason = "Without an API key, here are the closest report chunks (each tagged with its year):"
    else:
        reason = (
            f"An API key was provided but the LLM call failed, so showing retrieved evidence instead.\n\n"
            f"**LLM error:** {err}\n\nClosest report chunks (each tagged with its year):"
        )
    return f"Relevant evidence was found for this question{span}. {reason}\n\n{evidence}"


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
    cov = metrics.get("coverage") or {}
    ks = metrics.get("keyword_shift") or {}
    cs = (metrics.get("centroid_shift") or {}).get("euclid") or {}
    count = cov.get("count", {})
    share = cov.get("share_pct", {})
    years = metrics.get("years", ["2022", "2023", "2024"])

    coverage_line = "; ".join(f"{y}: {count.get(y, 0)} chunks ({share.get(y, 0)}% of that year)" for y in years)
    coverage_line += (f"; net change 2022->2024 {cov.get('delta_abs_22_24', 0):+d} chunks "
                      f"({cov.get('delta_share_pp_22_24', 0):+.1f} percentage points of share).")
    shift_line = (
        f"moved {cs.get('d_22_23')} from 2022 to 2023, then {cs.get('d_23_24')} from 2023 to 2024 "
        f"(total path {cs.get('path_total')}), vs. a direct 2022->2024 distance of {cs.get('d_22_24_direct')}"
        if cs else "not available"
    )

    prompt = f"""
You are helping present a text mining final project about TSMC ESG reports.

Write a concise, flowing {mode} narrative about how the topic {sdg} changed across 2022, 2023, and 2024,
based only on these cross-year analysis results.

Coverage (how much of each report this topic occupies):
{coverage_line}

Keyword change (TF-IDF top terms):
- Stayed all three years: {_fmt_terms(ks.get('persistent', []), 12)}
- Newly prominent by 2024: {_fmt_terms(ks.get('new', []), 12)}
- Faded out by 2024: {_fmt_terms(ks.get('dropped', []), 12)}

Semantic shift (how far the topic's average wording moved, embedding distance):
{shift_line}

Requirements:
- Open with the single most notable change, not with a list of numbers.
- Tell it as one connected story: tie the keyword turnover to the semantic shift (what kind of
  vocabulary came in vs. went out, and whether the meaning moved a lot or a little).
- If the total path is clearly larger than the direct 2022->2024 distance, point out the topic
  moved non-linearly (one direction, then another).
- Use a few representative keywords as evidence; weave the key figures into prose, don't list them all.
- Keep it presentation-ready (2 short paragraphs).
- Describe WHAT changed, not WHY. Do not invent facts, events, products, or company motives beyond the analysis above.
"""
    return generate_text(prompt, model=model, api_key=api_key) or cross_year_fallback_summary(metrics)
