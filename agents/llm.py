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


def answer_with_optional_llm(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    model: str = "gpt-4.1-mini",
    api_key: str | None = None,
) -> str:
    if not retrieved_chunks:
        return "I could not find a strong matching section. Try asking about climate, water, suppliers, employees, governance, or risk."
    context = "\n\n".join(f"[Chunk {item['chunk_id']}]\n{item['text']}" for item in retrieved_chunks)
    prompt = f"""
Answer the question using only the provided TSMC ESG report chunks.

Question:
{question}

Context:
{context}

Return:
1. A short answer.
2. 2-3 evidence points.
3. Mention the chunk ids used.
"""
    llm_answer = generate_text(prompt, model=model, api_key=api_key)
    if llm_answer:
        return llm_answer
    evidence = "\n\n".join(f"- Chunk {item['chunk_id']} (score {item['score']}): {item['text'][:260]}..." for item in retrieved_chunks[:3])
    return f"Relevant evidence was found for this question. Without an API key, here are the closest chunks:\n\n{evidence}"
