from __future__ import annotations

from .state import ESGState


ESG_COLORS = {
    "Environmental": "#14b8a6",
    "Social": "#f59e0b",
    "Governance": "#6366f1",
    "Other": "#94a3b8",
}


def visualization_agent(state: ESGState) -> ESGState:
    keywords = state.get("keywords", [])
    counts = state.get("esg_counts", {})
    wordcloud_freq = {
        str(item.get("term")): float(item.get("score") or item.get("frequency") or 1)
        for item in keywords
        if item.get("term")
    }
    chart_data = dict(state.get("chart_data", {}) or {})
    chart_data.update(
        {
            "wordcloud_freq": wordcloud_freq,
            "esg_counts": counts,
            "esg_colors": ESG_COLORS,
            "keyword_rows": keywords,
        }
    )
    return {"chart_data": chart_data}
