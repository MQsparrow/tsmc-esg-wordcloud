"""Cross-Year Compare page: a top-level Streamlit panel for one SDG across 2022/2023/2024.

Reads only the cached cross-year snapshot (outputs/cross_year_analysis.json via
cross_year_analysis.load_cross_year_payload) — no recomputation in the front end.
"""
from __future__ import annotations

import html
import os
from typing import Any

import pandas as pd
import plotly.express as px

from cross_year_analysis import YEARS, load_cross_year_payload

SDG_DISPLAY = {
    "SDG3_health": "SDG3 Health & Safety",
    "SDG4_education": "SDG4 Education",
    "SDG6_water": "SDG6 Water",
    "SDG7_energy": "SDG7 Energy",
    "SDG8_labor": "SDG8 Labor",
    "SDG9_innovation": "SDG9 Innovation",
    "SDG12_consumption": "SDG12 Responsible Consumption",
    "SDG13_climate": "SDG13 Climate",
    "SDG17_partnership": "SDG17 Partnership",
}

# term-status colors
_C_PERSIST = "#e2e8f0"   # gray  = persistent
_C_NEW = "#bbf7d0"       # green = newly appeared
_C_DROP = "#fecaca"      # red   = dropped


def _get_secret_api_key(st: Any) -> str:
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""
    return os.getenv("OPENAI_API_KEY", "") or str(secret_key or "")


def _cached_payload(st: Any) -> dict[str, Any]:
    @st.cache_data(show_spinner=False)
    def _load() -> dict[str, Any]:
        return load_cross_year_payload()

    return _load()


def _term_status(term: str, sets: dict[str, set]) -> str:
    in22, in23, in24 = term in sets["2022"], term in sets["2023"], term in sets["2024"]
    if in22 and in23 and in24:
        return "persist"
    if in22 and not in24:
        return "drop"
    if (not in22) and (in23 or in24):
        return "new"
    return "neutral"


def _keyword_table_html(top_terms: dict[str, list[str]]) -> str:
    sets = {y: set(top_terms.get(y, [])) for y in YEARS}
    n = max((len(top_terms.get(y, [])) for y in YEARS), default=0)
    head = "".join(f"<th style='padding:6px 10px;text-align:left'>{y}</th>" for y in YEARS)
    body_rows = []
    for i in range(n):
        cells = [f"<td style='padding:4px 8px;color:#94a3b8'>{i+1}</td>"]
        for y in YEARS:
            terms = top_terms.get(y, [])
            if i < len(terms):
                term = terms[i]
                status = _term_status(term, sets)
                bg = {"persist": _C_PERSIST, "new": _C_NEW, "drop": _C_DROP, "neutral": "transparent"}[status]
                deco = "text-decoration:line-through;" if status == "drop" else ""
                cells.append(
                    f"<td style='padding:4px 8px'>"
                    f"<span style='background:{bg};{deco}border-radius:6px;padding:2px 8px;display:inline-block'>"
                    f"{html.escape(term)}</span></td>"
                )
            else:
                cells.append("<td></td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table style='border-collapse:collapse;width:100%;font-size:0.9rem'>"
        f"<thead><tr><th style='padding:6px 10px'>#</th>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def render_page(render_page_header, render_metric_grid, render_card_grid, st) -> None:
    render_page_header(
        "Cross-Year Compare",
        "TSMC ESG Cross-Year Comparison (2022 / 2023 / 2024)",
        "Pick an SDG topic to see, at a glance, the change in quantity (coverage share) and quality (keywords + semantic shift) across the three years.",
    )

    try:
        payload = _cached_payload(st)
    except Exception as exc:
        st.error(
            "Failed to load the cross-year snapshot. Run `python cross_year_analysis.py` from the "
            f"project root to generate outputs/cross_year_analysis.json. Error: {exc}"
        )
        return

    sdgs = payload["sdgs"]
    with st.sidebar:
        st.markdown("### Cross-Year Compare")
        sdg = st.selectbox("SDG topic", sdgs, format_func=lambda s: SDG_DISPLAY.get(s, s), key="cy_sdg")

    cov_row = next((r for r in payload["coverage"]["rows"] if r["sdg"] == sdg), None)
    ks = payload["keyword_shift"][sdg]
    cs = payload["centroid_shift"].get(sdg, {})
    totals = payload["coverage"]["totals"]

    # ---- change in quantity: coverage -------------------------------------
    count = cov_row["count"] if cov_row else {y: 0 for y in YEARS}
    share = cov_row["share_pct"] if cov_row else {y: 0 for y in YEARS}
    render_metric_grid(
        [
            {"label": f"{y} coverage", "value": f"{share[y]:.1f}%", "note": f"{count[y]} chunks / {totals[y]} total"}
            for y in YEARS
        ]
        + [
            {
                "label": "Δ share 22→24",
                "value": f"{cov_row['delta_share_pp_22_24']:+.1f}pp" if cov_row else "—",
                "note": f"chunks {cov_row['delta_abs_22_24']:+d}" if cov_row else "",
            }
        ]
    )

    # ---- change in quantity: coverage (full width) ------------------------
    st.subheader("Change in quantity | coverage share")
    share_df = pd.DataFrame({"year": YEARS, "share_pct": [share[y] for y in YEARS], "count": [count[y] for y in YEARS]})
    fig = px.bar(share_df, x="year", y="share_pct", text="count", color="year",
                 color_discrete_sequence=["#94a3b8", "#0ea5a4", "#0f766e"])
    fig.update_traces(texttemplate="%{text} chunks", textposition="outside")
    fig.update_layout(height=320, yaxis_title="% of that year's chunks", xaxis_title="", showlegend=False,
                      margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ---- change in quality: semantic centroid shift -----------------------
    st.subheader("Change in quality | semantic centroid shift")
    euclid = cs.get("euclid", {}) if cs else {}
    if euclid:
        render_metric_grid(
            [
                {"label": "Total semantic shift", "value": f"{euclid.get('path_total')}",
                 "note": "sum of both yearly steps: (2022→2023) + (2023→2024)"},
                {"label": "2022 → 2023 shift", "value": f"{euclid.get('d_22_23')}",
                 "note": "how far the topic moved from 2022 to 2023"},
                {"label": "2023 → 2024 shift", "value": f"{euclid.get('d_23_24')}",
                 "note": "how far the topic moved from 2023 to 2024"},
                {"label": "2022 → 2024 (direct)", "value": f"{euclid.get('d_22_24_direct')}",
                 "note": "straight-line distance from 2022 to 2024"},
            ],
            columns=4,
        )
        st.caption(
            "All values are distances between the topic's average embedding (centroid) in each year — higher means the "
            "topic's wording/meaning changed more. When the Total semantic shift is larger than the 2022→2024 direct "
            "distance, the topic did not move in a straight line (it shifted one way, then another)."
        )
    else:
        st.info("No alignable embedding centroid for this topic; semantic shift skipped.")

    # ---- change in quality: keywords --------------------------------------
    st.subheader("Change in quality | Top keywords side by side")
    st.markdown(
        f"<span style='background:{_C_PERSIST};border-radius:6px;padding:2px 8px'>Persistent</span> "
        f"<span style='background:{_C_NEW};border-radius:6px;padding:2px 8px'>New</span> "
        f"<span style='background:{_C_DROP};border-radius:6px;padding:2px 8px;text-decoration:line-through'>Dropped</span>",
        unsafe_allow_html=True,
    )
    st.markdown(_keyword_table_html(ks["top_terms"]), unsafe_allow_html=True)
    st.markdown(
        f"\n**Persistent ({len(ks['persistent'])})**: {', '.join(ks['persistent']) or '(none)'}  \n"
        f"**New ({len(ks['new'])})**: {', '.join(ks['new']) or '(none)'}  \n"
        f"**Dropped ({len(ks['dropped'])})**: {', '.join(ks['dropped']) or '(none)'}"
    )

    # ---- optional grounded narrative (no-key fallback safe) ----------------
    st.subheader("Cross-year narrative")
    api_key = (st.text_input("OpenAI API key (optional; leave blank for the deterministic narrative)", type="password", key="cy_key").strip()
               or _get_secret_api_key(st))
    if st.button("Generate cross-year narrative", key="cy_narrative"):
        from agents.llm import summarize_cross_year
        from cross_year_analysis import get_cross_year_metrics

        with st.spinner("Generating grounded narrative..."):
            metrics = get_cross_year_metrics(sdg)
            text = summarize_cross_year(metrics, mode="executive", api_key=api_key)
        st.markdown(text)
        if not api_key:
            st.caption("(No API key: the text above is the deterministic template narrative, restating the data only.)")

    with st.expander("Cross-year consistency notes"):
        for note in payload["consistency_notes"]:
            st.caption(note)
