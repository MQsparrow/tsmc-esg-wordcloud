from __future__ import annotations


def render_page(
    *,
    df_chunks,
    df_section_tfidf,
    df_sdg_tfidf,
    render_page_header,
    render_metric_grid,
    render_card_grid,
    aggregate_terms_for_overview,
    render_interactive_wordcloud,
    st,
) -> None:
    render_page_header(
        "The Pitch",
        "Project Overview",
        "",
    )

    render_metric_grid(
        [
            {"label": "Original Report", "value": "200+ pages"},
            {"label": "Strategic Framing", "value": f"{df_chunks['section_label'].nunique()} sections"},
            {
                "label": "UN 2030 Agenda",
                "value": f'{df_chunks["sdg_labels"].str.split(",").explode().str.strip().pipe(lambda s: s[s != "unclassified"]).nunique()} themes',
            },
        ],
        columns=3,
    )
    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    render_card_grid(
        [
            {
                "title": "What We Decode",
                "text": "<ul style='margin:0;padding-left:1.2rem;color:#475569;line-height:1.8;font-size:0.95rem'>"
                "<li>Source: <strong>TSMC 2024 Sustainability Report</strong> — environment, talent, supply chain, social impact, governance</li>"
                "<li>Mapped to two universal frameworks: <strong>Strategic Framing</strong> and the <strong>UN SDGs (2030 Agenda)</strong></li>"
                "<li>Every finding speaks a language investors, regulators, and academics already understand</li>"
                "</ul>",
                "highlight": True,
            },
            {
                "title": "Why It Matters",
                "text": "<ul style='margin:0;padding-left:1.2rem;color:#475569;line-height:1.8;font-size:0.95rem'>"
                "<li>ESG reports are <strong>strategic narratives</strong>, not neutral disclosure</li>"
                "<li>We quantify vocabulary patterns to reveal where TSMC invests narrative weight "
                "(e.g. climate compliance, supplier oversight) vs. where it stays generic (e.g. labor rights, innovation IP)</li>"
                "<li>Result: a dense PDF becomes an <strong>auditable, comparable language profile</strong></li>"
                "</ul>",
            },
        ],
        columns=2,
    )
    st.markdown("")

    st.markdown("### What Do You See Before We Decode?")
    st.caption("A first glance at the report's most distinctive vocabulary — sized by TF-IDF weight. The patterns become clearer in the Analysis page.")
    overview_wordcloud_terms = aggregate_terms_for_overview(
        df_section_tfidf,
        df_sdg_tfidf,
        top_k=70,
    )
    render_interactive_wordcloud(overview_wordcloud_terms, top_k=70)

