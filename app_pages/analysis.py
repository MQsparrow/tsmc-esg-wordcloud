from __future__ import annotations


def render_page(
    *,
    theme_mode,
    top_k,
    df_chunks,
    filtered_chunks,
    df_section_tfidf,
    df_sdg_tfidf,
    df_section_similarity,
    df_sdg_similarity,
    render_page_header,
    render_metric_grid,
    render_interpretation,
    plot_distribution_bar,
    plot_top_terms,
    render_interactive_wordcloud,
    build_overlap_heatmap,
    plot_heatmap,
    summarize_similarity_pairs,
    render_chunk_cards,
    explode_sdg_chunks,
    filter_chunks_by_sdg,
    SECTION_LABEL_MAP,
    SECTION_ORDER,
    SECTION_NARRATIVE,
    SDG_LABEL_MAP,
    SDG_ORDER,
    SDG_NARRATIVE,
    st,
) -> None:
    render_page_header(
        "The Evidence",
        "Analysis",
        "Switch between Strategic Framing and UN SDG to explore each theme end-to-end.",
    )

    render_metric_grid(
        [
            {"label": "Total Chunks", "value": len(df_chunks)},
            {"label": "Strategic Framing", "value": df_chunks["section_label"].nunique()},
            {
                "label": "UN SDG Themes",
                "value": df_chunks["sdg_labels"].str.split(",").explode().str.strip().pipe(lambda s: s[s != "unclassified"]).nunique(),
            },
        ],
        columns=3,
    )

    if theme_mode == "Strategic Framing":
        st.markdown("### Distribution")
        plot_distribution_bar(
            df_chunks,
            "section_label",
            "Chunk Count by Strategic Framing",
            chart_key="analysis_distribution_section",
        )

        st.markdown("### Per-Section Analysis")
        tabs = st.tabs([SECTION_LABEL_MAP.get(x, x) for x in SECTION_ORDER])

        for tab, section in zip(tabs, SECTION_ORDER):
            with tab:
                df_terms = df_section_tfidf[df_section_tfidf["section_label"] == section].copy()
                count_section = len(df_chunks[df_chunks["section_label"] == section])
                st.subheader(f"{SECTION_LABEL_MAP.get(section, section)} ({count_section} chunks)")
                render_interpretation(
                    SECTION_LABEL_MAP.get(section, section),
                    SECTION_NARRATIVE.get(section, ""),
                )

                left, right = st.columns([1.2, 1])
                with left:
                    st.markdown("**Word Cloud**")
                    render_interactive_wordcloud(df_terms.head(80))
                with right:
                    st.markdown("**Top Keywords**")
                    plot_top_terms(
                        df_terms,
                        title=f"Top TF-IDF Terms: {SECTION_LABEL_MAP.get(section, section)}",
                        top_k=top_k,
                        chart_key=f"section_top_terms_{section}",
                    )

        st.markdown("### Top 2 Synthesis")
        section_counts = df_chunks["section_label"].value_counts()
        top2_sections = section_counts.head(2).index.tolist()

        syn_col1, syn_col2 = st.columns(2)
        for col, section in zip([syn_col1, syn_col2], top2_sections):
            with col:
                top_terms = (
                    df_section_tfidf[df_section_tfidf["section_label"] == section]
                    .sort_values("tfidf_score", ascending=False)
                    .head(3)["term"]
                    .tolist()
                )
                label = SECTION_LABEL_MAP.get(section, section)
                count = int(section_counts.get(section, 0))
                narrative = SECTION_NARRATIVE.get(section, "")
                with st.expander(f"**{label}** — {', '.join(top_terms)} ({count} chunks)", expanded=True):
                    if narrative:
                        st.markdown(narrative)

        st.markdown("---")
        st.markdown("### Validation")
        st.caption("Cross-check the results: do sections share the same buzzwords? Do they speak the same language overall?")

        val_col1, val_col2 = st.columns(2)
        with val_col1:
            st.markdown("**Keyword Overlap**")
            st.caption("How many of each section's top-20 keywords also appear in another section's top-20.")
            heatmap_section = build_overlap_heatmap(
                df_section_tfidf, group_col="section_label", ordered_groups=SECTION_ORDER, top_n=20
            )
            plot_heatmap(heatmap_section, "Top-Term Overlap", chart_key="section_overlap_heatmap")
        with val_col2:
            st.markdown("**Cosine Similarity**")
            st.caption("Overall vocabulary similarity using full TF-IDF vectors — not just top words.")
            plot_heatmap(df_section_similarity, "Cosine Similarity", chart_key="section_similarity_heatmap")

        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            st.markdown("**Most similar pairs**")
            st.dataframe(
                summarize_similarity_pairs(df_section_similarity, top_n=3, ascending=False),
                use_container_width=True, hide_index=True,
            )
        with sim_col2:
            st.markdown("**Most distinct pairs**")
            st.dataframe(
                summarize_similarity_pairs(df_section_similarity, top_n=3, ascending=True),
                use_container_width=True, hide_index=True,
            )

        st.markdown("**Example Chunks**")
        render_interpretation(
            "Evidence",
            "Representative chunks ranked by overlap with top section terms, then by SDG confidence.",
        )
        render_chunk_cards(filtered_chunks, df_section_tfidf, max_chunks=2)

    else:
        st.markdown("### Distribution")
        df_chunks_sdg_exploded = explode_sdg_chunks(df_chunks)
        plot_distribution_bar(
            df_chunks_sdg_exploded,
            "sdg_labels",
            "Chunk Count by SDG Theme",
            chart_key="analysis_distribution_sdg",
        )

        st.markdown("### Per-SDG Analysis")
        tabs = st.tabs([SDG_LABEL_MAP.get(x, x) for x in SDG_ORDER])

        for tab, sdg in zip(tabs, SDG_ORDER):
            with tab:
                df_terms = df_sdg_tfidf[df_sdg_tfidf["sdg_labels"] == sdg].copy()
                count_sdg = len(filter_chunks_by_sdg(df_chunks, sdg))
                st.subheader(f"{SDG_LABEL_MAP.get(sdg, sdg)} ({count_sdg} chunks)")
                render_interpretation(
                    SDG_LABEL_MAP.get(sdg, sdg),
                    SDG_NARRATIVE.get(sdg, ""),
                )

                left, right = st.columns([1.2, 1])
                with left:
                    st.markdown("**Word Cloud**")
                    render_interactive_wordcloud(df_terms.head(80))
                with right:
                    st.markdown("**Top Keywords**")
                    plot_top_terms(
                        df_terms,
                        title=f"Top TF-IDF Terms: {SDG_LABEL_MAP.get(sdg, sdg)}",
                        top_k=top_k,
                        chart_key=f"sdg_top_terms_{sdg}",
                    )

        st.markdown("### Top 2 Synthesis")
        sdg_counts = df_chunks_sdg_exploded["sdg_labels"].value_counts()
        top2_sdgs = sdg_counts.head(2).index.tolist()

        syn_col1, syn_col2 = st.columns(2)
        for col, sdg in zip([syn_col1, syn_col2], top2_sdgs):
            with col:
                top_terms = (
                    df_sdg_tfidf[df_sdg_tfidf["sdg_labels"] == sdg]
                    .sort_values("tfidf_score", ascending=False)
                    .head(3)["term"]
                    .tolist()
                )
                label = SDG_LABEL_MAP.get(sdg, sdg)
                count = int(sdg_counts.get(sdg, 0))
                narrative = SDG_NARRATIVE.get(sdg, "")
                with st.expander(f"**{label}** — {', '.join(top_terms)} ({count} chunks)", expanded=True):
                    if narrative:
                        st.markdown(narrative)

        st.markdown("---")
        st.markdown("### Validation")
        st.caption("Cross-check the results: do SDG themes share the same buzzwords? Do they speak the same language overall?")

        val_col1, val_col2 = st.columns(2)
        with val_col1:
            st.markdown("**Keyword Overlap**")
            st.caption("How many of each SDG's top-20 keywords also appear in another SDG's top-20.")
            heatmap_sdg = build_overlap_heatmap(
                df_sdg_tfidf, group_col="sdg_labels", ordered_groups=SDG_ORDER, top_n=20
            )
            plot_heatmap(heatmap_sdg, "Top-Term Overlap", chart_key="sdg_overlap_heatmap")
        with val_col2:
            st.markdown("**Cosine Similarity**")
            st.caption("Overall vocabulary similarity using full TF-IDF vectors — not just top words.")
            plot_heatmap(df_sdg_similarity, "Cosine Similarity", chart_key="sdg_similarity_heatmap")

        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            st.markdown("**Most similar pairs**")
            st.dataframe(
                summarize_similarity_pairs(df_sdg_similarity, top_n=3, ascending=False),
                use_container_width=True, hide_index=True,
            )
        with sim_col2:
            st.markdown("**Most distinct pairs**")
            st.dataframe(
                summarize_similarity_pairs(df_sdg_similarity, top_n=3, ascending=True),
                use_container_width=True, hide_index=True,
            )

        st.markdown("**Example Chunks**")
        render_interpretation(
            "Evidence",
            "Representative chunks ranked by overlap with top SDG terms, then by SDG confidence.",
        )
        render_chunk_cards(filtered_chunks, df_sdg_tfidf, max_chunks=2)
