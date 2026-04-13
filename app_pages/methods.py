from __future__ import annotations


def render_page(
    *,
    render_page_header,
    render_pipeline_steps,
    render_card_grid,
    st,
) -> None:
    render_page_header(
        "Methods",
        "How The Pipeline Works",
        "A concise walkthrough of how raw report text becomes a cleaned, labeled, and interpretable dashboard.",
    )

    st.markdown("### Pipeline Flow")
    st.markdown(
        '<div class="section-note">The workflow keeps each transformation explainable. We move from raw PDF text to cleaned chunks, then add labels and analysis layers only after the text is stable enough to defend.</div>',
        unsafe_allow_html=True,
    )
    render_pipeline_steps(
        [
            {
                "number": "Step 1",
                "title": "Extract",
                "text": "Pull text from the PDF and preserve enough structure to trace the source before cleanup begins.",
            },
            {
                "number": "Step 2",
                "title": "Clean",
                "text": "Remove page markers, repeated headers, appendix boilerplate, and leftover PDF artifacts that would distort later metrics.",
            },
            {
                "number": "Step 3",
                "title": "Chunk",
                "text": "Split the report into paragraph-level units and drop noisy fragments so each chunk holds a cleaner semantic idea.",
            },
            {
                "number": "Step 4",
                "title": "Preprocess",
                "text": "Use spaCy for normalization, lemmatization, stopword removal, and POS filtering so grouped language is comparable.",
            },
            {
                "number": "Step 5",
                "title": "Label",
                "text": "Assign Strategic Framing, orientation, SDG, and issue-frame labels with rule-based logic, stronger thresholds, and confidence-aware scoring.",
            },
            {
                "number": "Step 6",
                "title": "Analyze",
                "text": "Generate TF-IDF, cosine similarity, co-occurrence networks, and evidence views that feed directly into the dashboard.",
            },
        ],
        columns=3,
    )
    st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)

    render_card_grid(
        [
            {
                "title": "Data Preparation",
                "text": "We clean PDF-specific noise, remove repeated headers and boilerplate, then split the report into semantically cleaner chunks. After chunking, spaCy handles normalization, lemmatization, stopword removal, and POS filtering.",
                "highlight": True,
            },
            {
                "title": "Analysis",
                "text": "We aggregate grouped text with TF-IDF to surface distinctive vocabulary across Strategic Framing and UN SDGs. We also compare grouped language with cosine similarity.",
                "highlight": True,
            },
            {
                "title": "Labeling",
                "text": "Each chunk receives Strategic Framing and UN SDG labels using transparent rule-based logic. SDG assignment uses stronger thresholds, a dominance rule, and confidence scores to reduce noisy multi-labeling.",
            },
            {
                "title": "Validation",
                "text": "The pipeline includes an audit layer that checks duplicates, short chunks, repeated first-line patterns, and SDG label counts. This makes the output more reproducible and easier to defend during presentation.",
            },
        ],
        columns=2,
    )
    st.markdown("")

    st.markdown("### Methods by Outcome")
    render_card_grid(
        [
            {
                "title": "Interpretability",
                "text": "Representative chunks and SDG confidence scores support evidence-based explanation.",
                "kicker": "Outcome",
            },
            {
                "title": "Comparability",
                "text": "TF-IDF, overlap heatmaps, and cosine similarity show where categories align or differ in language.",
                "kicker": "Outcome",
            },
            {
                "title": "Presentation Value",
                "text": "Interactive views make the analysis easier to navigate quickly in a live demo.",
                "kicker": "Outcome",
                "highlight": True,
            },
        ],
        columns=3,
    )

