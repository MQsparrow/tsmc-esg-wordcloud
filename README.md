# Decoding TSMC's Sustainability Language

A text-mining dashboard that reveals how TSMC frames ESG topics across Strategic Framing and UN SDG themes — now spanning the 2022, 2023, and 2024 sustainability reports.

---

## Overview

This project turns TSMC's sustainability reports (2022, 2023, 2024 — 200+ pages each) into an interactive text-mining dashboard. Instead of reading the full reports, we parse their language computationally — mapping vocabulary patterns to two universal frameworks:

* **Strategic Framing** — how the report distributes language across ESG sections (Environment, Talent, Supply Chain, Social, Governance)
* **UN SDGs (2030 Agenda)** — which Sustainable Development Goals the language aligns to (9 themes mapped)

The result is an auditable, comparable language profile that reveals what TSMC emphasizes, what it downplays, and where its narrative overlaps across themes — and how that emphasis shifts year over year.

---

## What Changed in This Version

This release adds **multi-year coverage, semantic embeddings, and two new analysis pages**:

### Added
* **Multi-year pipeline** — processes 2022, 2023, and 2024 reports. Each year's outputs land in its own subdirectory (`outputs/2022/`, `outputs/2023/`, `outputs/2024/`).
* **Sentence embeddings** — every chunk is encoded with `sentence-transformers/all-MiniLM-L6-v2` (a lightweight BERT-style model) and saved as `outputs/{year}/chunk_embeddings.npz`.
* **Semantic Embedding Explorer** (Analysis page) — PCA scatter plot of chunks colored by section or primary SDG, nearest-neighbor lookup for any chunk, and group centroid similarity heatmap.
* **Kiwi's Week 9 Q1 page** — phrase mining, keyword co-occurrence networks, TF-IDF group comparison + chunk clustering, ESG tone heatmap, keyword context explorer.
* **Kiwi's Week 10 Q2 page** — Word2Vec embeddings trained on the report chunks: similar-word lookup, 2-D embedding map, group similarity, representative chunks.
* **Modular page structure** — pages now live as separate modules under `app_pages/` (project_overview, analysis, methods, week9_q1, week10_q2).

### Previously (simplification release)
Compared to the earlier version, an earlier release focused on **simplification and presentation clarity**:

### Removed
* **Orientation dimension** (action-oriented / people-centric / mixed) — removed entirely from the pipeline, analysis, and UI. The section-level framing already captures strategic differences more clearly.
* **Issue Frame dimension** (environment, labor/talent, supply chain, innovation, governance/risk) — removed because it overlapped heavily with the section categories. Keeping both added complexity without adding insight.
* **Co-occurrence network visualizations** — removed from the main Analysis flow along with issue frames (later reintroduced as part of the Week 9 page).
* **Separate Dashboard and Explorer pages** — merged into a single Analysis page.
* **Top Keywords Table** — removed from explorer tabs to reduce clutter (bar chart already shows the same data).

### Added / Changed
* **Unified Analysis page** with a theme mode switcher (Strategic Framing / UN SDG Themes). Each mode follows the same flow: Distribution → Per-category tabs (word cloud + top keywords) → Top 2 synthesis → Validation.
* **Validation section** — keyword overlap heatmap and cosine similarity heatmap shown side-by-side with interpretation, plus example chunks. This makes it easy to cross-check results.
* **Improved chunk cards** — term highlighting with regex-based matching, key sentence extraction with sliding window, side-by-side layout.
* **Stopwords updated** — added `fab` and `semiconductor` as domain-specific stopwords (too generic for a TSMC report to be informative).
* **Consistent terminology** — "Strategic Framing" and "UN SDGs" used consistently across all pages.
* **Smaller, cleaner metric cards** — reduced visual weight for better page balance.
* **Project Overview redesigned** — now serves as "the pitch" with metric cards (200+ pages → 6 sections → 9 themes), bullet-point explanation cards, and a signature word cloud.

### Page Structure (before → after)

| Before | After |
|---|---|
| Project Overview | Project Overview (the pitch) |
| Methods | Methods (moved to last) |
| Dashboard | Analysis (merged) |
| Explorer | _(merged into Analysis)_ |

---

## Key Features

* **Multi-year coverage** — 2022, 2023, 2024 reports processed in one pipeline run, with per-year output directories enabling year-over-year comparisons

* **Strategic Framing analysis** — 5 ESG sections: Environment, Talent, Supply Chain, Social, Governance

* **UN SDG classification** — 9 SDG themes with dominance control:
  * SDG 3 – Health, SDG 4 – Education, SDG 6 – Water, SDG 7 – Energy
  * SDG 8 – Labor, SDG 9 – Innovation, SDG 12 – Consumption
  * SDG 13 – Climate, SDG 17 – Partnership

  Each chunk is assigned at most 2 SDGs using keyword scoring, word-boundary matching, a dominance rule, and confidence scores.

* **Sentence embeddings (BERT-style)** — `all-MiniLM-L6-v2` chunk embeddings power the Semantic Embedding Explorer: PCA projection, nearest-neighbor retrieval, group centroid similarity

* **Word2Vec (gensim)** — locally trained word embeddings on report chunks, surfaced through similar-word lookup, embedding maps, and semantic group similarity (Week 10 page)

* **Phrase mining & co-occurrence** — bigram/trigram extraction, keyword co-occurrence graphs, TF-IDF clustering, ESG tone heatmaps, keyword context explorer (Week 9 page)

* **Visualizations** — word clouds, top keyword bar charts, distribution charts, overlap heatmaps, cosine similarity heatmaps, representative chunk cards with term highlighting

---

## Pipeline

```
PDF Extraction (per year) → Cleanup → Chunking → spaCy Preprocessing
  → Rule-based Labels → TF-IDF & Cosine Similarity
  → Sentence Embeddings (MiniLM)
  → Per-year outputs/{year}/
```

| Step | Details |
|---|---|
| **PDF Extraction** | `src/extract_pdf.py` extracts each report (2022/2023/2024) to `data/raw/tsmc_report_{year}.txt` |
| **Data Preparation** | Clean PDF noise, remove headers/boilerplate, paragraph-based chunking, spaCy lemmatization, custom stopwords, POS filtering |
| **Labeling** | Strategic Framing and UN SDG labels via rule-based logic. SDG uses stronger thresholds and a dominance rule |
| **Analysis** | TF-IDF across Strategic Framing and UN SDGs. Cosine similarity for cross-group comparison |
| **Embeddings** | Each chunk encoded with `sentence-transformers/all-MiniLM-L6-v2`, normalized, saved to `outputs/{year}/chunk_embeddings.npz` |
| **Validation** | Audit layer checks duplicates, short chunks, repeated patterns, SDG label counts |

---

## Project Structure

```
.
├── app.py                       # Streamlit entry point + shared layout
├── main.py                      # Multi-year pipeline runner
├── app_pages/                   # Modular page implementations
│   ├── project_overview.py      #   "the pitch" landing page
│   ├── analysis.py              #   Strategic Framing / UN SDG analysis + embedding explorer
│   ├── methods.py               #   pipeline walkthrough
│   ├── week9_q1.py              #   phrase mining, co-occurrence, TF-IDF clustering, tone heatmap
│   └── week10_q2.py             #   Word2Vec (similar terms, map, group similarity)
├── src/
│   ├── extract_pdf.py           # Multi-year PDF → text extraction
│   ├── pipeline.py              # Core NLP pipeline, classification, TF-IDF, multi-year orchestration
│   ├── embeddings.py            # MiniLM sentence-embedding pipeline
│   ├── preprocess.py            # Chunking and tokenization
│   └── analysis.py              # Diagnostic: keyword audit
├── outputs/
│   ├── 2022/                    # Per-year results
│   │   ├── chunks_processed.csv
│   │   ├── tfidf_by_section.csv
│   │   ├── tfidf_by_sdg.csv
│   │   ├── similarity_by_section.csv
│   │   ├── similarity_by_sdg.csv
│   │   └── chunk_embeddings.npz
│   ├── 2023/                    #   (same layout)
│   └── 2024/                    #   (same layout)
├── data/
│   └── raw/                     # PDFs: e-all_2022.pdf, e-all_2023.pdf, 2024-TSMC-Sustainability-Report-e.pdf
├── pic/                         # Screenshots used in the app
└── requirements.txt
```

---

## How to Run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Place PDFs in data/raw/:
#   e-all_2022.pdf
#   e-all_2023.pdf
#   2024-TSMC-Sustainability-Report-e.pdf

python main.py --audit          # extracts all PDFs, runs the pipeline for each year, generates embeddings
streamlit run app.py
```

The first pipeline run downloads the `all-MiniLM-L6-v2` model (~90 MB) on demand. Subsequent runs reuse the cached model.

---

## Demo Flow

Recommended presentation flow:

1. **Project Overview** — show the pitch: what we decode, why it matters, signature word cloud
2. **Analysis → Strategic Framing** — distribution, per-section word clouds, top 2 synthesis, validation heatmaps
3. **Analysis → UN SDG Themes** — same flow for SDG dimension
4. **Analysis → Semantic Embedding Explorer** — PCA scatter, nearest-neighbor lookup, centroid heatmap (per year)
5. **Kiwi's Week 9 Q1** — phrase mining, co-occurrence networks, TF-IDF clustering, ESG tone heatmap
6. **Kiwi's Week 10 Q2** — Word2Vec similar terms, embedding map, semantic group similarity
7. **Methods** — pipeline walkthrough (if audience asks "how?")

---

## Key Insights

* **Talent** and **Environment** are the two largest sections by chunk count
* **SDG 17 (Partnership)** dominates — driven by compliance infrastructure (committee, carbon reduction, chemical), not goodwill language
* **SDG 4 (Education)** ranks high but top terms overlap with talent retention and social welfare — the keyword boundary is leaking
* **Environment** and **Supply Chain** share the most keywords (supplier, risk, chain, management) — Scope 3 compliance forces environmental language into supply chain territory
* **Talent** and **Social** have the highest cosine similarity despite different top keywords — structurally similar language, topically distinct
* `fab` and `semiconductor` filtered as stopwords — too generic for a TSMC report

---

## Tech Stack

* Python, pandas, numpy
* spaCy (preprocessing)
* scikit-learn (TF-IDF, cosine similarity, PCA, KMeans)
* sentence-transformers (`all-MiniLM-L6-v2` chunk embeddings)
* gensim (Word2Vec for the Week 10 page)
* networkx (keyword co-occurrence graphs)
* pdfplumber (PDF extraction)
* Streamlit + streamlit-echarts
* Plotly
* WordCloud

---

## Author

Text Mining Project – Team 9
