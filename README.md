# Decoding TSMC's Sustainability Language

A text-mining dashboard that reveals how TSMC frames ESG topics across Strategic Framing and UN SDG themes.

---

## Overview

This project turns TSMC's 2024 sustainability report (200+ pages) into an interactive text-mining dashboard. Instead of reading the full report, we parse its language computationally — mapping vocabulary patterns to two universal frameworks:

* **Strategic Framing** — how the report distributes language across ESG sections (Environment, Talent, Supply Chain, Social, Governance)
* **UN SDGs (2030 Agenda)** — which Sustainable Development Goals the language aligns to (9 themes mapped)

The result is an auditable, comparable language profile that reveals what TSMC emphasizes, what it downplays, and where its narrative overlaps across themes.

---

## What Changed in This Version

Compared to the previous version, this release focuses on **simplification and presentation clarity**:

### Removed
* **Orientation dimension** (action-oriented / people-centric / mixed) — removed entirely from the pipeline, analysis, and UI. The section-level framing already captures strategic differences more clearly.
* **Issue Frame dimension** (environment, labor/talent, supply chain, innovation, governance/risk) — removed because it overlapped heavily with the section categories. Keeping both added complexity without adding insight.
* **Co-occurrence network visualizations** — removed along with issue frames. The related `networkx`, `pyvis`, `plotly.graph_objects` dependencies are no longer needed.
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

* **Strategic Framing analysis** — 5 ESG sections: Environment, Talent, Supply Chain, Social, Governance

* **UN SDG classification** — 9 SDG themes with dominance control:
  * SDG 3 – Health, SDG 4 – Education, SDG 6 – Water, SDG 7 – Energy
  * SDG 8 – Labor, SDG 9 – Innovation, SDG 12 – Consumption
  * SDG 13 – Climate, SDG 17 – Partnership

  Each chunk is assigned at most 2 SDGs using keyword scoring, word-boundary matching, a dominance rule, and confidence scores.

* **Visualizations** — word clouds, top keyword bar charts, distribution charts, overlap heatmaps, cosine similarity heatmaps, representative chunk cards with term highlighting

---

## Pipeline

```
PDF/Text Extraction → Cleanup → Chunking → spaCy Preprocessing → Rule-based Labels → TF-IDF & Similarity → Analysis
```

| Step | Details |
|---|---|
| **Data Preparation** | Clean PDF noise, remove headers/boilerplate, paragraph-based chunking, spaCy lemmatization, custom stopwords, POS filtering |
| **Labeling** | Strategic Framing and UN SDG labels via rule-based logic. SDG uses stronger thresholds and a dominance rule |
| **Analysis** | TF-IDF across Strategic Framing and UN SDGs. Cosine similarity for cross-group comparison |
| **Validation** | Audit layer checks duplicates, short chunks, repeated patterns, SDG label counts |

---

## Project Structure

```
.
├── app.py                # Streamlit dashboard (all pages)
├── main.py               # Pipeline runner
├── src/
│   ├── pipeline.py       # Core NLP pipeline, classification, TF-IDF
│   ├── preprocess.py     # Chunking and tokenization
│   └── analysis.py       # Diagnostic: keyword audit
├── outputs/
│   ├── chunks_processed.csv
│   ├── tfidf_by_section.csv
│   ├── tfidf_by_sdg.csv
│   ├── similarity_by_section.csv
│   └── similarity_by_sdg.csv
├── data/
│   └── raw/
└── requirements.txt
```

---

## How to Run

```bash
pip install -r requirements.txt
python main.py --audit
streamlit run app.py
```

---

## Demo Flow

Recommended presentation flow:

1. **Project Overview** — show the pitch: what we decode, why it matters, signature word cloud
2. **Analysis → Strategic Framing** — distribution, per-section word clouds, top 2 synthesis, validation heatmaps
3. **Analysis → UN SDG Themes** — same flow for SDG dimension
4. **Methods** — pipeline walkthrough (if audience asks "how?")

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

* Python, pandas
* spaCy (preprocessing)
* scikit-learn (TF-IDF, cosine similarity)
* Streamlit
* Plotly
* WordCloud

---

## Author

Text Mining Project – Team 9
