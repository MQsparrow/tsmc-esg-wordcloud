# TSMC ESG Word Cloud Dashboard

An interactive text mining project analyzing sustainability communication in ESG reports, focusing on how language differs across domains and narrative styles.

---

## Overview

This project explores how ESG (Environmental, Social, Governance) themes are communicated in corporate sustainability reports.

Using text mining techniques, we transform unstructured report text into an interactive dashboard that reveals:

* Key themes across ESG sections
* Differences between **action-oriented** and **people-centric** language
* Term importance using **TF-IDF**
* Cross-domain keyword relationships
* Universal language using **UN SDGs goal** to map and dig out problems int company

The final result is an interactive Streamlit dashboard for exploration and presentation.

---

## Key Features

* **Section-based analysis**

  * Environment
  * Talent
  * Supply Chain
  * Social
  * Governance

* **Narrative orientation analysis**

  * Action-Oriented language (e.g., reduce, improve, implement)
  * People-Centric language (e.g., employee, community, safety)

* **SDG multi-label classification**

  * SDG 3 – Good Health and Well-being
  * SDG 4 – Quality Education
  * SDG 6 – Clean Water and Sanitation
  * SDG 7 – Affordable and Clean Energy
  * SDG 8 – Decent Work and Economic Growth
  * SDG 9 – Industry, Innovation and Infrastructure
  * SDG 12 – Responsible Consumption and Production
  * SDG 13 – Climate Action
  * SDG 17 – Partnerships for the Goals

  Each chunk can be tagged with multiple SDGs (threshold ≥ 2 keyword hits). Word-boundary matching prevents substring false positives. Protected phrases (e.g., `trade secret`) are preserved through tokenization.

* **Visualizations**

  * Word clouds (TF-IDF weighted)
  * Top keyword bar charts
  * Cross-section heatmaps
  * Interactive chunk exploration
  * SDG distribution chart
  * Expandable SDG insight cards with editorial narrative

---

## Pipeline

The project follows a structured NLP pipeline:

```
PDF Reports
   ↓
Text Extraction
   ↓
Preprocessing
   - cleanup (remove headers, noise)
   - paragraph-based chunking
   - normalization
   - lemmatization
   - custom stopwords
   - POS filtering
   ↓
Classification
   - ESG section labeling (rule-based)
   - orientation (action vs people)
   - SDG multi-label (keyword hit threshold)
   ↓
TF-IDF Analysis
   - per section
   - per orientation
   - per SDG (exploded multi-label)
   ↓
Visualization (Streamlit)
```

---

## Preprocessing Details

To ensure high-quality results while maintaining interpretability:

* **Section-aware chunking**

  * Based on paragraphs instead of fixed length

* **Lemmatization (spaCy)**

  * More stable than stemming

* **Custom stopwords**

  * Removes domain noise such as:

    * `company`, `report`, `sustainability`, `wet`, `scrubber`, `ton`, `nature`

* **Protected phrases**

  * Multi-word terms joined with underscore before tokenization to survive stopword filtering (e.g., `trade_secret`)

* **POS filtering**

  * Keeps meaningful tokens:

    * Nouns, proper nouns, adjectives

* **N-grams**

  * Captures phrases like:

    * `carbon emission`, `supply chain`

---

## TF-IDF Strategy

TF-IDF is used instead of raw frequency to:

* Highlight **important but not overly common terms**
* Reduce dominance of generic words
* Improve interpretability of ESG themes

We compute TF-IDF:

* Per section (ESG domains)
* Per orientation (action vs people)
* Per SDG (UN SDG goals — using exploded multi-label dataframe, each chunk can contribute to multiple SDG groups)

---

## Dashboard

The Streamlit app includes:

### Overview Page

* Executive summary with section, orientation, and SDG highlights
* Metrics: total chunks, sections, orientations, SDG themes
* Quick Insights cards: Environment, Talent, Action-Oriented focus
* **SDG Focus**: top 2 SDGs by chunk count, with expandable editorial narrative explaining what the data really means
* Section & orientation distributions
* SDG distribution chart
* Heatmaps of keyword overlap

### Explorer Page

* Tab-based navigation: Section view, Orientation view, SDG view
* Word cloud per category
* Top keywords (bar chart)
* Representative text chunks with multi-label SDG tags

---

## Project Structure

```
.
├── app.py
├── main.py
├── src/
│   ├── pipeline.py       # core NLP pipeline, SDG classification, TF-IDF
│   ├── preprocess.py     # basic chunking and tokenization
│   └── analysis.py       # diagnostic: keyword audit, sentence co-occurrence
├── outputs/
│   ├── chunks_processed.csv
│   ├── tfidf_by_section.csv
│   ├── tfidf_by_orientation.csv
│   └── tfidf_by_sdg.csv
├── data/
│   └── raw/
└── requirements.txt
```

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deployment

This app is deployed using **Streamlit Community Cloud**.

Any update pushed to GitHub will automatically trigger a redeploy.

---

## Key Insights

* Environmental sections emphasize: carbon, waste, energy, recycling
* Governance focuses on: board, risk management, compliance
* Action-oriented language: reduce, improve, implement
* People-centric language: employee development, community engagement
* **SDG 17 (Partnership)** has the highest chunk count — driven by compliance infrastructure (committee, carbon reduction, chemical), not goodwill language
* **SDG 9 (Innovation)** is IP-led: patent and trade secret dominate over generic R&D terms
* **Chemical** appears across SDG 3, 6, 12 because chemical management is cross-cutting in semiconductor fab operations

---

## Tech Stack

* Python
* pandas
* spaCy (preprocessing)
* scikit-learn (TF-IDF)
* Streamlit
* Plotly
* WordCloud

---

## Notes

* This project prioritizes **interpretability over complexity**
* Rule-based classification is used for transparency
* No heavy deep learning models are required

---

## Future Improvements

* Multi-year ESG comparison
* Topic modeling / clustering
* LLM-based insight generation
* More advanced UI/UX enhancements

---

## Author

Text Mining Project – Team 9
