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

* **Visualizations**

  * Word clouds (TF-IDF weighted)
  * Top keyword bar charts
  * Cross-section heatmaps
  * Interactive chunk exploration

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
   ↓
TF-IDF Analysis
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

    * `company`, `report`, `sustainability`

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

---

## Dashboard

The Streamlit app includes:

### Overview Page

* Executive summary
* Section & orientation distributions
* Heatmaps of keyword overlap
* Quick insights

### Explorer Page

* Tab-based navigation
* Word cloud per category
* Top keywords (bar chart)
* Representative text chunks

---

## Project Structure

```
.
├── app.py
├── src/
│   └── pipeline.py
├── outputs/
│   ├── chunks_processed.csv
│   ├── tfidf_by_section.csv
│   └── tfidf_by_orientation.csv
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

## Key Insights (Example)

* Environmental sections emphasize:

  * carbon, waste, energy, recycling

* Governance focuses on:

  * board, risk management, compliance

* Action-oriented language:

  * reduce, improve, implement

* People-centric language:

  * employee development, community engagement

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
