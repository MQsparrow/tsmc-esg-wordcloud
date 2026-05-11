# BERT / FinBERT Experiment Plan

## Goal

This experiment supports the final project assignment requirement:

> Ask AI which BERT-family member may be helpful to the final project, apply it to the project data, and share five experiment pictures.

Our final project analyzes TSMC sustainability reports and asks:

- Which UN SDGs receive the most attention?
- How does TSMC frame ESG issues across report sections and years?
- Does ESG language describe measurable action or mostly aspirational commitments?
- How does semantic evidence compare with the mid-term TF-IDF baseline?

## Recommended BERT Family Member

Use:

`ProsusAI/finbert`

FinBERT is a BERT-family model adapted to financial and corporate language. It fits this project because ESG reports are corporate disclosure documents. They contain language about targets, achievements, risks, compliance, suppliers, governance, and performance indicators.

We also use:

`sentence-transformers/all-MiniLM-L6-v2`

MiniLM is used for semantic maps and similarity heatmaps because it is lightweight and strong for sentence / paragraph embeddings.

## Why Not Only TF-IDF?

TF-IDF is useful for keyword patterns, but it has limits:

- It does not understand context.
- It treats similar meanings with different words as unrelated.
- It may score future promises and completed actions similarly if they share keywords.
- It cannot directly measure whether a claim is concrete or vague.

BERT-family embeddings help us add semantic evidence. FinBERT is especially useful for corporate-report language, while MiniLM is useful for dashboard-friendly semantic visualization.

## Labeling Strategy

True F1 / accuracy needs ground truth labels. The current repo does not yet contain fully verified human labels for all chunks, so this week's evaluation uses **AI-assisted weak labels**.

Weak labels are generated as follows:

- SDG labels come from the existing rule-based `sdg_labels` column.
- Concreteness labels come from transparent heuristics using numbers, dates, units, completed-action verbs, and vague future-oriented language.
- The generated label file is saved as `outputs/weak_eval_labels.csv`.

Report wording:

> Because manually verified labels are still in progress, this week's experiment uses weak labels generated from our rule-based SDG pipeline and concreteness heuristics. These scores are pilot evaluation results, not final gold-standard performance.

This is defensible because the labels and scores are reproducible and inspectable.

## Model Comparison Design

Script:

`src/finbert_weak_label_evaluation.py`

The script compares three models:

1. **TF-IDF baseline**
   - TF-IDF vectorizer
   - logistic regression
   - represents the mid-term keyword-based approach

2. **FinBERT tuned classifier**
   - FinBERT embeddings
   - logistic regression
   - per-SDG threshold tuning using validation data

3. **Hybrid TF-IDF + FinBERT**
   - combines keyword features and semantic embeddings
   - per-SDG threshold tuning
   - intended as the strongest practical pilot model

Tasks:

- Multi-label SDG classification for 9 SDGs
- Binary concreteness detection: concrete vs vague

## Why Threshold Tuning?

The SDG task is multi-label and imbalanced. Some SDGs, such as SDG 8 and SDG 9, have very few examples. A default probability threshold of 0.5 is often too rigid.

The updated script tunes a threshold for each SDG on a validation split, then evaluates on the held-out test split. This is more appropriate than using the same cutoff for every SDG.

## Integration With Zoe's Multi-Year Work

Zoe's branch adds:

- multi-year outputs for 2022, 2023, and 2024
- `src/embeddings.py`
- semantic embedding inspector in the dashboard
- updated app structure and analysis pages

Our BERT / FinBERT work complements Zoe's work:

- Zoe's contribution strengthens the **dashboard and multi-year semantic exploration**.
- This experiment strengthens the **Part 1 BERT evaluation and accountability analysis**.

Recommended integration story:

> The final dashboard shows multi-year ESG / SDG patterns and semantic embedding exploration. The Part 1 FinBERT experiment validates whether BERT-family models can improve SDG classification and detect concreteness in ESG claims.

## Commands

Install dependencies:

```powershell
pip install -r requirements_bert.txt
```

Run MiniLM semantic experiment:

```powershell
python src/bert_experiments.py
```

Run FinBERT evaluation:

```powershell
python src/finbert_weak_label_evaluation.py
```

Run multi-year FinBERT evaluation:

```powershell
python src/finbert_weak_label_evaluation.py --multi-year
```

Regenerate weak labels:

```powershell
python src/finbert_weak_label_evaluation.py --force-labels
```

Recompute FinBERT embeddings:

```powershell
python src/finbert_weak_label_evaluation.py --force-embeddings
```

Only generate labels:

```powershell
python src/finbert_weak_label_evaluation.py --labels-only --force-labels
```

Only generate multi-year labels:

```powershell
python src/finbert_weak_label_evaluation.py --multi-year --labels-only --force-labels
```

## Expected Outputs

Evaluation tables:

- `outputs/weak_eval_labels.csv`
- `outputs/finbert_vs_tfidf_sdg_f1_scores.csv`
- `outputs/concreteness_detection_metrics.csv`
- `outputs/coverage_concreteness_by_sdg.csv`
- `outputs/sdg_tuned_thresholds.csv`
- `outputs/bert_experiment_model_summary.csv`
- `outputs/weak_eval_manifest.json`

Multi-year evaluation uses the same names with `_multiyear`, for example:

- `outputs/weak_eval_labels_multiyear.csv`
- `outputs/finbert_vs_tfidf_sdg_f1_scores_multiyear.csv`
- `outputs/bert_experiment_model_summary_multiyear.csv`
- `pic/finbert_vs_tfidf_sdg_f1_multiyear.png`

Five selected pictures:

- `pic/finbert_vs_tfidf_sdg_f1.png`
- `pic/concreteness_detection_accuracy.png`
- `pic/coverage_vs_concreteness_gap.png`
- `pic/tfidf_vs_bert_sdg_similarity.png`
- `pic/bert_sdg_embedding_pca.png`

Backup semantic pictures:

- `pic/bert_section_embedding_pca.png`
- `pic/bert_sdg_similarity_heatmap.png`
- `pic/bert_section_similarity_heatmap.png`

## GPU Requirement

GPU is not required.

This experiment uses pre-trained models for embedding inference plus lightweight logistic regression classifiers. The dataset is small enough to run on CPU. A GPU would only make embedding generation faster.

Report wording:

> We did not require GPU training because no large-scale fine-tuning was performed. We used pre-trained BERT-family embeddings and lightweight classifiers.

## What Counts as Done

For this assignment, the BERT part is done when:

- FinBERT is selected and justified.
- TF-IDF baseline and FinBERT/Hybrid results are generated.
- SDG F1 scores are exported.
- Concreteness accuracy is exported.
- Five pictures are selected and interpreted.
- The report clearly states that current scores are weak-label pilot results.

## Next Step for Full Marks

The strongest next step is a small manual validation set:

- sample 50 to 100 chunks
- manually correct SDG labels
- manually mark concrete vs vague
- rerun the same evaluation against this validated subset

This would turn the current pilot into a stronger final-project validation.
