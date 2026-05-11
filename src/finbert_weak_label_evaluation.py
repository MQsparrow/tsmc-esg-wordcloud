from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline


CHUNKS_CSV = Path("outputs/chunks_processed.csv")
OUTPUTS_DIR = Path("outputs")
PIC_DIR = Path("pic")
WEAK_LABELS_CSV = OUTPUTS_DIR / "weak_eval_labels.csv"
DEFAULT_FINBERT_MODEL = "ProsusAI/finbert"
MIN_TEXT_CHARS = 40
RANDOM_STATE = 42

SDG_LABELS = [
    "SDG3_health",
    "SDG4_education",
    "SDG6_water",
    "SDG7_energy",
    "SDG8_labor",
    "SDG9_innovation",
    "SDG12_consumption",
    "SDG13_climate",
    "SDG17_partnership",
]

SDG_DISPLAY = {
    "SDG3_health": "SDG 3\nHealth",
    "SDG4_education": "SDG 4\nEducation",
    "SDG6_water": "SDG 6\nWater",
    "SDG7_energy": "SDG 7\nEnergy",
    "SDG8_labor": "SDG 8\nLabor",
    "SDG9_innovation": "SDG 9\nInnovation",
    "SDG12_consumption": "SDG 12\nConsumption",
    "SDG13_climate": "SDG 13\nClimate",
    "SDG17_partnership": "SDG 17\nPartnership",
}

SPECIFIC_NUMBER_PATTERN = re.compile(
    r"(\b\d{4}\b|\b\d+(?:\.\d+)?\s?(?:%|percent|tonnes?|tons?|metric|nt\$|million|"
    r"billion|hours?|employees?|suppliers?|cases?|mw|mwh|kwh|gwh|liters?|m3|"
    r"kg|kilograms?|tco2e|co2e|ppm|pages?|years?)\b)",
    flags=re.IGNORECASE,
)
DATE_PATTERN = re.compile(
    r"\b(20\d{2}|by\s+20\d{2}|from\s+20\d{2}\s+to\s+20\d{2}|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)\b",
    flags=re.IGNORECASE,
)
CONCRETE_VERBS = {
    "achieved",
    "completed",
    "reduced",
    "increased",
    "implemented",
    "launched",
    "established",
    "certified",
    "verified",
    "recorded",
    "reached",
    "invested",
    "installed",
    "conducted",
    "audited",
}
VAGUE_TERMS = {
    "aim",
    "aims",
    "aspire",
    "committed",
    "commitment",
    "continue",
    "expect",
    "future",
    "hope",
    "intend",
    "plan",
    "promote",
    "strive",
    "targeting",
    "vision",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create weak labels and compare TF-IDF baseline with a FinBERT/BERT embedding "
            "classifier for SDG F1 and concreteness accuracy."
        )
    )
    parser.add_argument("--chunks", type=Path, default=CHUNKS_CSV)
    parser.add_argument("--label-file", type=Path, default=WEAK_LABELS_CSV)
    parser.add_argument("--finbert-model", default=DEFAULT_FINBERT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="Optional .npy cache. If omitted, uses outputs/finbert_embeddings.npy.",
    )
    parser.add_argument(
        "--force-labels",
        action="store_true",
        help="Regenerate weak labels even if the label CSV already exists.",
    )
    parser.add_argument(
        "--force-embeddings",
        action="store_true",
        help="Recompute FinBERT embeddings even if a cache exists.",
    )
    parser.add_argument(
        "--labels-only",
        action="store_true",
        help="Only generate/load weak labels, then stop before model evaluation.",
    )
    return parser.parse_args()


def parse_sdg_labels(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [label.strip() for label in str(value).split(",") if label.strip() in SDG_LABELS]


def text_for_model(row: pd.Series) -> str:
    clean = str(row.get("clean_text", "") if not pd.isna(row.get("clean_text", "")) else "").strip()
    raw = str(row.get("raw_text", "") if not pd.isna(row.get("raw_text", "")) else "").strip()
    return clean if len(clean) >= MIN_TEXT_CHARS else raw


def concreteness_score(text: str) -> float:
    lower = text.lower()
    words = re.findall(r"[a-zA-Z]+", lower)
    number_hits = len(SPECIFIC_NUMBER_PATTERN.findall(text))
    date_hits = len(DATE_PATTERN.findall(text))
    concrete_hits = sum(1 for word in words if word in CONCRETE_VERBS)
    vague_hits = sum(1 for word in words if word in VAGUE_TERMS)

    percent_hits = text.count("%")

    score = 0.0
    score += min(number_hits, 4) * 0.10
    score += min(date_hits, 3) * 0.06
    score += min(concrete_hits, 4) * 0.10
    score += min(percent_hits, 3) * 0.05
    score -= min(vague_hits, 5) * 0.08
    if number_hits and concrete_hits:
        score += 0.08
    return float(np.clip(score, 0.0, 1.0))


def generate_weak_labels(chunks_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(chunks_path)
    records = []

    for _, row in df.iterrows():
        text = text_for_model(row)
        if len(text) < MIN_TEXT_CHARS:
            continue
        raw_text = str(row.get("raw_text", "") if not pd.isna(row.get("raw_text", "")) else "").strip()
        concreteness_text = raw_text if raw_text else text
        labels = parse_sdg_labels(row.get("sdg_labels", ""))
        score = concreteness_score(concreteness_text)
        record = {
            "chunk_id": row.get("chunk_id"),
            "text": text,
            "weak_primary_sdg": labels[0] if labels else "unclassified",
            "weak_concreteness_score": round(score, 4),
            "weak_concrete": int(score >= 0.65),
            "label_source": "AI-assisted weak labels from rule-based SDG labels + concreteness heuristics",
        }
        for sdg in SDG_LABELS:
            record[sdg] = int(sdg in labels)
        records.append(record)

    labels_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved weak labels to {output_path}")
    return labels_df


def load_or_generate_labels(args: argparse.Namespace) -> pd.DataFrame:
    if args.label_file.exists() and not args.force_labels:
        print(f"Loading weak labels from {args.label_file}")
        return pd.read_csv(args.label_file)
    return generate_weak_labels(args.chunks, args.label_file)


def valid_label_columns(labels_df: pd.DataFrame) -> list[str]:
    valid = []
    for sdg in SDG_LABELS:
        positives = int(labels_df[sdg].sum())
        negatives = int((1 - labels_df[sdg]).sum())
        if positives >= 2 and negatives >= 2:
            valid.append(sdg)
    return valid


def split_data(labels_df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = labels_df.reset_index(drop=True).copy()
    work["_row_id"] = np.arange(len(work))
    concrete = work["weak_concrete"]
    stratify = concrete if concrete.nunique() == 2 and concrete.value_counts().min() >= 2 else None
    train_df = test_df = None
    for seed in range(RANDOM_STATE, RANDOM_STATE + 200):
        candidate_train, candidate_test = train_test_split(
            work,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
        train_ok = all(candidate_train[sdg].sum() > 0 for sdg in SDG_LABELS if work[sdg].sum() > 0)
        test_ok = all(candidate_test[sdg].sum() > 0 for sdg in SDG_LABELS if work[sdg].sum() > 0)
        if train_ok and test_ok:
            train_df, test_df = candidate_train, candidate_test
            break

    if train_df is None or test_df is None:
        train_df, test_df = train_test_split(
            work,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def evaluate_multilabel_predictions(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    model_name: str,
    labels: list[str],
) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true[labels].values,
        y_pred,
        average=None,
        zero_division=0,
    )
    rows = []
    for idx, sdg in enumerate(labels):
        rows.append(
            {
                "model": model_name,
                "sdg": sdg,
                "precision": round(float(precision[idx]), 4),
                "recall": round(float(recall[idx]), 4),
                "f1": round(float(f1[idx]), 4),
                "support": int(support[idx]),
            }
        )
    rows.append(
        {
            "model": model_name,
            "sdg": "MACRO_AVERAGE",
            "precision": np.nan,
            "recall": np.nan,
            "f1": round(float(f1_score(y_true[labels].values, y_pred, average="macro", zero_division=0)), 4),
            "support": int(y_true[labels].values.sum()),
        }
    )
    return pd.DataFrame(rows)


def fit_tfidf_sdg_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels: list[str],
) -> tuple[np.ndarray, object]:
    classifier = make_pipeline(
        TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.90,
            max_features=6000,
        ),
        OneVsRestClassifier(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
    )
    classifier.fit(train_df["text"], train_df[labels])
    return classifier.predict(test_df["text"]), classifier


def mean_pool_embeddings(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def compute_transformer_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int,
    cache_path: Path,
    force: bool,
) -> np.ndarray:
    if cache_path.exists() and not force:
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers/torch. Install with: pip install -r requirements_bert.txt"
        ) from exc

    print(f"Loading transformer model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            output = model(**encoded)
            pooled = mean_pool_embeddings(output.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy())
            print(f"Embedded {min(start + batch_size, len(texts))}/{len(texts)} chunks")

    embeddings = np.vstack(vectors)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"Saved embeddings to {cache_path}")
    return embeddings


def fit_embedding_sdg_classifier(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_df: pd.DataFrame,
    labels: list[str],
) -> np.ndarray:
    predictions = []
    for sdg in labels:
        y_train = train_df[sdg].values
        if len(np.unique(y_train)) < 2:
            clf = DummyClassifier(strategy="most_frequent")
        else:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        clf.fit(train_embeddings, y_train)
        predictions.append(clf.predict(test_embeddings))
    return np.vstack(predictions).T


def evaluate_concreteness_tfidf(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    if train_df["weak_concrete"].nunique() < 2:
        clf = make_pipeline(TfidfVectorizer(stop_words="english"), DummyClassifier(strategy="most_frequent"))
    else:
        clf = make_pipeline(
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.90,
                max_features=6000,
            ),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        )
    clf.fit(train_df["text"], train_df["weak_concrete"])
    pred = clf.predict(test_df["text"])
    return concreteness_metrics(test_df["weak_concrete"].values, pred, "TF-IDF baseline")


def evaluate_concreteness_embeddings(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
) -> dict[str, float]:
    if train_df["weak_concrete"].nunique() < 2:
        clf = DummyClassifier(strategy="most_frequent")
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(train_embeddings, train_df["weak_concrete"])
    pred = clf.predict(test_embeddings)
    return concreteness_metrics(test_df["weak_concrete"].values, pred, model_name)


def concreteness_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "positive_support": int(np.sum(y_true)),
        "total_test_chunks": int(len(y_true)),
    }


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_sdg_f1(scores: pd.DataFrame, output_path: Path) -> None:
    plot_df = scores[scores["sdg"] != "MACRO_AVERAGE"].copy()
    pivot = plot_df.pivot(index="sdg", columns="model", values="f1").reindex(SDG_LABELS)

    x = np.arange(len(pivot.index))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6.5))
    models = pivot.columns.tolist()
    for idx, model in enumerate(models):
        offset = (idx - (len(models) - 1) / 2) * width
        ax.bar(x + offset, pivot[model].fillna(0), width=width, label=model)

    ax.set_title("SDG Classification F1 by Category")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([SDG_DISPLAY.get(sdg, sdg) for sdg in pivot.index])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_concreteness_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    metric_cols = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metric_cols))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.8))

    for idx, (_, row) in enumerate(metrics.iterrows()):
        offset = (idx - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, [row[col] for col in metric_cols], width=width, label=row["model"])

    ax.set_title("Concreteness Detection Performance")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_coverage_concreteness(labels_df: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for sdg in SDG_LABELS:
        subset = labels_df[labels_df[sdg] == 1]
        if subset.empty:
            concrete_share = 0.0
        else:
            concrete_share = float(subset["weak_concrete"].mean())
        coverage = int(len(subset))
        risk = coverage * (1.0 - concrete_share)
        rows.append(
            {
                "sdg": sdg,
                "coverage": coverage,
                "concrete_share": concrete_share,
                "risk": risk,
            }
        )
    plot_df = pd.DataFrame(rows).sort_values("coverage", ascending=False)

    fig, ax1 = plt.subplots(figsize=(12, 6.2))
    x = np.arange(len(plot_df))
    ax1.bar(x, plot_df["coverage"], color="#4C78A8", alpha=0.82, label="Chunk coverage")
    ax1.set_ylabel("Number of chunks")
    ax1.set_xticks(x)
    ax1.set_xticklabels([SDG_DISPLAY.get(sdg, sdg) for sdg in plot_df["sdg"]])
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, plot_df["concrete_share"] * 100, color="#F58518", marker="o", linewidth=2.2)
    ax2.set_ylabel("Concrete chunks (%)")
    ax2.set_ylim(0, 105)

    fig.suptitle("Coverage vs Concreteness Gap by SDG", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")

    plot_df.to_csv(OUTPUTS_DIR / "coverage_concreteness_by_sdg.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    set_plot_style()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PIC_DIR.mkdir(parents=True, exist_ok=True)

    labels_df = load_or_generate_labels(args)
    if args.labels_only:
        print(f"Labels ready: {args.label_file} ({len(labels_df)} rows)")
        return

    valid_labels = valid_label_columns(labels_df)
    if not valid_labels:
        raise ValueError("No SDG labels have enough positive and negative examples for evaluation.")

    train_df, test_df = split_data(labels_df, args.test_size)
    tfidf_pred, _ = fit_tfidf_sdg_classifier(train_df, test_df, valid_labels)
    tfidf_scores = evaluate_multilabel_predictions(
        test_df, tfidf_pred, "TF-IDF baseline", valid_labels
    )

    cache_path = args.embedding_cache or (OUTPUTS_DIR / "finbert_embeddings.npy")
    embeddings = compute_transformer_embeddings(
        labels_df["text"].tolist(),
        model_name=args.finbert_model,
        batch_size=args.batch_size,
        cache_path=cache_path,
        force=args.force_embeddings,
    )
    train_embeddings = embeddings[train_df["_row_id"].values]
    test_embeddings = embeddings[test_df["_row_id"].values]
    finbert_pred = fit_embedding_sdg_classifier(
        train_embeddings, test_embeddings, train_df, valid_labels
    )
    finbert_scores = evaluate_multilabel_predictions(
        test_df, finbert_pred, "FinBERT embedding classifier", valid_labels
    )

    sdg_scores = pd.concat([tfidf_scores, finbert_scores], ignore_index=True)
    sdg_scores_path = OUTPUTS_DIR / "finbert_vs_tfidf_sdg_f1_scores.csv"
    sdg_scores.to_csv(sdg_scores_path, index=False, encoding="utf-8-sig")
    print(f"Saved {sdg_scores_path}")

    concreteness_rows = [
        evaluate_concreteness_tfidf(train_df, test_df),
        evaluate_concreteness_embeddings(
            train_embeddings,
            test_embeddings,
            train_df,
            test_df,
            "FinBERT embedding classifier",
        ),
    ]
    concreteness_metrics_df = pd.DataFrame(concreteness_rows)
    concreteness_path = OUTPUTS_DIR / "concreteness_detection_metrics.csv"
    concreteness_metrics_df.to_csv(concreteness_path, index=False, encoding="utf-8-sig")
    print(f"Saved {concreteness_path}")

    split_manifest = {
        "label_source": "weak labels, not human gold labels",
        "train_chunks": int(len(train_df)),
        "test_chunks": int(len(test_df)),
        "test_size": args.test_size,
        "random_state": RANDOM_STATE,
        "sdg_labels_evaluated": valid_labels,
        "finbert_model": args.finbert_model,
    }
    (OUTPUTS_DIR / "weak_eval_manifest.json").write_text(
        json.dumps(split_manifest, indent=2), encoding="utf-8"
    )

    plot_sdg_f1(sdg_scores, PIC_DIR / "finbert_vs_tfidf_sdg_f1.png")
    plot_concreteness_metrics(
        concreteness_metrics_df, PIC_DIR / "concreteness_detection_accuracy.png"
    )
    plot_coverage_concreteness(labels_df, PIC_DIR / "coverage_vs_concreteness_gap.png")

    print("Weak-label FinBERT evaluation complete.")


if __name__ == "__main__":
    main()
