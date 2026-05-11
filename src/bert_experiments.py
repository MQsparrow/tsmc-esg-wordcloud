from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


CHUNKS_CSV = Path("outputs/chunks_processed.csv")
TFIDF_SDG_SIMILARITY_CSV = Path("outputs/similarity_by_sdg.csv")
OUTPUTS_DIR = Path("outputs")
PIC_DIR = Path("pic")
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MIN_TEXT_CHARS = 40
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sentence-BERT experiments for the TSMC ESG text-mining project."
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=CHUNKS_CSV,
        help="Path to processed chunk CSV.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size for faster testing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute embeddings even if a cache exists.",
    )
    return parser.parse_args()


def slug_model_name(model_name: str) -> str:
    return model_name.split("/")[-1].replace(".", "_").replace("-", "-")


def parse_primary_label(value: object) -> str:
    if pd.isna(value):
        return "Unlabeled"
    labels = [label.strip() for label in str(value).split(",") if label.strip()]
    return labels[0] if labels else "Unlabeled"


def clean_display_label(label: object) -> str:
    if pd.isna(label) or not str(label).strip():
        return "Unlabeled"
    return str(label).strip()


def load_chunks(path: Path, sample_size: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing chunk file: {path}")

    df = pd.read_csv(path)
    required_columns = {"section_label", "sdg_labels"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    text_source = "clean_text" if "clean_text" in df.columns else "raw_text"
    if text_source not in df.columns:
        raise ValueError(f"{path} must include either clean_text or raw_text.")

    work = df.copy()
    work["text_for_bert"] = work[text_source].fillna("").astype(str).str.strip()
    if "raw_text" in work.columns:
        raw_text = work["raw_text"].fillna("").astype(str).str.strip()
        work.loc[work["text_for_bert"].str.len() < MIN_TEXT_CHARS, "text_for_bert"] = raw_text

    work = work[work["text_for_bert"].str.len() >= MIN_TEXT_CHARS].copy()
    work["primary_sdg"] = work["sdg_labels"].map(parse_primary_label)
    work["section_label"] = work["section_label"].map(clean_display_label)

    if "chunk_id" not in work.columns:
        work["chunk_id"] = np.arange(len(work))

    if sample_size and sample_size < len(work):
        work = (
            work.sample(n=sample_size, random_state=RANDOM_STATE)
            .sort_values("chunk_id")
            .reset_index(drop=True)
        )
    else:
        work = work.reset_index(drop=True)

    if work.empty:
        raise ValueError("No usable text chunks were found after filtering.")

    return work


def dataset_signature(df: pd.DataFrame, model_name: str) -> str:
    key_fields = df[["chunk_id", "text_for_bert", "primary_sdg", "section_label"]].astype(str)
    payload = key_fields.to_csv(index=False) + model_name
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_paths(model_name: str, sample_size: int | None) -> tuple[Path, Path, Path]:
    model_slug = slug_model_name(model_name)
    sample_suffix = f"_sample{sample_size}" if sample_size else ""
    embeddings_path = OUTPUTS_DIR / f"bert_embeddings_{model_slug}{sample_suffix}.npy"
    metadata_path = OUTPUTS_DIR / f"bert_embedding_metadata{sample_suffix}.csv"
    manifest_path = OUTPUTS_DIR / f"bert_embedding_manifest_{model_slug}{sample_suffix}.json"
    return embeddings_path, metadata_path, manifest_path


def load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sentence-transformers. "
            "Install it with: pip install -r requirements_bert.txt"
        ) from exc

    return SentenceTransformer(model_name)


def get_embeddings(
    df: pd.DataFrame,
    model_name: str,
    batch_size: int,
    force: bool,
    sample_size: int | None,
) -> np.ndarray:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path, metadata_path, manifest_path = cache_paths(model_name, sample_size)
    signature = dataset_signature(df, model_name)

    if not force and embeddings_path.exists() and metadata_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") == signature:
            print(f"Loading cached embeddings from {embeddings_path}")
            return np.load(embeddings_path)

    print(f"Loading model: {model_name}")
    model = load_sentence_transformer(model_name)
    print(f"Encoding {len(df):,} chunks...")
    embeddings = model.encode(
        df["text_for_bert"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(embeddings_path, embeddings)
    df[["chunk_id", "primary_sdg", "section_label", "text_for_bert"]].to_csv(
        metadata_path, index=False, encoding="utf-8-sig"
    )
    manifest_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "rows": len(df),
                "embedding_shape": list(embeddings.shape),
                "signature": signature,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved embeddings to {embeddings_path}")
    return embeddings


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def wrap_labels(labels: list[str], width: int = 16) -> list[str]:
    return ["\n".join(textwrap.wrap(str(label), width=width)) for label in labels]


def pca_coordinates(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(embeddings)
    return coords, pca.explained_variance_ratio_


def plot_embedding_scatter(
    coords: np.ndarray,
    labels: pd.Series,
    explained_variance: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    label_counts = labels.value_counts()
    ordered_labels = label_counts.index.tolist()
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(10, 7))
    for idx, label in enumerate(ordered_labels):
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=28,
            alpha=0.72,
            color=cmap(idx % 20),
            label=f"{label} ({int(mask.sum())})",
            edgecolors="none",
        )

    ax.set_title(title)
    ax.set_xlabel(f"PCA 1 ({explained_variance[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PCA 2 ({explained_variance[1] * 100:.1f}% variance)")
    ax.grid(True, linewidth=0.4, alpha=0.3)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def group_average_similarity(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    group_col: str,
    min_count: int = 2,
) -> pd.DataFrame:
    groups = []
    vectors = []
    counts = df[group_col].value_counts()

    for group in sorted(counts.index.tolist()):
        if group == "Unlabeled" or counts[group] < min_count:
            continue
        mask = df[group_col] == group
        groups.append(group)
        vectors.append(embeddings[mask].mean(axis=0))

    if not vectors:
        raise ValueError(f"No groups available for {group_col} similarity.")

    matrix = cosine_similarity(np.vstack(vectors))
    return pd.DataFrame(matrix, index=groups, columns=groups)


def plot_heatmap(matrix: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    im = ax.imshow(matrix.values, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(wrap_labels(matrix.columns.tolist()), rotation=45, ha="right")
    ax.set_yticklabels(wrap_labels(matrix.index.tolist()))

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iat[row, col]
            text_color = "white" if value >= 0.72 else "#1f2933"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Cosine similarity")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def load_tfidf_sdg_similarity(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Skipping TF-IDF comparison because {path} was not found.")
        return None

    matrix = pd.read_csv(path, index_col=0)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    return matrix


def plot_tfidf_vs_bert(
    tfidf_matrix: pd.DataFrame,
    bert_matrix: pd.DataFrame,
    output_path: Path,
) -> None:
    common = [label for label in tfidf_matrix.index if label in bert_matrix.index]
    if len(common) < 2:
        print("Skipping TF-IDF vs BERT comparison because fewer than two SDGs overlap.")
        return

    tfidf = tfidf_matrix.loc[common, common]
    bert = bert_matrix.loc[common, common]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), sharey=True)
    for ax, matrix, title in [
        (axes[0], tfidf, "Existing TF-IDF Similarity by SDG"),
        (axes[1], bert, "New BERT Semantic Similarity by SDG"),
    ]:
        im = ax.imshow(matrix.values, cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(common)))
        ax.set_yticks(np.arange(len(common)))
        ax.set_xticklabels(wrap_labels(common), rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(wrap_labels(common), fontsize=8)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix.iat[row, col]
                text_color = "white" if value >= 0.72 else "#1f2933"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=7)

    colorbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03)
    colorbar.set_label("Cosine similarity")
    fig.suptitle("TF-IDF vs BERT: Keyword Similarity and Semantic Similarity", y=1.02, fontsize=15)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def save_similarity_csv(matrix: pd.DataFrame, path: Path) -> None:
    matrix.to_csv(path, encoding="utf-8-sig")
    print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    set_plot_style()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PIC_DIR.mkdir(parents=True, exist_ok=True)

    df = load_chunks(args.chunks, args.sample_size)
    embeddings = get_embeddings(
        df=df,
        model_name=args.model,
        batch_size=args.batch_size,
        force=args.force,
        sample_size=args.sample_size,
    )

    coords, explained_variance = pca_coordinates(embeddings)

    plot_embedding_scatter(
        coords=coords,
        labels=df["primary_sdg"],
        explained_variance=explained_variance,
        title="Sentence-BERT Chunk Embeddings by Primary SDG",
        output_path=PIC_DIR / "bert_sdg_embedding_pca.png",
    )
    plot_embedding_scatter(
        coords=coords,
        labels=df["section_label"],
        explained_variance=explained_variance,
        title="Sentence-BERT Chunk Embeddings by Strategic Framing Section",
        output_path=PIC_DIR / "bert_section_embedding_pca.png",
    )

    sdg_similarity = group_average_similarity(df, embeddings, "primary_sdg")
    section_similarity = group_average_similarity(df, embeddings, "section_label")
    save_similarity_csv(sdg_similarity, OUTPUTS_DIR / "bert_similarity_by_sdg.csv")
    save_similarity_csv(section_similarity, OUTPUTS_DIR / "bert_similarity_by_section.csv")

    plot_heatmap(
        sdg_similarity,
        "Sentence-BERT Semantic Similarity by SDG",
        PIC_DIR / "bert_sdg_similarity_heatmap.png",
    )
    plot_heatmap(
        section_similarity,
        "Sentence-BERT Semantic Similarity by Strategic Framing Section",
        PIC_DIR / "bert_section_similarity_heatmap.png",
    )

    tfidf_sdg_similarity = load_tfidf_sdg_similarity(TFIDF_SDG_SIMILARITY_CSV)
    if tfidf_sdg_similarity is not None:
        plot_tfidf_vs_bert(
            tfidf_sdg_similarity,
            sdg_similarity,
            PIC_DIR / "tfidf_vs_bert_sdg_similarity.png",
        )

    print("BERT experiment complete.")


if __name__ == "__main__":
    main()
