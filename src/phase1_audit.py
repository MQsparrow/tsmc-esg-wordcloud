from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

CHUNKS_CSV = Path("outputs/chunks_processed.csv")
SECTION_TFIDF_CSV = Path("outputs/tfidf_by_section.csv")
ORIENTATION_TFIDF_CSV = Path("outputs/tfidf_by_orientation.csv")
SDG_TFIDF_CSV = Path("outputs/tfidf_by_sdg.csv")
SHORT_TOKEN_THRESHOLD = 20
SUSPICIOUS_LABEL_THRESHOLD = 4
MAX_EXAMPLES = 5


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def first_line(text: str) -> str:
    lines = [line.strip() for line in re.split(r"\r?\n", text) if line.strip()]
    return lines[0] if lines else ""


def token_count(text: str) -> int:
    return len(normalize_text(text).split())


def parse_labels(label_text: object) -> list[str]:
    text = normalize_text(label_text)
    if not text:
        return []
    return [label.strip() for label in text.split(",") if label.strip()]


def duplicate_chunk_summary(df: pd.DataFrame) -> tuple[int, int]:
    raw_dupes = int(df.duplicated(subset=["raw_text"]).sum())
    clean_dupes = int(df.duplicated(subset=["clean_text"]).sum())
    return raw_dupes, clean_dupes


def label_completeness_summary(df: pd.DataFrame) -> dict[str, int]:
    return {
        "missing_section_label": int(df["section_label"].fillna("").str.strip().eq("").sum()),
        "missing_orientation": int(df["orientation"].fillna("").str.strip().eq("").sum()),
        "missing_sdg_labels": int(df["sdg_labels"].fillna("").str.strip().eq("").sum()),
    }


def repeated_first_line_patterns(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    first_lines = df["raw_text"].fillna("").map(first_line)
    counts = first_lines.value_counts().reset_index()
    counts.columns = ["first_line", "count"]
    repeated = counts[counts["count"] > 1].head(top_k).copy()
    return repeated


def sdg_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    label_counts = df["sdg_labels"].map(parse_labels).map(len)
    distribution = label_counts.value_counts().sort_index().reset_index()
    distribution.columns = ["sdg_label_count", "chunk_count"]
    distribution["chunk_pct"] = (distribution["chunk_count"] / len(df) * 100).round(1)
    return distribution


def find_short_chunks(df: pd.DataFrame, threshold: int = SHORT_TOKEN_THRESHOLD) -> pd.DataFrame:
    audit = df.copy()
    audit["clean_token_count"] = audit["clean_text"].map(token_count)
    return audit[audit["clean_token_count"] < threshold].copy()


def find_suspicious_chunks(
    df: pd.DataFrame,
    label_threshold: int = SUSPICIOUS_LABEL_THRESHOLD,
) -> pd.DataFrame:
    audit = df.copy()
    audit["sdg_label_count"] = audit["sdg_labels"].map(parse_labels).map(len)
    return audit[audit["sdg_label_count"] >= label_threshold].copy()


def tfidf_output_summary() -> pd.DataFrame:
    specs = [
        ("section", SECTION_TFIDF_CSV, "section_label"),
        ("orientation", ORIENTATION_TFIDF_CSV, "orientation"),
        ("sdg", SDG_TFIDF_CSV, "sdg_labels"),
    ]

    records = []
    for name, path, group_col in specs:
        if not path.exists():
            records.append(
                {
                    "output": name,
                    "exists": False,
                    "rows": 0,
                    "groups": 0,
                    "nonpositive_scores": 0,
                    "missing_terms": 0,
                }
            )
            continue

        df = pd.read_csv(path)
        missing_terms = int(df["term"].fillna("").str.strip().eq("").sum()) if "term" in df.columns else len(df)
        nonpositive_scores = int((df.get("tfidf_score", pd.Series(dtype=float)).fillna(0) <= 0).sum())
        groups = int(df[group_col].nunique()) if group_col in df.columns else 0
        records.append(
            {
                "output": name,
                "exists": True,
                "rows": len(df),
                "groups": groups,
                "nonpositive_scores": nonpositive_scores,
                "missing_terms": missing_terms,
            }
        )

    return pd.DataFrame(records)


def collect_problem_examples(df: pd.DataFrame) -> pd.DataFrame:
    examples: list[dict[str, object]] = []
    seen_chunk_ids: set[object] = set()

    duplicate_mask = df.duplicated(subset=["clean_text"], keep=False)
    duplicate_examples = df.loc[duplicate_mask].copy()
    duplicate_examples["issue_type"] = "duplicate_clean_text"
    duplicate_examples["issue_detail"] = "Duplicate clean_text value"

    repeated_lines = set(repeated_first_line_patterns(df, top_k=20)["first_line"])
    repeated_examples = df[df["raw_text"].fillna("").map(first_line).isin(repeated_lines)].copy()
    repeated_examples["issue_type"] = "repeated_first_line"
    repeated_examples["issue_detail"] = repeated_examples["raw_text"].fillna("").map(first_line)

    short_examples = find_short_chunks(df)
    short_examples["issue_type"] = "short_chunk"
    short_examples["issue_detail"] = short_examples["clean_text"].map(token_count).map(lambda x: f"{x} clean tokens")

    suspicious_examples = find_suspicious_chunks(df)
    suspicious_examples["issue_type"] = "too_many_sdg_labels"
    suspicious_examples["issue_detail"] = suspicious_examples["sdg_labels"].map(parse_labels).map(len).map(
        lambda x: f"{x} SDG labels"
    )

    candidates = pd.concat(
        [duplicate_examples, repeated_examples, short_examples, suspicious_examples],
        ignore_index=True,
    )

    for _, row in candidates.iterrows():
        chunk_id = row.get("chunk_id")
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        examples.append(
            {
                "chunk_id": chunk_id,
                "issue_type": row["issue_type"],
                "issue_detail": row["issue_detail"],
                "section_label": row.get("section_label", ""),
                "orientation": row.get("orientation", ""),
                "sdg_labels": row.get("sdg_labels", ""),
                "raw_text_preview": normalize_text(row.get("raw_text", ""))[:280],
            }
        )
        if len(examples) >= MAX_EXAMPLES:
            break

    return pd.DataFrame(examples)


def print_report(df: pd.DataFrame) -> None:
    raw_dupes, clean_dupes = duplicate_chunk_summary(df)
    completeness = label_completeness_summary(df)
    short_chunks = find_short_chunks(df)
    suspicious_chunks = find_suspicious_chunks(df)
    repeated_lines = repeated_first_line_patterns(df)
    label_distribution = sdg_label_distribution(df)
    tfidf_summary = tfidf_output_summary()
    examples = collect_problem_examples(df)

    print("=== Phase 1 Dataset Audit ===")
    print(f"Input file: {CHUNKS_CSV}")
    print(f"Total chunks: {len(df)}")
    print(f"Duplicate raw_text rows: {raw_dupes}")
    print(f"Duplicate clean_text rows: {clean_dupes}")
    print(f"Empty raw_text rows: {int(df['raw_text'].fillna('').str.strip().eq('').sum())}")
    print(f"Empty clean_text rows: {int(df['clean_text'].fillna('').str.strip().eq('').sum())}")
    print(f"Short chunks (<{SHORT_TOKEN_THRESHOLD} clean tokens): {len(short_chunks)}")
    print(f"Suspicious chunks (>={SUSPICIOUS_LABEL_THRESHOLD} SDG labels): {len(suspicious_chunks)}")
    print(f"Missing section labels: {completeness['missing_section_label']}")
    print(f"Missing orientation labels: {completeness['missing_orientation']}")
    print(f"Missing SDG labels: {completeness['missing_sdg_labels']}")

    print("\n--- Repeated First-Line Patterns ---")
    if repeated_lines.empty:
        print("No repeated first-line patterns found.")
    else:
        print(repeated_lines.to_string(index=False))

    print("\n--- SDG Label Count Distribution ---")
    print(label_distribution.to_string(index=False))

    print("\n--- TF-IDF Output Health ---")
    print(tfidf_summary.to_string(index=False))

    print(f"\n--- Example Problematic Chunks (up to {MAX_EXAMPLES}) ---")
    if examples.empty:
        print("No problematic chunks found by the current heuristics.")
    else:
        for _, row in examples.iterrows():
            print(f"\n[chunk_id={row['chunk_id']}] {row['issue_type']} | {row['issue_detail']}")
            print(f"section={row['section_label']} | orientation={row['orientation']}")
            print(f"sdg_labels={row['sdg_labels']}")
            print(row["raw_text_preview"])


def main() -> None:
    if not CHUNKS_CSV.exists():
        raise FileNotFoundError(f"Audit input not found: {CHUNKS_CSV}")

    df = pd.read_csv(CHUNKS_CSV)
    required_cols = {"chunk_id", "raw_text", "clean_text", "section_label", "orientation", "sdg_labels"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print_report(df)


if __name__ == "__main__":
    main()
