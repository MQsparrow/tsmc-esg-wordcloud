# main.py

import argparse
from pathlib import Path

from src.phase1_audit import print_report
from src.pipeline import load_spacy_model, run_pipeline, save_outputs

TEXT_PATH = Path("data/raw/tsmc_report.txt")
OUTPUT_DIR = Path("outputs")


def print_top_terms(df, group_col: str, label: str) -> None:
    print(f"\n=== Top {label} terms ===")
    for group in df[group_col].unique():
        print(f"\n[{group}]")
        print(
            df.loc[df[group_col] == group, ["term", "tfidf_score"]]
            .head(10)
            .to_string(index=False)
        )


def main(run_audit: bool = False):
    if not TEXT_PATH.exists():
        raise FileNotFoundError(f"Input text file not found: {TEXT_PATH}")

    raw_text = TEXT_PATH.read_text(encoding="utf-8")

    nlp = load_spacy_model("en_core_web_sm")

    df_chunks = run_pipeline(raw_text, nlp)
    outputs = save_outputs(df_chunks, output_dir=str(OUTPUT_DIR))

    print(df_chunks.head())
    print(outputs["section_terms"].head(20))
    print(outputs["orientation_terms"].head(20))

    print("\n=== Chunk count ===")
    print(len(df_chunks))

    print("\n=== Section label distribution ===")
    print(df_chunks["section_label"].value_counts())

    print("\n=== Orientation distribution ===")
    print(df_chunks["orientation"].value_counts())

    print("\n=== SDG distribution (multi-label, exploded) ===")
    print(df_chunks["sdg_labels"].str.split(",").explode().str.strip().value_counts())

    print("\n=== Empty-ish clean text count (<5 words) ===")
    print((df_chunks["clean_text"].str.split().str.len() < 5).sum())

    print_top_terms(outputs["section_terms"], "section_label", "section")
    print_top_terms(outputs["orientation_terms"], "orientation", "orientation")
    print_top_terms(outputs["sdg_terms"], "sdg_labels", "SDG")
    print_top_terms(outputs["issue_frame_terms"], "issue_frame", "issue frame")

    if run_audit:
        print("\n=== Phase 1 Audit ===")
        print_report(df_chunks)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run the Phase 1 dataset audit after generating outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(run_audit=args.audit)
