# main.py

import argparse
import contextlib
import io
import sys
from pathlib import Path
from datetime import datetime

from src.phase1_audit import print_report
from src.pipeline import load_spacy_model, run_pipeline, save_outputs

TEXT_PATH = Path("data/raw/tsmc_report.txt")
OUTPUT_DIR = Path("outputs")
LOG_DIR = OUTPUT_DIR / "logs"


class TeeStream(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    raw_text = TEXT_PATH.read_text(encoding="utf-8")

    nlp = load_spacy_model("en_core_web_sm")

    df_chunks = run_pipeline(raw_text, nlp)
    outputs = save_outputs(df_chunks, output_dir=str(OUTPUT_DIR))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = LOG_DIR / f"main_{timestamp}.log"
    latest_log_path = LOG_DIR / "main_latest.log"

    with run_log_path.open("w", encoding="utf-8") as log_file:
        tee = TeeStream(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee):
            print(f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}")
            print(f"Input text file: {TEXT_PATH}")
            print(f"Output directory: {OUTPUT_DIR}")
            print(f"Log file: {run_log_path}")
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

    latest_log_path.write_text(run_log_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\nSaved run log to {run_log_path}")
    print(f"Updated latest log at {latest_log_path}")


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
