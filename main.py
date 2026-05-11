# main.py

import argparse
import contextlib
import io
import sys
from pathlib import Path
from datetime import datetime

from src.phase1_audit import print_report
from src.pipeline import process_all_years

OUTPUT_BASE_DIR = Path("outputs")
LOG_DIR = OUTPUT_BASE_DIR / "logs"


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
    """
    Main entry point: Process all available TSMC reports (2022, 2023, 2024).
    """
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline for all years
    all_results = process_all_years(output_base_dir=str(OUTPUT_BASE_DIR))
    
    # Log results for each year
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = LOG_DIR / f"main_{timestamp}.log"
    latest_log_path = LOG_DIR / "main_latest.log"
    
    with run_log_path.open("w", encoding="utf-8") as log_file:
        tee = TeeStream(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee):
            print(f"Run timestamp: {datetime.now().isoformat(timespec='seconds')}")
            print(f"Output base directory: {OUTPUT_BASE_DIR}")
            print(f"Log file: {run_log_path}\n")
            
            # Log details for each year
            for year in sorted(all_results.keys()):
                result = all_results[year]
                df_chunks = result["chunks"]
                outputs = result["outputs"]
                
                print(f"\n{'='*60}")
                print(f"Year {year} Report")
                print(f"{'='*60}")
                print(f"Input: {result['text_path']}")
                print(f"Output: {result['output_dir']}")
                print(f"\nFirst 5 chunks:")
                print(df_chunks.head())
                print(f"\nTop section terms (first 20):")
                print(outputs["section_terms"].head(20))
                
                print(f"\n=== Chunk count ===")
                print(len(df_chunks))
                
                print(f"\n=== Section label distribution ===")
                print(df_chunks["section_label"].value_counts())
                
                print(f"\n=== Orientation distribution ===")
                print(df_chunks["orientation"].value_counts())
                
                print(f"\n=== SDG distribution (multi-label, exploded) ===")
                print(df_chunks["sdg_labels"].str.split(",").explode().str.strip().value_counts())
                
                print(f"\n=== Empty-ish clean text count (<5 words) ===")
                print((df_chunks["clean_text"].str.split().str.len() < 5).sum())
                
                if run_audit:
                    print(f"\n=== Phase 1 Audit for {year} ===")
                    print_report(df_chunks)
    
    latest_log_path.write_text(run_log_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\nSaved run log to {run_log_path}")




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
