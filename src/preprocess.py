from __future__ import annotations

"""
Legacy compatibility entrypoint.

This project now uses `src.pipeline.py` as the single preprocessing source of truth.
The old NLTK-based preprocessing path was removed to avoid inconsistent chunking and
tokenization between scripts.
"""

from pathlib import Path


SOURCE_OF_TRUTH = Path("outputs/chunks_processed.csv")


def main() -> None:
    print("`src/preprocess.py` is now a legacy compatibility stub.")
    print("Use `python main.py` to run the preprocessing + labeling + TF-IDF pipeline.")

    if SOURCE_OF_TRUTH.exists():
        print(f"Current source-of-truth dataset: {SOURCE_OF_TRUTH}")
    else:
        print("No pipeline output found yet. Run `python main.py` first.")


if __name__ == "__main__":
    main()
