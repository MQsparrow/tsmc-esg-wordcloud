# main.py

from pathlib import Path
from src.pipeline import load_spacy_model, run_pipeline, save_outputs

def main():
    text_path = Path("data/raw/tsmc_report.txt")
    raw_text = text_path.read_text(encoding="utf-8")

    nlp = load_spacy_model("en_core_web_sm")

    df_chunks = run_pipeline(raw_text, nlp)
    outputs = save_outputs(df_chunks, output_dir="outputs")

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

    print("\n=== Top section terms ===")
    for section in outputs["section_terms"]["section_label"].unique():
        print(f"\n[{section}]")
        print(
            outputs["section_terms"]
            .loc[outputs["section_terms"]["section_label"] == section, ["term", "tfidf_score"]]
            .head(10)
            .to_string(index=False)
        )

    print("\n=== Top orientation terms ===")
    for ori in outputs["orientation_terms"]["orientation"].unique():
        print(f"\n[{ori}]")
        print(
            outputs["orientation_terms"]
            .loc[outputs["orientation_terms"]["orientation"] == ori, ["term", "tfidf_score"]]
            .head(10)
            .to_string(index=False)
        )

    print("\n=== Top SDG terms ===")
    for sdg in outputs["sdg_terms"]["sdg_labels"].unique():
        print(f"\n[{sdg}]")
        print(
            outputs["sdg_terms"]
            .loc[outputs["sdg_terms"]["sdg_labels"] == sdg, ["term", "tfidf_score"]]
            .head(10)
            .to_string(index=False)
        )

if __name__ == "__main__":
    main()