from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

KEYWORD_FILE = Path("outputs/legacy/models/top_keywords.csv")
OUTPUT_DIR = Path("outputs/legacy/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_IMAGE = OUTPUT_DIR / "tsmc_wordcloud.png"


def main() -> None:
    if not KEYWORD_FILE.exists():
        raise FileNotFoundError("top_keywords.csv not found. Run `python main.py` and then `python src/feature.py`.")

    df = pd.read_csv(KEYWORD_FILE)

    # 轉成 wordcloud 需要的 dict 格式
    freq_dict = dict(zip(df["keyword"], df["score"]))

    wordcloud = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        collocations=False
    ).generate_from_frequencies(freq_dict)

    plt.figure(figsize=(16, 9))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved word cloud to: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
