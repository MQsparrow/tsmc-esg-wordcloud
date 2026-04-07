from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

# paths
WORDCLOUD_IMG = Path("outputs/figures/tsmc_wordcloud.png")
KEYWORD_FILE = Path("outputs/models/top_keywords.csv")


# page config
st.set_page_config(
    page_title="TSMC ESG Dashboard",
    layout="wide"
)

st.title("📊 TSMC ESG Text Mining Dashboard")
st.markdown("Analyze sustainability communication using TF-IDF and NLP")

# --- Word Cloud ---
st.header("☁️ Word Cloud")

if WORDCLOUD_IMG.exists():
    st.image(str(WORDCLOUD_IMG), use_container_width=True)
else:
    st.warning("Word cloud image not found. Run visualization.py first.")

# --- Keywords Table ---
st.header("🔑 Top Keywords")

if KEYWORD_FILE.exists():
    df = pd.read_csv(KEYWORD_FILE)

    st.dataframe(df, use_container_width=True)

    # 下載按鈕（加分）
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Download Keywords CSV",
        data=csv,
        file_name="top_keywords.csv",
        mime="text/csv"
    )
else:
    st.warning("Keyword file not found. Run feature.py first.")

# --- Footer ---
st.markdown("---")
st.caption("Built with Streamlit | ESG Text Mining Project")