from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from app_pages.analysis import render_page as render_analysis_page
from app_pages.methods import render_page as render_methods_page
from app_pages.project_overview import render_page as render_project_overview_page

try:
    from streamlit_echarts import st_echarts
except ImportError:
    st_echarts = None


px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = px.colors.sequential.Tealgrn
px.defaults.color_discrete_sequence = ["#0f766e", "#0ea5a4", "#2563eb", "#0891b2", "#1d4ed8"]


def inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        .app-shell {
            background:
                radial-gradient(circle at top left, rgba(20,184,166,0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(37,99,235,0.10), transparent 24%),
                linear-gradient(180deg, #f8fffe 0%, #ffffff 35%, #f7fafc 100%);
            border: 1px solid rgba(15,118,110,0.08);
            border-radius: 24px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.06);
        }
        .page-kicker {
            color: #0f766e;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }
        .page-title {
            color: #0f172a;
            font-size: 2.25rem;
            line-height: 1.05;
            font-weight: 800;
            margin: 0 0 0.6rem 0;
        }
        .page-subtitle {
            color: #334155;
            font-size: 1rem;
            line-height: 1.7;
            max-width: 780px;
            margin: 0;
        }
        .soft-card {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            min-height: 210px;
            margin-bottom: 0.1rem;
        }
        .soft-card h4, .highlight-card h4 {
            color: #0f172a;
            margin: 0 0 0.6rem 0;
            font-size: 1.25rem;
            font-weight: 700;
        }
        .soft-card p, .highlight-card p {
            color: #475569;
            margin: 0;
            line-height: 1.65;
            font-size: 0.95rem;
        }
        .highlight-card {
            background: linear-gradient(145deg, #ecfeff 0%, #f8fafc 100%);
            border: 1px solid rgba(14, 165, 164, 0.18);
            border-radius: 18px;
            padding: 1rem;
            min-height: 210px;
            margin-bottom: 0.1rem;
        }
        .mini-kicker {
            color: #0f766e;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
        }
        .section-note {
            color: #64748b;
            font-size: 0.94rem;
            margin-top: -0.45rem;
            margin-bottom: 1rem;
        }
        .card-grid {
            display: grid;
            grid-template-columns: repeat(var(--cols, 2), minmax(0, 1fr));
            gap: 1.1rem;
            margin: 0.5rem 0 1.2rem 0;
        }
        .stat-card {
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            padding: 0.75rem 0.9rem 0.7rem 0.9rem;
            min-height: 90px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
        .stat-card .label {
            color: #64748b;
            font-size: 0.82rem;
            margin-bottom: 0.35rem;
            font-weight: 600;
        }
        .stat-card .value {
            color: #0f172a;
            font-size: 1.6rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 0.25rem;
        }
        .stat-card .note {
            color: #64748b;
            font-size: 0.88rem;
            line-height: 1.5;
        }
        .flow-band {
            background: rgba(255,255,255,0.86);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            margin: 0.65rem 0 1.2rem 0;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            overflow-x: auto;
        }
        .flow-band code {
            color: #0f766e;
            font-size: 1.02rem;
            white-space: nowrap;
        }
        .pipeline-step-card {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 1rem;
            min-height: 210px;
            margin-bottom: 0.85rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .pipeline-step-card::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 4px;
            background: linear-gradient(90deg, #14b8a6 0%, #2563eb 100%);
            opacity: 0.9;
        }
        .pipeline-step-number {
            color: #0f766e;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }
        .pipeline-step-title {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }
        .pipeline-step-body {
            color: #475569;
            font-size: 0.94rem;
            line-height: 1.65;
            margin: 0;
        }
        .tight-section {
            margin-top: 0.4rem;
            margin-bottom: 0.9rem;
        }
        [data-testid="stAlert"] {
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.20);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
        [data-testid="stTabs"] [role="tablist"] {
            gap: 0.5rem;
            background: rgba(255,255,255,0.75);
            padding: 0.35rem;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.18);
        }
        [data-testid="stTabs"] [role="tab"] {
            border-radius: 10px;
            padding: 0.45rem 0.9rem;
            color: #475569;
        }
        [data-testid="stTabs"] [aria-selected="true"] {
            background: linear-gradient(135deg, #0f766e 0%, #0ea5a4 100%);
            color: white;
        }
        .streamlit-expanderHeader {
            font-weight: 700;
            color: #0f172a;
        }
        [data-testid="stExpander"] summary p {
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        [data-testid="stExpander"] {
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            overflow: hidden;
        }
        [data-testid="stExpander"] div[role="region"] {
            min-height: 250px;
        }
        [data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }
        .reveal-ready {
            opacity: 0;
            transform: translateY(16px);
            transition: opacity 0.45s ease, transform 0.45s ease;
        }
        .reveal-ready.reveal-visible {
            opacity: 1;
            transform: translateY(0);
        }
        [data-testid="stMetricValue"] {
            color: #0f172a;
            font-weight: 800;
        }
        [data-testid="stMetricLabel"] {
            color: #475569;
            font-weight: 600;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fafc 0%, #eff6ff 100%);
        }
        [data-testid="stHorizontalBlock"] {
            gap: 1.15rem;
            align-items: stretch;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-top: calc(4.75rem + env(safe-area-inset-top));
                padding-left: 1.15rem;
                padding-right: 1.15rem;
                padding-bottom: 2.25rem;
                max-width: 100%;
            }
            .app-shell {
                padding: 1.15rem 1rem;
                border-radius: 18px;
                margin-bottom: 1rem;
            }
            .page-title {
                font-size: 1.7rem;
            }
            .page-subtitle {
                font-size: 0.95rem;
                line-height: 1.6;
            }
            [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                align-items: stretch !important;
                gap: 1rem !important;
            }
            [data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
                padding-left: 0.05rem !important;
                padding-right: 0.05rem !important;
            }
            .soft-card, .highlight-card {
                min-height: auto;
                padding: 0.95rem;
                margin-bottom: 0.2rem;
            }
            .stat-card {
                min-height: auto;
                padding: 0.9rem;
                margin-bottom: 0.2rem;
            }
            .stat-card .value {
                font-size: 2rem;
            }
            .flow-band {
                padding: 0.8rem 0.9rem;
            }
            .flow-band code {
                font-size: 0.92rem;
            }
            .pipeline-step-card {
                min-height: auto;
                padding: 0.9rem;
                margin-bottom: 0.85rem;
            }
            [data-testid="stTabs"] [role="tablist"] {
                flex-wrap: wrap;
                gap: 0.35rem;
                padding: 0.25rem;
            }
            [data-testid="stTabs"] [role="tab"] {
                flex: 1 1 calc(50% - 0.35rem);
                justify-content: center;
                min-height: 42px;
                padding: 0.45rem 0.6rem;
                font-size: 0.92rem;
            }
            [data-testid="stExpander"] summary p {
                white-space: normal !important;
                overflow: visible !important;
                text-overflow: unset !important;
            }
            [data-testid="stExpander"] div[role="region"] {
                min-height: auto;
            }
            [data-testid="stPlotlyChart"] {
                margin-bottom: 0.35rem;
            }
            [data-testid="stDataFrame"] {
                overflow-x: auto;
            }
            iframe {
                max-width: 100% !important;
            }
            [data-testid="stSidebar"] {
                min-width: 82vw !important;
                max-width: 82vw !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="app-shell">
            <div class="page-kicker">{kicker}</div>
            <div class="page-title">{title}</div>
            <p class="page-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_text_card(title: str, text: str, highlight: bool = False) -> None:
    card_class = "highlight-card" if highlight else "soft-card"
    st.markdown(
        f"""
        <div class="{card_class}">
            <h4>{title}</h4>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kicker_card(kicker: str, title: str, text: str, highlight: bool = False) -> None:
    card_class = "highlight-card" if highlight else "soft-card"
    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="mini-kicker">{kicker}</div>
            <h4>{title}</h4>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_card_grid(cards: list[dict], columns: int = 2) -> None:
    for start in range(0, len(cards), columns):
        row_cards = cards[start:start + columns]
        cols = st.columns(columns, gap="large")
        for col, card in zip(cols, row_cards):
            with col:
                if card.get("kicker"):
                    render_kicker_card(
                        kicker=card["kicker"],
                        title=card["title"],
                        text=card["text"],
                        highlight=card.get("highlight", False),
                    )
                else:
                    render_text_card(
                        title=card["title"],
                        text=card["text"],
                        highlight=card.get("highlight", False),
                    )
        if len(row_cards) < columns:
            for col in cols[len(row_cards):]:
                with col:
                    st.empty()
        if start + columns < len(cards):
            st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)


def render_metric_grid(metrics: list[dict], columns: int = 4) -> None:
    for start in range(0, len(metrics), columns):
        row_metrics = metrics[start:start + columns]
        cols = st.columns(columns, gap="large")
        for col, metric in zip(cols, row_metrics):
            note = f'<div class="note">{metric["note"]}</div>' if metric.get("note") else ""
            with col:
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="label">{metric['label']}</div>
                        <div class="value stat-value" data-target="{metric['value']}">{metric['value']}</div>
                        {note}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        if len(row_metrics) < columns:
            for col in cols[len(row_metrics):]:
                with col:
                    st.empty()
        if start + columns < len(metrics):
            st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)


# def render_flow_band(text: str) -> None:
#     st.markdown(
#         f'<div class="flow-band"><code>{text}</code></div>',
#         unsafe_allow_html=True,
#     )

def render_flow_band(flow_string):
    steps = [step.strip() for step in flow_string.split("->")]
    
    st.markdown("""
        <style>
        .flow-container {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
            padding: 15px;
            background-color: #f8f9fb;
            border-radius: 8px;
        }
        .flow-step {
            background: #e0f2f1;
            color: #00796b;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 500;
            font-size: 1.5rem;
            border: 1px solid #b2dfdb;
        }
        .flow-arrow {
            color: #999;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # 生成帶有箭頭的 HTML
    flow_html = '<div class="flow-container">'
    for i, step in enumerate(steps):
        flow_html += f'<div class="flow-step">{step}</div>'
        if i < len(steps) - 1:
            flow_html += '<span class="flow-arrow">→</span>'
    flow_html += '</div>'
    
    st.markdown(flow_html, unsafe_allow_html=True)


def render_pipeline_steps(steps: list[dict], columns: int = 3) -> None:
    for start in range(0, len(steps), columns):
        row_steps = steps[start:start + columns]
        cols = st.columns(columns, gap="large")
        for col, step in zip(cols, row_steps):
            with col:
                st.markdown(
                    f"""
                    <div class="pipeline-step-card">
                        <div class="pipeline-step-number">{step['number']}</div>
                        <div class="pipeline-step-title">{step['title']}</div>
                        <p class="pipeline-step-body">{step['text']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        if len(row_steps) < columns:
            for col in cols[len(row_steps):]:
                with col:
                    st.empty()

def inject_motion_script() -> None:
    components.html(
        """
        <script>
        const isMobile = () => window.parent.innerWidth <= 768;
        const selectors = [
          '.app-shell',
          '.soft-card',
          '.highlight-card',
          '.stat-card',
          '[data-testid="stMetric"]',
          '[data-testid="stPlotlyChart"]',
          '[data-testid="stExpander"]',
          '[data-testid="stDataFrame"]'
        ];

        function markRevealTargets() {
          selectors.forEach((selector) => {
            window.parent.document.querySelectorAll(selector).forEach((el) => {
              if (!el.classList.contains('reveal-ready')) {
                el.classList.add('reveal-ready');
              }
            });
          });
        }

        function observeRevealTargets() {
          const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry) => {
              if (entry.isIntersecting) {
                entry.target.classList.add('reveal-visible');
              }
            });
          }, { threshold: 0.08 });

          window.parent.document.querySelectorAll('.reveal-ready').forEach((el) => observer.observe(el));
        }

        function animateStatValues() {
          window.parent.document.querySelectorAll('.stat-value').forEach((el) => {
            if (el.dataset.animated === 'true') return;
            const target = Number(el.dataset.target || el.textContent || 0);
            if (!Number.isFinite(target)) return;
            el.dataset.animated = 'true';
            const duration = 700;
            const start = performance.now();

            function tick(now) {
              const progress = Math.min((now - start) / duration, 1);
              const eased = 1 - Math.pow(1 - progress, 3);
              el.textContent = Math.round(target * eased).toString();
              if (progress < 1) requestAnimationFrame(tick);
            }
            requestAnimationFrame(tick);
          });
        }

        function attachCardTilt() {
          if (isMobile()) return;
          window.parent.document.querySelectorAll('.soft-card, .highlight-card, .stat-card').forEach((card) => {
            if (card.dataset.tiltAttached === 'true') return;
            card.dataset.tiltAttached = 'true';
            card.style.transition = 'transform 0.2s ease, box-shadow 0.2s ease';

            card.addEventListener('mousemove', (event) => {
              const rect = card.getBoundingClientRect();
              const x = event.clientX - rect.left;
              const y = event.clientY - rect.top;
              const rotateY = ((x / rect.width) - 0.5) * 5;
              const rotateX = ((0.5 - y / rect.height)) * 5;
              card.style.transform = `perspective(900px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-2px)`;
              card.style.boxShadow = '0 18px 36px rgba(15, 23, 42, 0.10)';
            });

            card.addEventListener('mouseleave', () => {
              card.style.transform = '';
              card.style.boxShadow = '';
            });
          });
        }

        setTimeout(() => {
          markRevealTargets();
          observeRevealTargets();
          animateStatValues();
          attachCardTilt();
        }, 120);
        </script>
        """,
        height=0,
    )


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="TSMC ESG Word Cloud Dashboard",
    page_icon="🌍",
    layout="wide"
)

inject_custom_styles()
render_page_header(
    "TSMC ESG Text Mining",
    "Decoding TSMC's Sustainability Language",
    "A text-mining dashboard that reveals how TSMC frames ESG topics across Strategic Framing and UN SDG themes.",
)
inject_motion_script()


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

CHUNKS_PATH = OUTPUT_DIR / "chunks_processed.csv"
SECTION_TFIDF_PATH = OUTPUT_DIR / "tfidf_by_section.csv"
SDG_TFIDF_PATH = OUTPUT_DIR / "tfidf_by_sdg.csv"
SECTION_SIMILARITY_PATH = OUTPUT_DIR / "similarity_by_section.csv"
SDG_SIMILARITY_PATH = OUTPUT_DIR / "similarity_by_sdg.csv"


# =========================
# Constants
# =========================
SECTION_ORDER = ["environment", "talent", "supply_chain", "social", "governance"]
SDG_ORDER = ["SDG3_health", "SDG4_education", "SDG6_water", "SDG7_energy", "SDG8_labor", "SDG9_innovation", "SDG12_consumption", "SDG13_climate", "SDG17_partnership"]
PLOTLY_INTERACTIVE_CONFIG = {"scrollZoom": True, "displayModeBar": True}
WORDCLOUD_COLORS = ["#0f766e", "#0891b2", "#2563eb", "#059669", "#0ea5e9", "#14b8a6"]

SECTION_LABEL_MAP = {
    "environment": "Environment",
    "talent": "Talent",
    "supply_chain": "Supply Chain",
    "social": "Social",
    "governance": "Governance",
}

SDG_LABEL_MAP = {
    "SDG3_health": "SDG 3: Health",
    "SDG4_education": "SDG 4: Education",
    "SDG6_water": "SDG 6: Water",
    "SDG7_energy": "SDG 7: Energy",
    "SDG8_labor": "SDG 8: Labor",
    "SDG9_innovation": "SDG 9: Innovation",
    "SDG12_consumption": "SDG 12: Consumption",
    "SDG13_climate": "SDG 13: Climate",
    "SDG17_partnership": "SDG 17: Partnership",
}

SDG_NARRATIVE = {
    "SDG17_partnership": (
        "**Partnership dominates because it has to.**\n\n"
        "SDG 17 has the highest chunk count and its top terms are *committee, carbon reduction, chemical* — "
        "this isn't goodwill language, it's compliance infrastructure. TSMC is building governance structures "
        "to push Scope 3 emissions and chemical accountability down to suppliers, driven by CSRD exposure "
        "from the Dresden fab."
    ),
    "SDG4_education": (
        "**Education coverage is inflated.**\n\n"
        "SDG 4 ranks highly but top terms *learning, senior, care* suggest significant overlap with talent "
        "retention and social welfare content. The keyword boundary is leaking — not all of these chunks "
        "are genuinely about education."
    ),
    "SDG12_consumption": (
        "**Consumption reflects manufacturing reality.**\n\n"
        "Top terms *chemical, recycle, acid* point to TSMC's fab-level material management: "
        "acid recovery, chemical reuse loops, and hazardous waste reduction — core to responsible "
        "semiconductor production at scale."
    ),
    "SDG13_climate": (
        "**Climate is the backbone of ESG reporting.**\n\n"
        "Top terms *carbon reduction, equivalent, electricity* reflect TSMC's RE100 commitment and "
        "Scope 2 decarbonization strategy. The high chunk count signals that climate disclosure "
        "is embedded across sections, not siloed."
    ),
    "SDG8_labor": (
        "**Labor language is precise and formal.**\n\n"
        "Top terms *permanent employee, intern, recruitment* indicate structured HR reporting. "
        "Coverage is driven by headcount disclosures and employment contract data, not narrative."
    ),
    "SDG9_innovation": (
        "**Innovation is IP-driven.**\n\n"
        "Top terms *patent, trade secret, patent application* show that TSMC frames innovation "
        "through intellectual property — not just R&D spend. This reflects a defensive IP strategy "
        "alongside process node advancement."
    ),
    "SDG3_health": (
        "**Health focus is safety-led.**\n\n"
        "Top terms *safety health, chemical, injury* indicate occupational health dominates over "
        "community health. Chemical exposure and injury prevention are the primary concerns in a fab environment."
    ),
    "SDG6_water": (
        "**Water and chemical management are intertwined.**\n\n"
        "Top terms *chemical, indicator, acid* reflect that water reporting in semiconductor fabs "
        "is inseparable from chemical discharge management — acidic wastewater treatment is the core challenge."
    ),
    "SDG7_energy": (
        "**Energy reporting centers on carbon.**\n\n"
        "Top terms *carbon reduction, electricity, charity* show decarbonization of electricity "
        "consumption is the primary energy narrative, aligned with TSMC's RE100 and net-zero targets."
    ),
}

SECTION_NARRATIVE = {
    "environment": "Environment language centers on operational resource management, especially water, energy, waste, and carbon.",
    "talent": "Talent language emphasizes employees, health, safety, and development, showing a people-and-workforce framing.",
    "supply_chain": "Supply chain language is compliance-heavy, with supplier controls, audits, and material-risk management standing out.",
    "social": "Social language highlights foundations, education, volunteering, and public-facing community programs.",
    "governance": "Governance language is structured around oversight, directors, committees, tax, and risk management.",
}

# =========================
# Load data
# =========================
@st.cache_data
def load_data():
    df_chunks = pd.read_csv(CHUNKS_PATH)
    df_section = pd.read_csv(SECTION_TFIDF_PATH)
    df_sdg = pd.read_csv(SDG_TFIDF_PATH)
    df_section_similarity = pd.read_csv(SECTION_SIMILARITY_PATH, index_col=0)
    df_sdg_similarity = pd.read_csv(SDG_SIMILARITY_PATH, index_col=0)

    for col in ["raw_text", "clean_text", "section_label", "orientation", "sdg_labels"]:
        if col in df_chunks.columns:
            df_chunks[col] = df_chunks[col].fillna("").astype(str)
    if "sdg_confidence" in df_chunks.columns:
        df_chunks["sdg_confidence"] = pd.to_numeric(df_chunks["sdg_confidence"], errors="coerce").fillna(0.0)

    return (
        df_chunks,
        df_section,
        df_sdg,
        df_section_similarity,
        df_sdg_similarity,
    )


(
    df_chunks,
    df_section_tfidf,
    df_sdg_tfidf,
    df_section_similarity,
    df_sdg_similarity,
) = load_data()


# =========================
# Helper functions
# =========================
def prettify_label(x: str) -> str:
    if x in SECTION_LABEL_MAP:
        return SECTION_LABEL_MAP[x]
    if x in SDG_LABEL_MAP:
        return SDG_LABEL_MAP[x]
    return x.replace("_", " ").title()


def compact_heatmap_label(x: str) -> str:
    label = prettify_label(x)
    sdg_match = re.match(r"SDG\s+(\d+)", label)
    if sdg_match:
        return f"SDG {sdg_match.group(1)}"
    return label


def parse_sdg_labels(label_text: str) -> list[str]:
    if not label_text:
        return []
    return [label.strip() for label in str(label_text).split(",") if label.strip() and label.strip() != "unclassified"]


def explode_sdg_chunks(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(sdg_labels=df["sdg_labels"].map(parse_sdg_labels))
        .explode("sdg_labels")
        .dropna(subset=["sdg_labels"])
    )


def filter_chunks_by_sdg(df: pd.DataFrame, sdg: str) -> pd.DataFrame:
    return df[df["sdg_labels"].map(parse_sdg_labels).map(lambda labels: sdg in labels)].copy()


def make_wordcloud_from_tfidf(
    df: pd.DataFrame,
    term_col: str = "term",
    score_col: str = "tfidf_score",
):
    freq = {row[term_col]: float(row[score_col]) for _, row in df.iterrows()}
    if not freq:
        return None

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        collocations=False,
        max_words=100
    ).generate_from_frequencies(freq)
    return wc


def plot_wordcloud(wc: WordCloud):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)


def render_interactive_wordcloud(df: pd.DataFrame, top_k: int = 60):
    if df.empty:
        st.info("No terms available.")
        return

    if st_echarts is None:
        wc = make_wordcloud_from_tfidf(df.head(80))
        if wc:
            plot_wordcloud(wc)
        else:
            st.info("No terms available.")
        return

    data = (
        df.sort_values("tfidf_score", ascending=False)
        .head(top_k)[["term", "tfidf_score"]]
        .copy()
    )
    max_score = float(data["tfidf_score"].max()) if not data.empty else 1.0
    min_score = float(data["tfidf_score"].min()) if not data.empty else 0.0
    score_span = max(max_score - min_score, 1e-9)
    word_data = [
        {
            "name": row["term"],
            "value": round(28 + (float(row["tfidf_score"]) - min_score) * 92 / score_span, 2),
            "textStyle": {"color": color},
        }
        for (_, row), color in zip(
            data.iterrows(),
            WORDCLOUD_COLORS * 20,
        )
    ]

    options = {
        "tooltip": {"show": True},
        "series": [
            {
                "type": "wordCloud",
                "shape": "circle",
                "keepAspect": True,
                "left": "center",
                "top": "center",
                "width": "100%",
                "height": 420,
                "sizeRange": [18, 72],
                "rotationRange": [-45, 45],
                "rotationStep": 15,
                "gridSize": 10,
                "drawOutOfBound": False,
                "textStyle": {
                    "fontFamily": "Arial",
                    "fontWeight": "bold",
                },
                "emphasis": {
                    "focus": "self",
                    "textStyle": {
                        "shadowBlur": 8,
                        "shadowColor": "#94a3b8",
                    },
                },
                "data": word_data,
            }
        ],
    }
    st_echarts(options=options, height="440px")


def plot_top_terms(df: pd.DataFrame, title: str, top_k: int = 15, chart_key: str | None = None):
    df_plot = df.sort_values("tfidf_score", ascending=False).head(top_k).copy()

    fig = px.bar(
        df_plot.iloc[::-1],
        x="tfidf_score",
        y="term",
        orientation="h",
        title=title,
        color="tfidf_score",
    )
    fig.update_layout(
        height=500,
        xaxis_title="TF-IDF Score",
        yaxis_title="Term",
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_showscale=False,
        plot_bgcolor="rgba(255,255,255,0.92)",
        paper_bgcolor="rgba(255,255,255,0.0)",
    )
    fig.update_traces(
        hovertemplate="Term: %{y}<br>TF-IDF: %{x:.4f}<extra></extra>"
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOTLY_INTERACTIVE_CONFIG,
        key=chart_key,
    )


def get_top_terms_table(df: pd.DataFrame, top_k: int = 20):
    out = (
        df.sort_values("tfidf_score", ascending=False)
        .head(top_k)[["term", "tfidf_score"]]
        .reset_index(drop=True)
        .copy()
    )
    out["tfidf_score"] = out["tfidf_score"].round(4)
    return out


def get_top_terms_list(df: pd.DataFrame, top_k: int = 8) -> list[str]:
    if df.empty:
        return []
    return (
        df.sort_values("tfidf_score", ascending=False)
        .head(top_k)["term"]
        .astype(str)
        .tolist()
    )


def aggregate_terms_for_overview(*dfs: pd.DataFrame, top_k: int = 60) -> pd.DataFrame:
    valid_frames = [df[["term", "tfidf_score"]].copy() for df in dfs if not df.empty]
    if not valid_frames:
        return pd.DataFrame(columns=["term", "tfidf_score"])

    combined = pd.concat(valid_frames, ignore_index=True)
    aggregated = (
        combined.groupby("term", as_index=False)["tfidf_score"]
        .mean()
        .sort_values("tfidf_score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    return aggregated


def chunk_match_count(raw_text: str, terms: list[str]) -> int:
    text = str(raw_text).lower()
    return sum(1 for term in terms if re.search(r"\b" + re.escape(term.lower()) + r"\w*\b", text))


def rank_representative_chunks(
    df: pd.DataFrame,
    df_terms: pd.DataFrame,
    max_chunks: int,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    ranked = df.copy()
    top_terms = get_top_terms_list(df_terms, top_k=3)
    ranked["token_count"] = ranked["clean_text"].fillna("").str.split().str.len()
    ranked["term_matches"] = ranked["raw_text"].fillna("").map(lambda text: chunk_match_count(text, top_terms))
    if "sdg_confidence" not in ranked.columns:
        ranked["sdg_confidence"] = 0.0

    ranked = ranked[ranked["term_matches"] >= 2]
    ranked = ranked.sort_values(
        ["term_matches", "sdg_confidence", "token_count"],
        ascending=[False, False, False],
    )
    return ranked.head(max_chunks).copy()


def render_interpretation(title: str, text: str):
    st.markdown(f"**Interpretation: {title}**")
    st.caption(text)


def highlight_terms_in_text(text: str, terms: list[str]) -> str:
    highlighted = text
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(r"(?i)\b" + re.escape(term) + r"\w*\b")
        highlighted = pattern.sub(
            lambda m: f'<mark style="background-color:#fef08a;padding:1px 3px;border-radius:3px;font-weight:600">{m.group()}</mark>',
            highlighted,
        )
    return highlighted


def extract_key_sentences(text: str, terms: list[str], max_sentences: int = 2) -> list[str]:
    fragments = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    sentences = []
    for frag in fragments:
        frag = frag.strip()
        if len(frag.split()) < 5:
            continue
        sentences.append(frag)

    scored = []
    for sent in sentences:
        lower = sent.lower()
        hits = sum(1 for t in terms if re.search(r"\b" + re.escape(t.lower()) + r"\w*\b", lower))
        if hits > 0:
            words = sent.split()
            if len(words) > 35:
                best_start = 0
                best_hits = 0
                for start in range(0, max(1, len(words) - 34)):
                    window = " ".join(words[start:start + 35]).lower()
                    wh = sum(1 for t in terms if re.search(r"\b" + re.escape(t.lower()) + r"\w*\b", window))
                    if wh > best_hits:
                        best_hits = wh
                        best_start = start
                excerpt = " ".join(words[best_start:best_start + 35])
                if best_start > 0:
                    excerpt = "... " + excerpt
                if best_start + 35 < len(words):
                    excerpt += " ..."
            else:
                excerpt = sent
            scored.append((hits, excerpt))
    scored.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    result = []
    for _, sent in scored:
        if sent not in seen:
            seen.add(sent)
            result.append(sent)
        if len(result) >= max_sentences:
            break
    return result


def render_chunk_cards(df: pd.DataFrame, df_terms: pd.DataFrame, max_chunks: int = 2, **_kwargs):
    view_df = df.copy()
    view_df = rank_representative_chunks(view_df, df_terms, max_chunks=min(max_chunks, 2))
    top_terms = get_top_terms_list(df_terms, top_k=3)

    cols = st.columns(len(view_df))
    for col, (_, row) in zip(cols, view_df.iterrows()):
        raw = str(row["raw_text"]).replace("~", "")
        matched_terms = [term for term in top_terms if re.search(r"\b" + re.escape(term.lower()) + r"\w*\b", raw.lower())]
        key_sentences = extract_key_sentences(raw, matched_terms, max_sentences=2)

        with col:
            if key_sentences:
                for sent in key_sentences:
                    highlighted = highlight_terms_in_text(sent, matched_terms)
                    st.markdown(
                        f'<div style="font-size:0.85rem;line-height:1.6;margin-bottom:8px;padding:10px 14px;background:#f8fafc;border-left:3px solid #0891b2;border-radius:4px">{highlighted}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No key sentences found.")


def make_frequency_dict(df: pd.DataFrame, group_col: str, ordered_groups: list[str], top_n: int = 20):
    out = {}
    for group in ordered_groups:
        sub = (
            df[df[group_col] == group]
            .sort_values("tfidf_score", ascending=False)
            .head(top_n)
        )
        if len(sub) > 0:
            out[group] = dict(zip(sub["term"], sub["tfidf_score"]))
    return out


def build_overlap_heatmap(df: pd.DataFrame, group_col: str, ordered_groups: list[str], top_n: int = 20):
    freq_dict = make_frequency_dict(df, group_col=group_col, ordered_groups=ordered_groups, top_n=top_n)
    groups = list(freq_dict.keys())

    matrix = []
    for g1 in groups:
        row = []
        set1 = set(freq_dict[g1].keys())
        for g2 in groups:
            set2 = set(freq_dict[g2].keys())
            overlap = len(set1 & set2)
            row.append(overlap)
        matrix.append(row)

    return pd.DataFrame(matrix, index=groups, columns=groups)


def plot_heatmap(heatmap_df: pd.DataFrame, title: str, chart_key: str | None = None):
    pretty_index = [compact_heatmap_label(x) for x in heatmap_df.index]
    pretty_cols = [compact_heatmap_label(x) for x in heatmap_df.columns]
    hover_index = [prettify_label(x) for x in heatmap_df.index]
    hover_cols = [prettify_label(x) for x in heatmap_df.columns]
    values = heatmap_df.values
    is_integer_like = np.allclose(values, np.round(values))
    chart_width = max(720, 82 * len(pretty_cols) + 220)

    fig = px.imshow(
        values,
        x=pretty_cols,
        y=pretty_index,
        text_auto=".0f" if is_integer_like else ".2f",
        aspect="auto",
        title=title,
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        width=chart_width,
        height=480,
        margin=dict(l=28, r=18, t=60, b=64),
        xaxis_title="",
        yaxis_title="",
        plot_bgcolor="rgba(255,255,255,0.92)",
        paper_bgcolor="rgba(255,255,255,0.0)",
        xaxis=dict(tickangle=90, automargin=True, tickfont=dict(size=11)),
        yaxis=dict(automargin=True, tickfont=dict(size=11)),
    )
    fig.update_traces(
        customdata=[
            [{"full_x": hover_cols[col_idx], "full_y": hover_index[row_idx]} for col_idx in range(len(hover_cols))]
            for row_idx in range(len(hover_index))
        ],
        hovertemplate=(
            "X: %{customdata.full_x}<br>Y: %{customdata.full_y}<br>Value: %{z:.0f}<extra></extra>"
            if is_integer_like
            else "X: %{customdata.full_x}<br>Y: %{customdata.full_y}<br>Value: %{z:.2f}<extra></extra>"
        ),
        textfont=dict(size=9),
    )
    scrollable_chart_html = f"""
    <div style="width:100%; overflow-x:auto; overflow-y:hidden; padding-bottom:0.35rem;">
        <div style="min-width:{chart_width}px;">
            {fig.to_html(full_html=False, include_plotlyjs='cdn', config=PLOTLY_INTERACTIVE_CONFIG)}
        </div>
    </div>
    """
    components.html(
        scrollable_chart_html,
        height=540,
        scrolling=False,
    )


def summarize_similarity_pairs(similarity_df: pd.DataFrame, top_n: int = 3, ascending: bool = False) -> pd.DataFrame:
    if similarity_df.empty:
        return pd.DataFrame(columns=["group_a", "group_b", "similarity"])

    records = []
    labels = similarity_df.index.tolist()
    for i, label_a in enumerate(labels):
        for j, label_b in enumerate(labels):
            if j <= i:
                continue
            records.append(
                {
                    "group_a": label_a,
                    "group_b": label_b,
                    "similarity": float(similarity_df.iloc[i, j]),
                }
            )

    if not records:
        return pd.DataFrame(columns=["group_a", "group_b", "similarity"])

    out = pd.DataFrame(records).sort_values("similarity", ascending=ascending).head(top_n).copy()
    out["group_a"] = out["group_a"].map(prettify_label)
    out["group_b"] = out["group_b"].map(prettify_label)
    out["similarity"] = out["similarity"].round(3)
    return out.reset_index(drop=True)


def plot_distribution_bar(df: pd.DataFrame, col: str, title: str, chart_key: str | None = None):
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    counts[col] = counts[col].map(prettify_label)
    counts = counts.sort_values("count", ascending=False).reset_index(drop=True)

    fig = px.bar(
        counts,
        x=col,
        y="count",
        title=title,
        text="count",
        color="count",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="",
        yaxis_title="Count",
        coloraxis_showscale=False,
        plot_bgcolor="rgba(255,255,255,0.92)",
        paper_bgcolor="rgba(255,255,255,0.0)",
    )
    fig.update_traces(
        hovertemplate="Label: %{x}<br>Count: %{y}<extra></extra>"
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOTLY_INTERACTIVE_CONFIG,
        key=chart_key,
    )




_SDG9_EXCLUSIVE_TERMS = {"trade_secret", "trade_secret registration", "patent application", "patent"}

def get_top_insight(df_terms: pd.DataFrame, label_col: str, top_k: int = 3):
    insights = []
    for label in df_terms[label_col].unique():
        sub = df_terms[df_terms[label_col] == label].sort_values("tfidf_score", ascending=False)
        if label_col == "sdg_labels" and label != "SDG9_innovation":
            sub = sub[~sub["term"].isin(_SDG9_EXCLUSIVE_TERMS)]
        sub = sub.head(top_k)
        terms = ", ".join(sub["term"].tolist())
        insights.append((prettify_label(label), terms))
    return insights



def filter_chunks_by_keyword(df: pd.DataFrame, keyword: str):
    if not keyword.strip():
        return df.copy()

    kw = keyword.strip().lower()
    return df[df["raw_text"].str.lower().str.contains(kw, na=False, regex=False)].copy()


# =========================
# Sidebar
# =========================
st.sidebar.header("Controls")

page_mode = st.sidebar.selectbox(
    "Page",
    ["Project Overview", "Analysis", "Methods"]
)

theme_mode = None
if page_mode == "Analysis":
    theme_mode = st.sidebar.radio(
        "Theme",
        ["Strategic Framing", "UN SDG Themes"]
    )

top_k = st.sidebar.slider("Top keywords", min_value=10, max_value=30, value=15, step=5)

st.sidebar.markdown("---")
keyword_search = st.sidebar.text_input("Keyword search in raw text", "")

filtered_chunks = filter_chunks_by_keyword(df_chunks, keyword_search)


# =========================
# Top-level Pages
# =========================
if page_mode == "Project Overview":
    render_project_overview_page(
        df_chunks=df_chunks,
        df_section_tfidf=df_section_tfidf,
        df_sdg_tfidf=df_sdg_tfidf,
        render_page_header=render_page_header,
        render_metric_grid=render_metric_grid,
        render_card_grid=render_card_grid,
        aggregate_terms_for_overview=aggregate_terms_for_overview,
        render_interactive_wordcloud=render_interactive_wordcloud,
        st=st,
    )
elif False and page_mode == "Methods":
    render_methods_page(
        render_page_header=render_page_header,
        render_pipeline_steps=render_pipeline_steps,
        render_card_grid=render_card_grid,
        st=st,
    )
elif False and page_mode == "Analysis":
    render_analysis_page(
        theme_mode=theme_mode,
        top_k=top_k,
        df_chunks=df_chunks,
        filtered_chunks=filtered_chunks,
        df_section_tfidf=df_section_tfidf,
        df_sdg_tfidf=df_sdg_tfidf,
        df_section_similarity=df_section_similarity,
        df_sdg_similarity=df_sdg_similarity,
        render_page_header=render_page_header,
        render_metric_grid=render_metric_grid,
        render_interpretation=render_interpretation,
        plot_distribution_bar=plot_distribution_bar,
        plot_top_terms=plot_top_terms,
        render_interactive_wordcloud=render_interactive_wordcloud,
        build_overlap_heatmap=build_overlap_heatmap,
        plot_heatmap=plot_heatmap,
        summarize_similarity_pairs=summarize_similarity_pairs,
        render_chunk_cards=render_chunk_cards,
        explode_sdg_chunks=explode_sdg_chunks,
        filter_chunks_by_sdg=filter_chunks_by_sdg,
        SECTION_LABEL_MAP=SECTION_LABEL_MAP,
        SECTION_ORDER=SECTION_ORDER,
        SECTION_NARRATIVE=SECTION_NARRATIVE,
        SDG_LABEL_MAP=SDG_LABEL_MAP,
        SDG_ORDER=SDG_ORDER,
        SDG_NARRATIVE=SDG_NARRATIVE,
        st=st,
    )

if False and page_mode == "Project Overview":
    render_page_header(
        "The Pitch",
        "Project Overview",
        "",
    )

    render_metric_grid(
        [
            {"label": "Original Report", "value": "200+ pages"},
            {"label": "Strategic Framing", "value": f"{df_chunks['section_label'].nunique()} sections"},
            {
                "label": "UN 2030 Agenda",
                "value": f'{df_chunks["sdg_labels"].str.split(",").explode().str.strip().pipe(lambda s: s[s != "unclassified"]).nunique()} themes',
            },
        ],
        columns=3,
    )
    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    render_card_grid(
        [
            {
                "title": "What We Decode",
                "text": "<ul style='margin:0;padding-left:1.2rem;color:#475569;line-height:1.8;font-size:0.95rem'>"
                        "<li>Source: <strong>TSMC 2024 Sustainability Report</strong> — environment, talent, supply chain, social impact, governance</li>"
                        "<li>Mapped to two universal frameworks: <strong>Strategic Framing</strong> and the <strong>UN SDGs (2030 Agenda)</strong></li>"
                        "<li>Every finding speaks a language investors, regulators, and academics already understand</li>"
                        "</ul>",
                "highlight": True,
            },
            {
                "title": "Why It Matters",
                "text": "<ul style='margin:0;padding-left:1.2rem;color:#475569;line-height:1.8;font-size:0.95rem'>"
                        "<li>ESG reports are <strong>strategic narratives</strong>, not neutral disclosure</li>"
                        "<li>We quantify vocabulary patterns to reveal where TSMC invests narrative weight "
                        "(e.g. climate compliance, supplier oversight) vs. where it stays generic (e.g. labor rights, innovation IP)</li>"
                        "<li>Result: a dense PDF becomes an <strong>auditable, comparable language profile</strong></li>"
                        "</ul>",
            },
        ],
        columns=2,
    )
    st.markdown("")

    st.markdown("### What Do You See Before We Decode?")
    st.caption("A first glance at the report's most distinctive vocabulary — sized by TF-IDF weight. The patterns become clearer in the Analysis page.")
    overview_wordcloud_terms = aggregate_terms_for_overview(
        df_section_tfidf,
        df_sdg_tfidf,
        top_k=70,
    )
    render_interactive_wordcloud(overview_wordcloud_terms, top_k=70)

elif page_mode == "Methods":
    render_page_header(
        "Methods",
        "How The Pipeline Works",
        "A concise walkthrough of how raw report text becomes a cleaned, labeled, and interpretable dashboard.",
    )

    st.markdown("### Pipeline Flow")
    st.markdown(
        '<div class="section-note">The workflow keeps each transformation explainable. We move from raw PDF text to cleaned chunks, then add labels and analysis layers only after the text is stable enough to defend.</div>',
        unsafe_allow_html=True,
    )
    render_pipeline_steps(
        [
            {
                "number": "Step 1",
                "title": "Extract",
                "text": "Pull text from the PDF and preserve enough structure to trace the source before cleanup begins.",
            },
            {
                "number": "Step 2",
                "title": "Clean",
                "text": "Remove page markers, repeated headers, appendix boilerplate, and leftover PDF artifacts that would distort later metrics.",
            },
            {
                "number": "Step 3",
                "title": "Chunk",
                "text": "Split the report into paragraph-level units and drop noisy fragments so each chunk holds a cleaner semantic idea.",
            },
            {
                "number": "Step 4",
                "title": "Preprocess",
                "text": "Use spaCy for normalization, lemmatization, stopword removal, and POS filtering so grouped language is comparable.",
            },
            {
                "number": "Step 5",
                "title": "Label",
                "text": "Assign Strategic Framing, orientation, SDG, and issue-frame labels with rule-based logic, stronger thresholds, and confidence-aware scoring.",
            },
            {
                "number": "Step 6",
                "title": "Analyze",
                "text": "Generate TF-IDF, cosine similarity, co-occurrence networks, and evidence views that feed directly into the dashboard.",
            },
        ],
        columns=3,
    )
    st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)
    
    if False:
        st.markdown("""
    <style>
    [data-testid="column"] {
        padding: 15px; /* 增加列內部的填充 */
    }
    .stMarkdown {
        margin-bottom: 20px; /* 增加每張卡片下方的距離 */
    }
    </style>
    """, unsafe_allow_html=True)

    render_card_grid(
        [
            {
                "title": "Data Preparation",
                "text": "We clean PDF-specific noise, remove repeated headers and boilerplate, then split the report into semantically cleaner chunks. After chunking, spaCy handles normalization, lemmatization, stopword removal, and POS filtering.",
                "highlight": True,
            },
            {
                "title": "Analysis",
                "text": "We aggregate grouped text with TF-IDF to surface distinctive vocabulary across Strategic Framing and UN SDGs. We also compare grouped language with cosine similarity.",
                "highlight": True,
            },
            {
                "title": "Labeling",
                "text": "Each chunk receives Strategic Framing and UN SDG labels using transparent rule-based logic. SDG assignment uses stronger thresholds, a dominance rule, and confidence scores to reduce noisy multi-labeling.",
            
            },
            {
                "title": "Validation",
                "text": "The pipeline includes an audit layer that checks duplicates, short chunks, repeated first-line patterns, and SDG label counts. This makes the output more reproducible and easier to defend during presentation.",
            },
        ],
        columns=2,
    )
    st.markdown("")

    st.markdown("### Methods by Outcome")
    render_card_grid(
        [
            {
                "title": "Interpretability",
                "text": "Representative chunks and SDG confidence scores support evidence-based explanation.",
                "kicker": "Outcome",
            },
            {
                "title": "Comparability",
                "text": "TF-IDF, overlap heatmaps, and cosine similarity show where categories align or differ in language.",
                "kicker": "Outcome",
            },
            {
                "title": "Presentation Value",
                "text": "Interactive views make the analysis easier to navigate quickly in a live demo.",
                "kicker": "Outcome",
                "highlight": True,
            },
        ],
        columns=3,
    )

elif page_mode == "Analysis":
    render_page_header(
        "The Evidence",
        "Analysis",
        "Switch between Strategic Framing and UN SDG to explore each theme end-to-end.",
    )

    render_metric_grid(
        [
            {"label": "Total Chunks", "value": len(df_chunks)},
            {"label": "Strategic Framing", "value": df_chunks["section_label"].nunique()},
            {
                "label": "UN SDG Themes",
                "value": df_chunks["sdg_labels"].str.split(",").explode().str.strip().pipe(lambda s: s[s != "unclassified"]).nunique(),
            },
        ],
        columns=3,
    )

    if False:
        st.markdown("""
        <style>
        [data-testid="stExpander"] summary p {
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        [data-testid="stExpander"] div[role="region"] {
            min-height: 250px;
        }
        [data-testid="stExpander"] {
            height: auto !important;
        }
        </style>
        """, unsafe_allow_html=True)

    if theme_mode == "Strategic Framing":
        # ---- 1. Distribution ----
        st.markdown("### Distribution")
        plot_distribution_bar(df_chunks, "section_label", "Chunk Count by Strategic Framing")

        # ---- 2. Per-section explorer tabs ----
        st.markdown("### Per-Section Analysis")
        tabs = st.tabs([SECTION_LABEL_MAP.get(x, x) for x in SECTION_ORDER])

        for tab, section in zip(tabs, SECTION_ORDER):
            with tab:
                df_terms = df_section_tfidf[df_section_tfidf["section_label"] == section].copy()
                count_section = len(df_chunks[df_chunks["section_label"] == section])
                st.subheader(f"{SECTION_LABEL_MAP.get(section, section)} ({count_section} chunks)")
                render_interpretation(
                    SECTION_LABEL_MAP.get(section, section),
                    SECTION_NARRATIVE.get(section, ""),
                )

                left, right = st.columns([1.2, 1])
                with left:
                    st.markdown("**Word Cloud**")
                    render_interactive_wordcloud(df_terms.head(80))
                with right:
                    st.markdown("**Top Keywords**")
                    plot_top_terms(
                        df_terms,
                        title=f"Top TF-IDF Terms: {SECTION_LABEL_MAP.get(section, section)}",
                        top_k=top_k,
                    )

        # ---- 3. Top 2 synthesis ----
        st.markdown("### Top 2 Synthesis")
        section_counts = df_chunks["section_label"].value_counts()
        top2_sections = section_counts.head(2).index.tolist()

        syn_col1, syn_col2 = st.columns(2)
        for col, section in zip([syn_col1, syn_col2], top2_sections):
            with col:
                top_terms = (
                    df_section_tfidf[df_section_tfidf["section_label"] == section]
                    .sort_values("tfidf_score", ascending=False)
                    .head(3)["term"]
                    .tolist()
                )
                label = SECTION_LABEL_MAP.get(section, section)
                count = int(section_counts.get(section, 0))
                narrative = SECTION_NARRATIVE.get(section, "")
                with st.expander(f"**{label}** — {', '.join(top_terms)} ({count} chunks)", expanded=True):
                    if narrative:
                        st.markdown(narrative)

        # ---- 4. Validation ----
        st.markdown("---")
        st.markdown("### Validation")
        st.caption("Cross-check the results: do sections share the same buzzwords? Do they speak the same language overall?")

        val_col1, val_col2 = st.columns(2)
        with val_col1:
            st.markdown("**Keyword Overlap**")
            st.caption("How many of each section's top-20 keywords also appear in another section's top-20.")
            heatmap_section = build_overlap_heatmap(
                df_section_tfidf, group_col="section_label", ordered_groups=SECTION_ORDER, top_n=20
            )
            plot_heatmap(heatmap_section, "Top-Term Overlap")
        with val_col2:
            st.markdown("**Cosine Similarity**")
            st.caption("Overall vocabulary similarity using full TF-IDF vectors — not just top words.")
            plot_heatmap(df_section_similarity, "Cosine Similarity")

        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            st.markdown("**Most similar pairs**")
            st.dataframe(
                summarize_similarity_pairs(df_section_similarity, top_n=3, ascending=False),
                use_container_width=True, hide_index=True,
            )
        with sim_col2:
            st.markdown("**Most distinct pairs**")
            st.dataframe(
                summarize_similarity_pairs(df_section_similarity, top_n=3, ascending=True),
                use_container_width=True, hide_index=True,
            )

        st.markdown("**Example Chunks**")
        render_interpretation(
            "Evidence",
            "Representative chunks ranked by overlap with top section terms, then by SDG confidence.",
        )
        render_chunk_cards(filtered_chunks, df_section_tfidf, max_chunks=2)

    else:  # UN SDG Themes
        # ---- 1. Distribution ----
        st.markdown("### Distribution")
        df_chunks_sdg_exploded = explode_sdg_chunks(df_chunks)
        plot_distribution_bar(df_chunks_sdg_exploded, "sdg_labels", "Chunk Count by SDG Theme")

        # ---- 2. Per-SDG explorer tabs ----
        st.markdown("### Per-SDG Analysis")
        tabs = st.tabs([SDG_LABEL_MAP.get(x, x) for x in SDG_ORDER])

        for tab, sdg in zip(tabs, SDG_ORDER):
            with tab:
                df_terms = df_sdg_tfidf[df_sdg_tfidf["sdg_labels"] == sdg].copy()
                count_sdg = len(filter_chunks_by_sdg(df_chunks, sdg))
                st.subheader(f"{SDG_LABEL_MAP.get(sdg, sdg)} ({count_sdg} chunks)")
                render_interpretation(
                    SDG_LABEL_MAP.get(sdg, sdg),
                    SDG_NARRATIVE.get(sdg, ""),
                )

                left, right = st.columns([1.2, 1])
                with left:
                    st.markdown("**Word Cloud**")
                    render_interactive_wordcloud(df_terms.head(80))
                with right:
                    st.markdown("**Top Keywords**")
                    plot_top_terms(
                        df_terms,
                        title=f"Top TF-IDF Terms: {SDG_LABEL_MAP.get(sdg, sdg)}",
                        top_k=top_k,
                    )

        # ---- 3. Top 2 synthesis ----
        st.markdown("### Top 2 Synthesis")
        sdg_counts = df_chunks_sdg_exploded["sdg_labels"].value_counts()
        top2_sdgs = sdg_counts.head(2).index.tolist()

        syn_col1, syn_col2 = st.columns(2)
        for col, sdg in zip([syn_col1, syn_col2], top2_sdgs):
            with col:
                top_terms = (
                    df_sdg_tfidf[df_sdg_tfidf["sdg_labels"] == sdg]
                    .sort_values("tfidf_score", ascending=False)
                    .head(3)["term"]
                    .tolist()
                )
                label = SDG_LABEL_MAP.get(sdg, sdg)
                count = int(sdg_counts.get(sdg, 0))
                narrative = SDG_NARRATIVE.get(sdg, "")
                with st.expander(f"**{label}** — {', '.join(top_terms)} ({count} chunks)", expanded=True):
                    if narrative:
                        st.markdown(narrative)

        # ---- 4. Validation ----
        st.markdown("---")
        st.markdown("### Validation")
        st.caption("Cross-check the results: do SDG themes share the same buzzwords? Do they speak the same language overall?")

        val_col1, val_col2 = st.columns(2)
        with val_col1:
            st.markdown("**Keyword Overlap**")
            st.caption("How many of each SDG's top-20 keywords also appear in another SDG's top-20.")
            heatmap_sdg = build_overlap_heatmap(
                df_sdg_tfidf, group_col="sdg_labels", ordered_groups=SDG_ORDER, top_n=20
            )
            plot_heatmap(heatmap_sdg, "Top-Term Overlap")
        with val_col2:
            st.markdown("**Cosine Similarity**")
            st.caption("Overall vocabulary similarity using full TF-IDF vectors — not just top words.")
            plot_heatmap(df_sdg_similarity, "Cosine Similarity")

        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            st.markdown("**Most similar pairs**")
            st.dataframe(
                summarize_similarity_pairs(df_sdg_similarity, top_n=3, ascending=False),
                use_container_width=True, hide_index=True,
            )
        with sim_col2:
            st.markdown("**Most distinct pairs**")
            st.dataframe(
                summarize_similarity_pairs(df_sdg_similarity, top_n=3, ascending=True),
                use_container_width=True, hide_index=True,
            )

        st.markdown("**Example Chunks**")
        render_interpretation(
            "Evidence",
            "Representative chunks ranked by overlap with top SDG terms, then by SDG confidence.",
        )
        render_chunk_cards(filtered_chunks, df_sdg_tfidf, max_chunks=2)



# =========================
# Footer
# =========================
st.markdown("---")
st.caption("Built from TSMC's 2024 Sustainability Report using automated text-mining and TF-IDF analysis.")
