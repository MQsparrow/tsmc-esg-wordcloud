from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PIC_DIR = PROJECT_ROOT / "pic"

VALID_YEARS = ("2022", "2023", "2024")

SECTION_ALIASES = {
    "environment": "environment",
    "environmental": "environment",
    "climate section": "environment",
    "talent": "talent",
    "employee": "talent",
    "employees": "talent",
    "people": "talent",
    "workforce": "talent",
    "supply": "supply_chain",
    "supply chain": "supply_chain",
    "supplier": "supply_chain",
    "suppliers": "supply_chain",
    "social": "social",
    "society": "social",
    "governance": "governance",
    "risk": "governance",
}

SDG_ALIASES = {
    "sdg3": "SDG3_health",
    "sdg 3": "SDG3_health",
    "health": "SDG3_health",
    "sdg4": "SDG4_education",
    "sdg 4": "SDG4_education",
    "education": "SDG4_education",
    "sdg6": "SDG6_water",
    "sdg 6": "SDG6_water",
    "water": "SDG6_water",
    "sdg7": "SDG7_energy",
    "sdg 7": "SDG7_energy",
    "energy": "SDG7_energy",
    "sdg8": "SDG8_labor",
    "sdg 8": "SDG8_labor",
    "labor": "SDG8_labor",
    "labour": "SDG8_labor",
    "sdg9": "SDG9_innovation",
    "sdg 9": "SDG9_innovation",
    "innovation": "SDG9_innovation",
    "sdg12": "SDG12_consumption",
    "sdg 12": "SDG12_consumption",
    "consumption": "SDG12_consumption",
    "sdg13": "SDG13_climate",
    "sdg 13": "SDG13_climate",
    "climate": "SDG13_climate",
    "sdg17": "SDG17_partnership",
    "sdg 17": "SDG17_partnership",
    "partnership": "SDG17_partnership",
}

BERT_ASSETS = [
    {
        "name": "BERT SDG embedding PCA",
        "path": PIC_DIR / "bert_sdg_embedding_pca.png",
        "use": "Shows whether SDG-labeled chunks occupy related semantic regions.",
    },
    {
        "name": "BERT SDG similarity heatmap",
        "path": PIC_DIR / "bert_sdg_similarity_heatmap.png",
        "use": "Shows semantic similarity between SDG groups.",
    },
    {
        "name": "BERT section embedding PCA",
        "path": PIC_DIR / "bert_section_embedding_pca.png",
        "use": "Shows section-level semantic structure.",
    },
    {
        "name": "BERT section similarity heatmap",
        "path": PIC_DIR / "bert_section_similarity_heatmap.png",
        "use": "Shows semantic similarity between strategic framing sections.",
    },
]


class AgentToolError(ValueError):
    """Raised when a project analysis tool cannot produce a valid result."""


def detect_year(query: str, default: str = "2024") -> str:
    for year in VALID_YEARS:
        if year in str(query):
            return year
    return default


def detect_group(query: str, default_group_type: str = "sdg") -> tuple[str, str]:
    text = str(query).lower()
    for alias, label in SDG_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return "sdg", label
    for alias, label in SECTION_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return "section", label
    if default_group_type == "section":
        return "section", "environment"
    return "sdg", "SDG13_climate"


def normalize_group_name(group_type: str, group_name: str) -> str:
    value = str(group_name).strip()
    lookup = value.lower().replace("_", " ")
    compact_lookup = lookup.replace(" ", "")

    if group_type == "sdg":
        if value in set(SDG_ALIASES.values()):
            return value
        for alias, label in SDG_ALIASES.items():
            alias_lookup = alias.lower().replace(" ", "")
            if compact_lookup == alias_lookup or compact_lookup in label.lower().replace("_", ""):
                return label
    if group_type == "section":
        if value in set(SECTION_ALIASES.values()):
            return value
        for alias, label in SECTION_ALIASES.items():
            if lookup == alias.lower() or lookup == label.lower().replace("_", " "):
                return label
    return value


def detect_groups(query: str) -> tuple[str, list[str]]:
    text = str(query).lower()
    sdg_matches = []
    for alias, label in SDG_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text) and label not in sdg_matches:
            sdg_matches.append(label)
    if len(sdg_matches) >= 2:
        return "sdg", sdg_matches[:2]

    section_matches = []
    for alias, label in SECTION_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", text) and label not in section_matches:
            section_matches.append(label)
    if len(section_matches) >= 2:
        return "section", section_matches[:2]

    group_type, group_name = detect_group(query)
    return group_type, [group_name]


def load_output(year: str, filename: str) -> pd.DataFrame:
    path = OUTPUTS_DIR / str(year) / filename
    if not path.exists():
        fallback = OUTPUTS_DIR / filename
        path = fallback if fallback.exists() else path
    if not path.exists():
        raise AgentToolError(f"Missing required output file: {path}")
    return pd.read_csv(path)


def _label_column(group_type: str) -> str:
    if group_type == "sdg":
        return "sdg_labels"
    if group_type == "section":
        return "section_label"
    raise AgentToolError(f"Unsupported group type: {group_type}")


def _tfidf_filename(group_type: str) -> str:
    if group_type == "sdg":
        return "tfidf_by_sdg.csv"
    if group_type == "section":
        return "tfidf_by_section.csv"
    raise AgentToolError(f"Unsupported group type: {group_type}")


def _similarity_filename(group_type: str) -> str:
    if group_type == "sdg":
        return "similarity_by_sdg.csv"
    if group_type == "section":
        return "similarity_by_section.csv"
    raise AgentToolError(f"Unsupported group type: {group_type}")


def _chunk_matches(row_value: Any, group_name: str, group_type: str) -> bool:
    if group_type == "sdg":
        labels = [part.strip() for part in str(row_value).split(",")]
        return group_name in labels
    return str(row_value) == group_name


def get_top_terms(year: str, group_type: str, group_name: str, top_n: int = 10) -> dict[str, Any]:
    group_name = normalize_group_name(group_type, group_name)
    df = load_output(year, _tfidf_filename(group_type))
    label_col = _label_column(group_type)
    if label_col not in df.columns:
        raise AgentToolError(f"Column {label_col} not found in TF-IDF output.")
    rows = df[df[label_col] == group_name].sort_values("tfidf_score", ascending=False).head(top_n)
    return {
        "tool": "TF-IDF Keyword Tool",
        "year": year,
        "group_type": group_type,
        "group_name": group_name,
        "rows": rows.reset_index(drop=True),
        "summary": ", ".join(rows["term"].astype(str).head(6).tolist()),
    }


def get_representative_chunks(year: str, group_type: str, group_name: str, top_n: int = 3) -> dict[str, Any]:
    group_name = normalize_group_name(group_type, group_name)
    df = load_output(year, "chunks_processed.csv")
    label_col = _label_column(group_type)
    if label_col not in df.columns:
        raise AgentToolError(f"Column {label_col} not found in chunk output.")
    mask = df[label_col].map(lambda value: _chunk_matches(value, group_name, group_type))
    rows = df[mask].copy()
    rows["excerpt"] = rows["clean_text"].astype(str).str.slice(0, 420)
    display_cols = [col for col in ["chunk_id", "section_label", "sdg_labels", "excerpt"] if col in rows.columns]
    rows = rows[display_cols].head(top_n)
    return {
        "tool": "Evidence Chunk Tool",
        "year": year,
        "group_type": group_type,
        "group_name": group_name,
        "rows": rows.reset_index(drop=True),
        "summary": f"Retrieved {len(rows)} representative chunks for {group_name}.",
    }


def compare_groups(year: str, group_type: str, group_a: str, group_b: str, top_n: int = 8) -> dict[str, Any]:
    group_a = normalize_group_name(group_type, group_a)
    group_b = normalize_group_name(group_type, group_b)
    df = load_output(year, _tfidf_filename(group_type))
    label_col = _label_column(group_type)
    a_terms = df[df[label_col] == group_a].sort_values("tfidf_score", ascending=False).head(top_n)
    b_terms = df[df[label_col] == group_b].sort_values("tfidf_score", ascending=False).head(top_n)
    rows = pd.concat(
        [
            a_terms.assign(group=group_a),
            b_terms.assign(group=group_b),
        ],
        ignore_index=True,
    )
    shared = sorted(set(a_terms["term"].astype(str)) & set(b_terms["term"].astype(str)))
    return {
        "tool": "Comparison Tool",
        "year": year,
        "group_type": group_type,
        "group_name": f"{group_a} vs {group_b}",
        "rows": rows[["group", "term", "tfidf_score"]].reset_index(drop=True),
        "summary": f"Shared top terms: {', '.join(shared) if shared else 'none among the displayed top terms'}.",
    }


def get_similarity_neighbors(year: str, group_type: str, group_name: str, top_n: int = 5) -> dict[str, Any]:
    group_name = normalize_group_name(group_type, group_name)
    path = OUTPUTS_DIR / str(year) / _similarity_filename(group_type)
    if not path.exists():
        path = OUTPUTS_DIR / _similarity_filename(group_type)
    if not path.exists():
        raise AgentToolError(f"Missing required output file: {path}")
    sim = pd.read_csv(path, index_col=0)
    if group_name not in sim.index:
        raise AgentToolError(f"{group_name} not found in similarity matrix.")
    rows = (
        sim.loc[group_name]
        .drop(labels=[group_name], errors="ignore")
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    rows.columns = ["related_group", "similarity"]
    return {
        "tool": "Similarity Tool",
        "year": year,
        "group_type": group_type,
        "group_name": group_name,
        "rows": rows,
        "summary": ", ".join(f"{r.related_group} ({r.similarity:.2f})" for r in rows.itertuples()),
    }


def get_bert_assets() -> dict[str, Any]:
    rows = pd.DataFrame(
        [
            {
                "asset": asset["name"],
                "path": str(asset["path"].relative_to(PROJECT_ROOT)),
                "available": asset["path"].exists(),
                "use": asset["use"],
            }
            for asset in BERT_ASSETS
        ]
    )
    return {
        "tool": "BERT Semantic Tool",
        "year": "multi-year",
        "group_type": "semantic",
        "group_name": "BERT assets",
        "rows": rows,
        "summary": "BERT-style figures connect the agent demo to semantic similarity and embedding experiments.",
    }


def run_fallback_agent(query: str, year: str = "2024") -> dict[str, Any]:
    year = detect_year(query, default=year)
    query_lc = str(query).lower()
    group_type, groups = detect_groups(query)
    tool_results: list[dict[str, Any]] = []

    if "bert" in query_lc or "semantic" in query_lc or "embedding" in query_lc:
        tool_results.append(get_bert_assets())

    if "compare" in query_lc or " vs " in query_lc or "difference" in query_lc:
        if len(groups) < 2:
            groups = ["SDG12_consumption", "SDG13_climate"] if group_type == "sdg" else ["environment", "talent"]
        tool_results.append(compare_groups(year, group_type, groups[0], groups[1]))
        tool_results.append(get_similarity_neighbors(year, group_type, groups[0]))
    elif "similar" in query_lc or "similarity" in query_lc or "related" in query_lc:
        tool_results.append(get_similarity_neighbors(year, group_type, groups[0]))
    elif "evidence" in query_lc or "chunk" in query_lc or "example" in query_lc:
        tool_results.append(get_representative_chunks(year, group_type, groups[0]))
    else:
        tool_results.append(get_top_terms(year, group_type, groups[0]))
        tool_results.append(get_representative_chunks(year, group_type, groups[0]))

    answer = build_structured_answer(query, tool_results)
    return {
        "mode": "Fallback demo mode",
        "year": year,
        "answer": answer,
        "tool_results": tool_results,
    }


def build_structured_answer(query: str, tool_results: list[dict[str, Any]]) -> str:
    lines = [
        f"Question interpreted: {query}",
        "",
        "Agent answer:",
    ]
    for result in tool_results:
        lines.append(f"- {result['tool']} for {result['group_name']}: {result['summary']}")
    lines.append("")
    lines.append(
        "Interpretation: the agent uses project-specific tools instead of guessing from memory, "
        "so the response is grounded in existing TSMC ESG analysis outputs."
    )
    return "\n".join(lines)
