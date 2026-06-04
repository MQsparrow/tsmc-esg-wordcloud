from __future__ import annotations

from collections import Counter
from typing import Any

from .state import ESGState


SECTION_TO_ESG = {
    "environment": "Environmental",
    "talent": "Social",
    "social": "Social",
    "supply_chain": "Social",
    "governance": "Governance",
}

RULES = {
    "Environmental": [
        "carbon", "climate", "emission", "energy", "water", "waste", "renewable", "biodiversity",
        "greenhouse", "net zero", "recycling", "pollution",
    ],
    "Social": [
        "employee", "talent", "safety", "health", "training", "diversity", "human rights",
        "supplier", "community", "education", "labor", "workplace",
    ],
    "Governance": [
        "board", "governance", "risk", "compliance", "ethics", "audit", "integrity",
        "privacy", "security", "regulation", "anticorruption",
    ],
}


def classify_text(text: str, section_label: str = "") -> tuple[str, float, str]:
    section = str(section_label or "").strip().lower()
    if section in SECTION_TO_ESG:
        label = SECTION_TO_ESG[section]
        return label, 0.9, f"Mapped from section label: {section}"

    lowered = str(text).lower()
    scores: dict[str, int] = {}
    for label, terms in RULES.items():
        scores[label] = sum(1 for term in terms if term in lowered)
    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score <= 0:
        return "Other", 0.35, "No strong ESG keyword match"
    total = sum(scores.values()) or 1
    confidence = min(0.95, 0.45 + best_score / total * 0.45)
    return best_label, round(confidence, 2), f"Matched {best_score} {best_label.lower()} keyword signals"


def classification_agent(state: ESGState) -> ESGState:
    records = state.get("chunk_records", [])
    chunks = state.get("chunks", [])
    if not records:
        records = [{"chunk_id": idx, "text": text, "section_label": ""} for idx, text in enumerate(chunks)]

    classifications: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records:
        text = str(record.get("text", ""))
        label, confidence, reason = classify_text(text, str(record.get("section_label", "")))
        counts[label] += 1
        classifications.append(
            {
                "chunk_id": record.get("chunk_id"),
                "label": label,
                "confidence": confidence,
                "reason": reason,
                "section_label": record.get("section_label", ""),
                "sdg_labels": record.get("sdg_labels", ""),
                "text": text[:900],
            }
        )
    return {"esg_classifications": classifications, "esg_counts": dict(counts)}
