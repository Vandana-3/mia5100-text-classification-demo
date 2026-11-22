"""Inference helpers for the Streamlit demo."""

import html
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: str
    top_scores: list[tuple[str, float]]
    needs_review: bool
    influential_terms: list[tuple[str, float]]


def clean_statement(text: str) -> str:
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", " ", text)
    text = re.sub(r"(?i)\b(?:https?://|www\.)\S+", " [URL] ", text)
    text = re.sub(
        r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        " [EMAIL] ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def load_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    missing = {"model", "vectorizer", "metadata"}.difference(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing: {', '.join(sorted(missing))}")
    return bundle


def _influential_terms(vectorizer, model, transformed, predicted_class, limit=6):
    if not hasattr(model, "coef_") or not hasattr(vectorizer, "get_feature_names_out"):
        return []
    class_index = list(model.classes_).index(predicted_class)
    row = transformed.getrow(0)
    if row.nnz == 0:
        return []
    contributions = row.data * model.coef_[class_index, row.indices]
    positive = np.where(contributions > 0)[0]
    order = positive[np.argsort(contributions[positive])[::-1]][:limit]
    names = vectorizer.get_feature_names_out()
    return [(str(names[row.indices[i]]), float(contributions[i])) for i in order]


def predict_statement(
    bundle: dict[str, Any],
    text: str,
    uncertainty_threshold: float = 0.55,
    margin_threshold: float = 0.10,
) -> PredictionResult:
    cleaned = clean_statement(text)
    if not cleaned:
        raise ValueError("Enter a non-empty statement.")

    vectorizer = bundle["vectorizer"]
    model = bundle["model"]
    transformed = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(transformed)[0]
    ranked = np.argsort(probabilities)[::-1]
    classes = np.asarray(model.classes_, dtype=str)
    top_scores = [(str(classes[i]), float(probabilities[i])) for i in ranked[:3]]
    predicted = top_scores[0][0]
    margin = top_scores[0][1] - top_scores[1][1]

    return PredictionResult(
        predicted_class=predicted,
        top_scores=top_scores,
        needs_review=top_scores[0][1] < uncertainty_threshold or margin < margin_threshold,
        influential_terms=_influential_terms(
            vectorizer, model, transformed, predicted
        ),
    )
