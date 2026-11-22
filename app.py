"""Streamlit interface for the MIA5100 Group 8 text-classification demo."""

from pathlib import Path

import pandas as pd
import streamlit as st

from model_utils import load_bundle, predict_statement


MODEL_PATH = Path(__file__).parent / "artifacts" / "group8_lr_tfidf.joblib"

st.set_page_config(
    page_title="MIA5100 Text Classification Demo",
    page_icon="🧭",
    layout="centered",
)

st.title("Mental-Health Statement Classification")
st.caption("MIA5100 Group 8 · Educational machine-learning demonstration")
st.warning(
    "This demo predicts labels from the project dataset. It is not a diagnostic, "
    "screening, treatment, or emergency-response system. Model scores are not "
    "clinical probabilities."
)


@st.cache_resource
def get_bundle():
    return load_bundle(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("The trained model artifact is unavailable.")
    st.stop()

try:
    bundle = get_bundle()
except Exception as exc:
    st.error(f"The model could not be loaded: {exc}")
    st.stop()

statement = st.text_area(
    "Statement to classify",
    height=150,
    placeholder="Enter a fictional statement for the research demonstration…",
    help="The application does not intentionally save submitted text.",
)

if st.button("Analyse statement", type="primary", use_container_width=True):
    try:
        result = predict_statement(bundle, statement)
    except ValueError as exc:
        st.info(str(exc))
    else:
        st.subheader("Model output")
        st.metric("Predicted dataset category", result.predicted_class)

        scores = pd.DataFrame(result.top_scores, columns=["Category", "Model score"])
        scores["Model score"] = scores["Model score"].round(3)
        st.dataframe(scores, hide_index=True, use_container_width=True)
        st.bar_chart(scores.set_index("Category"), horizontal=True)

        if result.needs_review:
            st.warning(
                "The leading scores are low or close. This result requires human "
                "review and should not drive a decision."
            )
        else:
            st.info(
                "The leading score is separated from the alternatives, but the "
                "result still requires human interpretation."
            )

        if result.influential_terms:
            with st.expander("Influential model features"):
                st.write(
                    "Positive linear contributions for the predicted class. These "
                    "features are not clinical explanations or causes."
                )
                st.write(", ".join(term for term, _ in result.influential_terms))

        if result.predicted_class == "Suicidal":
            st.error(
                "A ‘Suicidal’ dataset label is not a risk assessment. The model "
                "also produces false negatives and false positives. Real safety "
                "concerns require qualified human assessment."
            )

with st.expander("How it works"):
    st.markdown(
        """
1. Conservative text cleaning matches the project notebook.
2. The fitted TF-IDF vectorizer converts words and two-word phrases to features.
3. Logistic Regression scores the seven dataset labels.
4. The interface shows alternatives and an uncertainty flag.

The labels are Anxiety, Bipolar, Depression, Normal, Personality disorder, Stress,
and Suicidal. They are dataset categories, not independently verified diagnoses.
"""
    )

with st.expander("Limitations and privacy"):
    st.markdown(
        """
- Submitted text is not intentionally written to a file or database.
- Hosting and network logs are outside the model and require separate review.
- Performance on one dataset may not generalize to new platforms or populations.
- Short, ambiguous, figurative, multilingual, or unfamiliar text may be unreliable.
- Do not use this system for diagnosis, treatment, surveillance, or intervention.
"""
    )

metadata = bundle.get("metadata", {})
with st.sidebar:
    st.header("Model")
    st.write("**Classifier:** Logistic Regression")
    st.write("**Features:** TF-IDF unigrams and bigrams")
    if metadata.get("test_accuracy") is not None:
        st.write(f"**Artifact holdout accuracy:** {metadata['test_accuracy']:.3f}")
    if metadata.get("test_macro_f1") is not None:
        st.write(f"**Artifact holdout macro F1:** {metadata['test_macro_f1']:.3f}")
