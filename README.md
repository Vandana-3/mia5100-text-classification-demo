# MIA5100 Text Classification Demo

Streamlit demonstration of Group 8's TF-IDF and Logistic Regression text classifier.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The fitted model bundle must exist at:

```text
artifacts/group8_lr_tfidf.joblib
```

## Scope

The app predicts one of seven labels supplied by the course-project dataset and
shows the top model scores, an uncertainty flag, and influential linear features.

It is an educational demonstration—not a diagnostic, screening, treatment, or
emergency-response system. Dataset labels were not independently clinically verified,
and results may not generalize beyond the evaluated data.

## Model

- Conservative, meaning-preserving text cleaning
- TF-IDF with unigrams and bigrams
- Tuned Logistic Regression
- Team notebook test accuracy: 0.7513
- Team notebook test macro F1: 0.6986

The deployed artifact is a local reproduction trained from the original public Kaggle
CSV with the same preprocessing, split seed, TF-IDF settings, and tuned Logistic
Regression parameters. Its holdout results are stored in the artifact metadata and
shown in the app sidebar.

The dataset is not included in this repository. Source: [Sentiment Analysis for
Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health).
