# Model Card

## Intended use

Course demonstration of multiclass text classification and a human-review interface.

## Out of scope

Diagnosis, screening, treatment, surveillance, emergency response, automated
intervention, or decisions about an individual.

## Data and labels

The model uses seven labels supplied by the project dataset: Anxiety, Bipolar,
Depression, Normal, Personality disorder, Stress, and Suicidal. The labels were not
independently clinically verified.

## Reported performance

The team notebook's tuned Logistic Regression reported test accuracy 0.7513 and
macro F1 0.6986 on 10,203 held-out statements. Performance varied by class.
Suicidal recall was 0.69, so safety-relevant false negatives remain.

The deployed artifact is a reproducible local fit using the original public CSV and
the same model settings. Its holdout accuracy is 0.7531 and macro F1 is 0.7034.

## Limitations

One dataset and one random holdout split cannot establish clinical validity or broad
generalization. Results may change across sources, platforms, populations, languages,
and time periods. Scores are not calibrated clinical risk estimates.
