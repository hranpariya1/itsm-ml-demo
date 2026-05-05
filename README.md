# itsm-ml-demo

A beginner-friendly **Machine Learning example for an ITSM (IT Service
Management) tool**, written in Python with scikit-learn.

## What it does

In an ITSM tool (ServiceNow, Jira Service Management, BMC Remedy, etc.) users
open tickets describing their issue in free text. Sorting those tickets into
the right support queue (Network, Hardware, Software, Account, Access, ...)
is usually a manual job.

This demo trains a small text-classification model that learns to predict the
**category of a ticket from its short description**, so tickets can be auto
routed to the correct team.

## How it works

1. **Data** – `data/tickets.csv` contains 40 sample tickets, each labelled
   with one of 5 categories: `Account`, `Software`, `Hardware`, `Network`,
   `Access`.
2. **Features** – ticket text is converted to numeric vectors with a
   **TF-IDF** vectorizer (unigrams + bigrams, English stop-words removed).
3. **Model** – a **Logistic Regression** classifier is trained on 75% of the
   data and evaluated on the remaining 25%.
4. **Prediction** – the trained model is used to classify a few brand-new
   example tickets.

The whole pipeline lives in [`ticket_classifier.py`](ticket_classifier.py).

## Setup

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

## Run

```bash
python ticket_classifier.py
```

You should see something like:

```
Loaded 40 tickets from .../data/tickets.csv
Categories: ['Access', 'Account', 'Hardware', 'Network', 'Software']

=== Evaluation on held-out test set ===
Accuracy: 0.70
...

=== Predictions on new tickets ===
[Account ] I forgot my password and cannot log in this morning
[Hardware] The office wifi keeps disconnecting from my laptop
[Access  ] Need access to the project management tool please
[Software] Outlook will not open after the latest windows update
[Hardware] My monitor is showing a black screen randomly
```

Exact numbers will vary slightly because the dataset is intentionally tiny —
this is a learning example, not a production model.

## Things to try next

A great way to learn more is to extend this demo:

- Add more rows (and more categories) to `data/tickets.csv` and rerun.
- Swap `LogisticRegression` for `RandomForestClassifier` or
  `LinearSVC` and compare accuracy.
- Add a second target column such as `priority` (Low / Medium / High) and
  train a second model on it.
- Save the trained model with `joblib.dump(...)` and load it from a small
  Flask / FastAPI endpoint to simulate an ITSM auto-routing service.