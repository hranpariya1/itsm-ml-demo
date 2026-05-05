"""
ITSM Ticket Classifier - a beginner-friendly machine learning example.

Use case
--------
In an IT Service Management (ITSM) tool such as ServiceNow, Jira Service
Management, or BMC Remedy, end users open tickets describing their problem
in free text. Routing those tickets to the right team (Network, Hardware,
Software, Account, Access, ...) is usually done manually.

This script trains a simple text-classification model that learns to predict
the *category* of a ticket from its short description, so it can be auto
routed to the correct support queue.

Pipeline
--------
1. Load a small CSV of historical tickets (text + category label).
2. Convert the text into numeric features using TF-IDF.
3. Train a Logistic Regression classifier.
4. Evaluate it on a held-out test set (accuracy + per-class report).
5. Predict the category of a few brand-new example tickets.

Run
---
    pip install -r requirements.txt
    python ticket_classifier.py
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path(__file__).parent / "data" / "tickets.csv"
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    """Load the tickets CSV into a DataFrame."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} tickets from {path}")
    print("Categories:", sorted(df["category"].unique()))
    return df


def build_model() -> Pipeline:
    """Build a TF-IDF + Logistic Regression pipeline.

    TF-IDF turns each ticket description into a vector of word importances.
    Logistic Regression is a simple, fast, interpretable classifier that
    works well for short-text problems like this one.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )


def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    """Split the data, train the model, and print evaluation metrics."""
    X = df["text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n=== Evaluation on held-out test set ===")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return model


def predict_examples(model: Pipeline) -> None:
    """Use the trained model to classify a few new example tickets."""
    new_tickets = [
        "I forgot my password and cannot log in this morning",
        "The office wifi keeps disconnecting from my laptop",
        "Need access to the project management tool please",
        "Outlook will not open after the latest windows update",
        "My monitor is showing a black screen randomly",
    ]

    print("\n=== Predictions on new tickets ===")
    predictions = model.predict(new_tickets)
    for ticket, category in zip(new_tickets, predictions):
        print(f"[{category:8s}] {ticket}")


def main() -> None:
    df = load_data(DATA_PATH)
    model = train_and_evaluate(df)
    predict_examples(model)


if __name__ == "__main__":
    main()
