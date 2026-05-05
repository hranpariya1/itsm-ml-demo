"""
Train a simple ITSM ticket classifier.

Given the free-text description of an IT service ticket, this script trains a
machine learning model that predicts which category the ticket belongs to
(Network, Email, Account, Hardware, Software).

This is a classic text-classification pipeline:
    raw text  ->  TF-IDF features  ->  Logistic Regression  ->  category

The trained pipeline is saved to disk with joblib so it can be reused by
predict.py without retraining.

Usage:
    python train.py
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "tickets.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "ticket_classifier.joblib"


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the ITSM tickets dataset from CSV."""
    df = pd.read_csv(csv_path)
    # Drop any rows with missing values just in case
    df = df.dropna(subset=["description", "category"])
    return df


def build_pipeline() -> Pipeline:
    """Build the text-classification pipeline.

    TfidfVectorizer turns the ticket description text into numeric features
    based on word importance, and LogisticRegression learns to map those
    features to a category label.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),  # use single words and word pairs
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def main() -> None:
    print(f"Loading data from {DATA_PATH} ...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} tickets across {df['category'].nunique()} categories.")
    print(df["category"].value_counts().to_string())

    X = df["description"]
    y = df["category"]

    # Hold out 25% of the data for evaluation.
    # `stratify=y` keeps the same class balance in train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    print("\nTraining model ...")
    pipeline.fit(X_train, y_train)

    print("\nEvaluation on held-out test set:")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
