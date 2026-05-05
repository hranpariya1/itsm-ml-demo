"""
Use the trained ITSM ticket classifier to predict the category of new tickets.

Usage:
    # Predict for one description passed on the command line
    python predict.py "My laptop will not turn on after the update"

    # Or run with no arguments to see a few built-in demo predictions
    python predict.py
"""

import sys
from pathlib import Path
from typing import List, Tuple

import joblib
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "ticket_classifier.joblib"


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Please run `python train.py` first to train and save the model."
        )
    return joblib.load(MODEL_PATH)


def predict(
    model: Pipeline, descriptions: List[str]
) -> List[Tuple[str, str, float]]:
    """Return a list of (description, predicted_category, confidence) tuples."""
    predictions = model.predict(descriptions)
    # predict_proba gives the model's confidence for each class
    probabilities = model.predict_proba(descriptions)
    classes = model.classes_

    results = []
    for desc, pred, probs in zip(descriptions, predictions, probabilities):
        confidence = float(probs[list(classes).index(pred)])
        results.append((desc, pred, confidence))
    return results


def main() -> None:
    model = load_model()

    if len(sys.argv) > 1:
        # Treat all CLI args (joined) as a single ticket description
        descriptions = [" ".join(sys.argv[1:])]
    else:
        # Demo examples that were not part of the training data
        descriptions = [
            "My laptop will not turn on after the update",
            "Cannot access the company VPN from my hotel room",
            "Please reset my password for the HR portal",
            "Outlook is stuck on loading profile",
            "Need Microsoft Visio installed for diagramming",
        ]

    results = predict(model, descriptions)
    print("Predictions:")
    for desc, category, confidence in results:
        print(f"  [{category:<8}] ({confidence:.0%})  {desc}")


if __name__ == "__main__":
    main()
