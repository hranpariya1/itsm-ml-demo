# itsm-ml-demo

A small, beginner-friendly machine learning example for an **ITSM (IT Service
Management) tool**. It trains a model that reads the free-text description of
a support ticket and predicts which **category** the ticket belongs to:

- `Network` (VPN, Wi-Fi, DNS, ...)
- `Email` (Outlook, mailboxes, calendar, ...)
- `Account` (passwords, account locks, access requests, ...)
- `Hardware` (laptops, monitors, headsets, printers, ...)
- `Software` (Office apps, browsers, installs, ...)

Auto-categorizing tickets is one of the most common ML use cases in real ITSM
tools (ServiceNow, Jira Service Management, BMC Remedy, etc.) because it lets
tickets be routed to the right support team automatically.

## How it works

This is a classic **text classification** pipeline using
[scikit-learn](https://scikit-learn.org/):

```
ticket description  ->  TF-IDF features  ->  Logistic Regression  ->  category
```

- **TF-IDF** turns words into numbers, giving more weight to words that are
  important for a particular ticket (e.g. *VPN*, *password*, *printer*).
- **Logistic Regression** is a simple, fast classifier that learns which words
  point to which category.

## Project layout

```
itsm-ml-demo/
├── data/
│   └── tickets.csv          # small synthetic dataset of example tickets
├── train.py                 # trains the model and saves it to model/
├── predict.py               # loads the saved model and predicts new tickets
├── requirements.txt         # Python dependencies
└── README.md
```

## Setup

Requires Python 3.9+.

```bash
# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## 1. Train the model

```bash
python train.py
```

This will:

1. Load `data/tickets.csv`.
2. Split the data into a training set and a test set.
3. Train a TF-IDF + Logistic Regression pipeline.
4. Print a classification report (precision / recall / F1) on the test set.
5. Save the trained model to `model/ticket_classifier.joblib`.

## 2. Predict the category of new tickets

```bash
# predict for one description you pass in
python predict.py "My laptop will not turn on after the update"

# or run with no arguments to see a few built-in demo predictions
python predict.py
```

Example output:

```
Predictions:
  [Hardware] (62%)  My laptop will not turn on after the update
  [Network ] (58%)  Cannot access the company VPN from my hotel room
  [Account ] (71%)  Please reset my password for the HR portal
```

> Note: the bundled dataset is intentionally tiny (a few dozen tickets) so the
> example trains in seconds and is easy to read. With a real ITSM dataset of
> thousands of tickets, accuracy and confidence scores will be much higher.

## Ideas to extend this demo

Once you are comfortable with the basics, try:

- Adding more training tickets to `data/tickets.csv` and retraining.
- Predicting **priority** (Low/Medium/High) instead of category.
- Trying a different model, e.g. `MultinomialNB` or `LinearSVC` from
  scikit-learn, and comparing the classification reports.
- Saving predictions back to a CSV for bulk triage of historical tickets.
- Wrapping `predict.py` in a small Flask or FastAPI web service so an ITSM
  tool can call it as an API.
