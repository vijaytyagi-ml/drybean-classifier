"""
Dry Bean Classification - ML Assignment 2 (M.Tech DSE)
======================================================

Trains and evaluates five classifiers on the UCI Dry Bean dataset:
    1. Logistic Regression
    2. Decision Tree
    3. k-Nearest Neighbours
    4. Gaussian Naive Bayes
    5. Support Vector Machine (RBF kernel)

For every model the following metrics are computed on a held-out test split:
Accuracy, AUC (one-vs-rest, macro), Precision, Recall, F1 (macro) and MCC.

Outputs (written next to this file / project root):
    model/<model_key>.joblib        fitted sklearn pipelines
    model/label_encoder.joblib      class-name <-> integer mapping
    model/metrics_summary.csv       comparison table
    model/best_params.json          hyper-parameters chosen by grid search
    test_data.csv                   held-out test split (features + Class)

Run from the project root:
    python model/train_models.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SEED = 2611                 # fixed seed -> reproducible split and models
TEST_FRACTION = 0.20
TARGET = "Class"

MODEL_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODEL_DIR.parent
DATA_FILE = ROOT_DIR / "data" / "dry_bean_full.csv"

FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
    "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
    "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2",
    "ShapeFactor3", "ShapeFactor4",
]

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "kNN",
    "naive_bayes": "Naive Bayes",
    "svm": "Support Vector Machine",
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)   # 68 exact duplicates in UCI file
    df[TARGET] = df[TARGET].str.strip().str.upper()
    print(f"Loaded {n_raw} rows -> {len(df)} after dropping duplicates")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    return df


# --------------------------------------------------------------------------
# Model definitions (pipeline + small hyper-parameter grid)
# --------------------------------------------------------------------------
def model_search_spaces() -> dict[str, tuple[Pipeline, dict]]:
    return {
        "logistic_regression": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000, random_state=SEED))]),
            {"clf__C": [0.1, 1.0, 10.0, 50.0]},
        ),
        "decision_tree": (
            Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
            {"clf__max_depth": [6, 8, 10, 14, None],
             "clf__min_samples_leaf": [1, 5, 10, 20],
             "clf__criterion": ["gini", "entropy"]},
        ),
        "knn": (
            Pipeline([("scale", StandardScaler()),
                      ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [5, 9, 13, 17, 21],
             "clf__weights": ["uniform", "distance"]},
        ),
        # Gaussian NB assumes normally distributed features; a Yeo-Johnson
        # power transform makes the skewed area/perimeter features more Gaussian.
        "naive_bayes": (
            Pipeline([("power", PowerTransformer(method="yeo-johnson")),
                      ("clf", GaussianNB())]),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
        ),
        "svm": (
            # SVC has no native probabilities (needed for AUC); calibrate its
            # decision function with Platt scaling via CalibratedClassifierCV.
            Pipeline([("scale", StandardScaler()),
                      ("clf", CalibratedClassifierCV(SVC(kernel="rbf", random_state=SEED),
                                                     method="sigmoid", cv=3, ensemble=False))]),
            {"clf__estimator__C": [1, 10, 30],
             "clf__estimator__gamma": ["scale", 0.05, 0.1]},
        ),
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def evaluate(y_true, y_pred, y_proba, n_classes: int) -> dict[str, float]:
    if n_classes == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro",
                            labels=list(range(n_classes)))
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    df = load_and_clean()
    X = df[FEATURES]
    encoder = LabelEncoder().fit(df[TARGET])
    y = encoder.transform(df[TARGET])
    n_classes = len(encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, stratify=y, random_state=SEED)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, classes: {list(encoder.classes_)}")

    # Persist the held-out split for the Streamlit app
    test_df = X_test.copy()
    test_df[TARGET] = encoder.inverse_transform(y_test)
    test_df.to_csv(ROOT_DIR / "test_data.csv", index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, best_params = [], {}

    for key, (pipe, grid) in model_search_spaces().items():
        t0 = time.time()
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        scores = evaluate(y_test, y_pred, y_proba, n_classes)

        rows.append({"ML Model Name": MODEL_NAMES[key], **scores,
                     "CV F1 (train)": search.best_score_})
        best_params[key] = {k.replace("clf__estimator__", "").replace("clf__", ""): v for k, v in search.best_params_.items()}
        joblib.dump(model, MODEL_DIR / f"{key}.joblib", compress=3)
        print(f"{MODEL_NAMES[key]:<24} done in {time.time() - t0:5.1f}s  "
              f"best={best_params[key]}  test acc={scores['Accuracy']:.4f}")

    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")
    summary = pd.DataFrame(rows).round(4)
    summary.to_csv(MODEL_DIR / "metrics_summary.csv", index=False)
    with open(MODEL_DIR / "best_params.json", "w") as fh:
        json.dump(best_params, fh, indent=2, default=str)

    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
