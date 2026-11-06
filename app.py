"""
BeanSense - Dry Bean Variety Classifier (Streamlit front-end)
ML Assignment 2 - M.Tech DSE

Upload the held-out test CSV, pick one of the five trained classifiers and
inspect its evaluation metrics, confusion matrix and classification report.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "Class"

FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
    "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
    "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2",
    "ShapeFactor3", "ShapeFactor4",
]

MODELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "kNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}

MODEL_BLURBS = {
    "Logistic Regression": "Linear multinomial model on standardised features.",
    "Decision Tree": "Entropy-based tree, depth-limited to curb overfitting.",
    "kNN": "Distance-weighted vote of the nearest neighbours (scaled features).",
    "Naive Bayes": "Gaussian NB on Yeo-Johnson power-transformed features.",
    "Support Vector Machine": "RBF-kernel SVM with Platt-calibrated probabilities.",
}

BEAN_PALETTE = sns.light_palette("#6B4F2A", as_cmap=True)

st.set_page_config(page_title="BeanSense | Dry Bean Classifier",
                   page_icon="🫘", layout="wide")


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------
def _refit_model(key: str):
    """Fallback if a pickled model cannot be loaded (e.g. sklearn version
    mismatch on the host): rebuild the pipeline with the tuned
    hyper-parameters and fit it on the training portion of the dataset."""
    import sys
    sys.path.insert(0, str(MODEL_DIR))
    import train_models as tm  # noqa: E402
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(ROOT / "data" / "dry_bean_full.csv").drop_duplicates()
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    y = encoder.transform(df[TARGET].str.strip().str.upper())
    X_tr, _, y_tr, _ = train_test_split(df[FEATURES], y, test_size=tm.TEST_FRACTION,
                                        stratify=y, random_state=tm.SEED)
    pipe, _ = tm.model_search_spaces()[key]
    params = json.loads((MODEL_DIR / "best_params.json").read_text())[key]
    prefix = "clf__estimator__" if key == "svm" else "clf__"
    pipe.set_params(**{prefix + k: v for k, v in params.items()})
    return pipe.fit(X_tr, y_tr)


@st.cache_resource(show_spinner="Loading trained models ...")
def load_artifacts():
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    models, notes = {}, []
    for name, key in MODELS.items():
        try:
            models[name] = joblib.load(MODEL_DIR / f"{key}.joblib")
        except Exception as exc:  # pragma: no cover - environment dependent
            notes.append(f"{name}: re-trained on start-up ({type(exc).__name__})")
            models[name] = _refit_model(key)
    return models, encoder, notes


@st.cache_data
def load_reference_metrics() -> pd.DataFrame | None:
    path = MODEL_DIR / "metrics_summary.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba, n_classes: int) -> dict[str, float]:
    present = np.unique(y_true)
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        elif len(present) == n_classes:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        elif len(present) > 1:   # uploaded file misses some classes
            auc = roc_auc_score(y_true, y_proba[:, present] /
                                y_proba[:, present].sum(axis=1, keepdims=True),
                                multi_class="ovr", average="macro", labels=present)
        else:
            auc = float("nan")
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, labels=present, average="macro",
                               zero_division=0),
        "Recall": recall_score(y_true, y_pred, labels=present, average="macro",
                            zero_division=0),
        "F1": f1_score(y_true, y_pred, labels=present, average="macro",
                        zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion(cm: np.ndarray, labels, normalise: bool):
    data = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) if normalise else cm
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(data, annot=True, fmt=".2f" if normalise else "d", cmap=BEAN_PALETTE,
                xticklabels=labels, yticklabels=labels, cbar=False,
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_xlabel("Predicted variety")
    ax.set_ylabel("Actual variety")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    return fig


def validate(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURES if c not in df.columns]


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
models, encoder, load_notes = load_artifacts()
class_names = list(encoder.classes_)

with st.sidebar:
    st.title("🫘 BeanSense")
    st.caption("Dry Bean variety classification · ML Assignment 2")

    st.subheader("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"],
                                help="16 shape features + optional 'Class' column. "
                                     "Upload test data only (free-tier limits).")
    sample_path = ROOT / "test_data.csv"
    if sample_path.exists():
        st.download_button("⬇ Download sample test_data.csv", sample_path.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")
        use_sample = st.checkbox("Use bundled test_data.csv", value=uploaded is None,
                                 disabled=uploaded is not None)
    else:
        use_sample = False

    st.subheader("2 · Model")
    model_name = st.selectbox("Choose a classifier", list(MODELS.keys()), index=4)
    st.caption(MODEL_BLURBS[model_name])

    st.subheader("3 · Display")
    normalise_cm = st.toggle("Normalise confusion matrix", value=False)

    for note in load_notes:
        st.warning(note)

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("Dry Bean Variety Classifier")
st.markdown(
    "Seven registered dry bean varieties (Barbunya, Bombay, Cali, Dermason, Horoz, "
    "Seker, Sira) are identified from **16 shape features** extracted from "
    "high-resolution grain images. Upload the held-out test set to evaluate any of "
    "the five trained models."
)

if uploaded is not None:
    data = read_csv(uploaded)
    source = f"uploaded file **{uploaded.name}**"
elif use_sample:
    data = read_csv(sample_path)
    source = "bundled **test_data.csv**"
else:
    st.info("⬅ Upload a CSV file (or tick *Use bundled test_data.csv*) to begin.")
    st.stop()

missing = validate(data)
if missing:
    st.error(f"The file is missing required feature columns: {', '.join(missing)}")
    st.stop()

X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
bad_rows = X.isna().any(axis=1)
if bad_rows.any():
    st.warning(f"Dropped {int(bad_rows.sum())} row(s) with missing / non-numeric values.")
    data, X = data.loc[~bad_rows].reset_index(drop=True), X.loc[~bad_rows].reset_index(drop=True)

has_labels = TARGET in data.columns
y_true = None
if has_labels:
    labels = data[TARGET].astype(str).str.strip().str.upper()
    unknown = sorted(set(labels) - set(class_names))
    if unknown:
        st.error(f"Unknown class label(s) in '{TARGET}': {', '.join(unknown)}. "
                 f"Expected one of {', '.join(class_names)}.")
        st.stop()
    y_true = encoder.transform(labels)

c1, c2, c3 = st.columns(3)
c1.metric("Rows evaluated", f"{len(X):,}")
c2.metric("Features", len(FEATURES))
c3.metric("Ground truth", "Available" if has_labels else "Not provided")
st.caption(f"Data source: {source}")

model = models[model_name]
y_pred = model.predict(X)
y_proba = model.predict_proba(X)

tab_eval, tab_compare, tab_pred, tab_data = st.tabs(
    ["📊 Evaluation", "🏁 Model comparison", "🔮 Predictions", "🗂 Data preview"])

# ---- Evaluation tab -------------------------------------------------------
with tab_eval:
    if not has_labels:
        st.warning(f"No '{TARGET}' column found, so metrics cannot be computed. "
                   "See the Predictions tab for model output.")
    else:
        st.subheader(f"{model_name} · evaluation metrics")
        scores = compute_metrics(y_true, y_pred, y_proba, len(class_names))
        cols = st.columns(6)
        for col, (k, v) in zip(cols, scores.items()):
            col.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")
        st.caption("Precision, Recall and F1 are macro-averaged; AUC is one-vs-rest macro.")

        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("**Confusion matrix**")
            cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
            st.pyplot(plot_confusion(cm, class_names, normalise_cm))
        with right:
            st.markdown("**Classification report**")
            report = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                           target_names=class_names, output_dict=True,
                                           zero_division=0)
            rep_df = pd.DataFrame(report).T
            rep_df["support"] = rep_df["support"].astype(int)
            st.dataframe(rep_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                              "f1-score": "{:.3f}"})
                         .background_gradient(cmap=BEAN_PALETTE, subset=["f1-score"]),
                         width="stretch")

# ---- Comparison tab -------------------------------------------------------
with tab_compare:
    if has_labels:
        st.subheader("All five models on this data")
        rows = []
        for name, mdl in models.items():
            pred = mdl.predict(X)
            rows.append({"ML Model Name": name,
                         **compute_metrics(y_true, pred, mdl.predict_proba(X),
                                           len(class_names))})
        comp = pd.DataFrame(rows).set_index("ML Model Name")
        st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#E8D9B5"),
                     width="stretch")

        metric_pick = st.radio("Metric to chart", list(comp.columns), horizontal=True,
                               index=list(comp.columns).index("F1"))
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ordered = comp[metric_pick].sort_values()
        colours = ["#6B4F2A" if n == model_name else "#C9B48A" for n in ordered.index]
        ax.barh(ordered.index, ordered.values, color=colours)
        ax.set_xlim(max(0.0, ordered.min() - 0.05), 1.0)
        for i, v in enumerate(ordered.values):
            ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)
        ax.set_xlabel(metric_pick)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(f"Highlighted bar = currently selected model ({model_name}).")
    else:
        st.info("Upload data with a 'Class' column to compare models.")

    ref = load_reference_metrics()
    if ref is not None:
        with st.expander("Reference results recorded at training time (test split)"):
            st.dataframe(ref, width="stretch", hide_index=True)

# ---- Predictions tab ------------------------------------------------------
with tab_pred:
    out = data.copy()
    out["Predicted"] = encoder.inverse_transform(y_pred)
    out["Confidence"] = y_proba.max(axis=1).round(4)
    if has_labels:
        out["Correct"] = out["Predicted"] == data[TARGET].astype(str).str.strip().str.upper()
        st.write(f"Misclassified rows: **{int((~out['Correct']).sum())}** of {len(out)}")
        only_wrong = st.checkbox("Show only misclassified rows")
        view = out.loc[~out["Correct"]] if only_wrong else out
    else:
        view = out
    st.dataframe(view, width="stretch", height=380)
    st.download_button("⬇ Download predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODELS[model_name]}.csv", mime="text/csv")

    st.markdown("**Predicted class distribution**")
    st.bar_chart(out["Predicted"].value_counts().reindex(class_names, fill_value=0),
                 color="#6B4F2A")

# ---- Data tab -------------------------------------------------------------
with tab_data:
    st.dataframe(data.head(200), width="stretch")
    st.markdown("**Summary statistics**")
    st.dataframe(X.describe().T, width="stretch")

st.divider()
st.caption("Dataset: Koklu & Ozkan (2020), *Dry Bean Dataset*, UCI Machine Learning "
           "Repository. Built with scikit-learn and Streamlit.")
