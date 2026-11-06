# 🫘 BeanSense — Dry Bean Variety Classification

**ML Assignment 2 · M.Tech (Data Science & Engineering) · BITS Pilani WILP**

Live app: `<PASTE YOUR STREAMLIT APP LINK HERE>`

---

## a. Problem statement

Dry beans are the most widely produced pulse crop, and seed quality is a key driver of
yield and market price. Sorting beans by variety is still largely manual, slow and
error-prone. The goal of this project is to **automatically identify the variety of a dry
bean grain (7 classes) from geometric shape features extracted from its image**, by
building, comparing and deploying five classical machine-learning classifiers:
Logistic Regression, Decision Tree, k-Nearest Neighbours, Gaussian Naive Bayes and a
Support Vector Machine. This is a **multi-class classification** problem.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Dry Bean Dataset |
| Source | UCI Machine Learning Repository (ID 602) — Koklu, M. & Ozkan, I. A. (2020), *Multiclass classification of dry beans using computer vision and machine learning techniques*, Computers and Electronics in Agriculture, 174, 105507 |
| Instances | 13,611 grains (13,543 after removing 68 exact duplicate rows) |
| Features | 16 numeric (12 dimension features + 4 shape factors) |
| Target | `Class` — 7 varieties: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA |
| Missing values | None |

**Features.** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation
(major/minor axis ratio), Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity,
roundness, Compactness, ShapeFactor1–ShapeFactor4. All of them were computed from
high-resolution images of individual grains.

**Class distribution (imbalanced).** DERMASON 26.1%, SIRA 19.4%, SEKER 14.9%, HOROZ 14.2%,
CALI 12.0%, BARBUNYA 9.7%, BOMBAY 3.8%.

**Key properties found during EDA.**
- Size features (Area, ConvexArea, Perimeter, EquivDiameter, axis lengths) are very
  strongly correlated (r > 0.95).
- Feature scales differ by up to 8 orders of magnitude (Area ≈ 10⁴–10⁵,
  ShapeFactor2 ≈ 10⁻³), so scaling matters for distance- and margin-based models.

**Pre-processing and experimental setup.**
- Removed duplicate rows; label-encoded the target.
- Stratified **80 / 20 train–test split** (seed 2611). Train = 10,834 rows,
  Test = 2,709 rows. The test split is saved as `test_data.csv`.
- `StandardScaler` for LR, kNN and SVM. A Yeo-Johnson `PowerTransformer` for Gaussian NB
  (makes the skewed features closer to Gaussian). No scaling for the Decision Tree.
- Hyper-parameters tuned with `GridSearchCV` (5-fold stratified CV, macro-F1) on the
  training set only. The test set was used once, for the final evaluation.

## c. GitHub repository link

`https://github.com/<your-username>/drybean-classifier`

```
drybean-classifier/
├── app.py                         # Streamlit application
├── requirements.txt
├── README.md
├── test_data.csv                  # held-out test split (2,709 rows, with labels)
├── .streamlit/config.toml         # custom theme
├── data/
│   └── dry_bean_full.csv          # full UCI dataset (used for training)
└── model/
    ├── train_models.py            # training + evaluation script
    ├── DryBean_Classification.ipynb  # EDA + experiments notebook
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── label_encoder.joblib
    ├── best_params.json           # tuned hyper-parameters
    └── metrics_summary.csv        # comparison table
```

## d. Models used

| Model | Pipeline | Tuned hyper-parameters |
|---|---|---|
| Logistic Regression | StandardScaler → LogisticRegression (multinomial) | C = 1.0 |
| Decision Tree | DecisionTreeClassifier | criterion = entropy, max_depth = 10, min_samples_leaf = 10 |
| kNN | StandardScaler → KNeighborsClassifier | k = 9, weights = distance |
| Naive Bayes | PowerTransformer → GaussianNB | var_smoothing = 1e-3 |
| Support Vector Machine | StandardScaler → SVC (RBF), Platt-calibrated | C = 10, gamma = 0.1 |

### Comparison table (test set, n = 2,709)

Precision, Recall and F1 are **macro-averaged** across the 7 classes. AUC is
**one-vs-rest macro** AUC.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9276 | **0.9945** | 0.9394 | 0.9363 | 0.9377 | 0.9125 |
| Decision Tree | 0.9081 | 0.9817 | 0.9260 | 0.9191 | 0.9223 | 0.8887 |
| kNN | 0.9288 | 0.9889 | 0.9421 | 0.9377 | 0.9396 | 0.9138 |
| Naive Bayes | 0.8955 | 0.9918 | 0.9071 | 0.9048 | 0.9049 | 0.8742 |
| Support Vector Machine | **0.9358** | 0.9924 | **0.9484** | **0.9443** | **0.9462** | **0.9223** |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A strong linear baseline: 92.8% accuracy and MCC 0.913, only about 0.8 pp behind the best model. It has the **highest AUC (0.9945)**, so its class probabilities rank the samples best, even though a linear boundary loses some accuracy where classes overlap. Most of its errors are SIRA↔DERMASON (84 of 2,709) and BARBUNYA→CALI (16). These are look-alike varieties with overlapping size and shape. Fast, stable, and easy to interpret. |
| Decision Tree | Second weakest overall (only NB is lower on accuracy, F1 and MCC): 90.8% accuracy, F1 0.922, MCC 0.889, and the **lowest AUC (0.982)** because leaf-based probabilities are coarse. Axis-parallel splits cannot capture the oblique boundaries between correlated size features. It is tied with Naive Bayes for the most SIRA↔DERMASON confusions (109). Depth was limited to 10 with min_samples_leaf = 10 to control overfitting (CV F1 0.923 ≈ test F1 0.922, so no overfitting gap). |
| kNN | Second best overall (accuracy 92.9%, F1 0.940, MCC 0.914), just ahead of LR. Scaling was essential, and distance weighting with k = 9 worked best. Local neighbourhoods separate the compact clusters well (BOMBAY F1 = 1.0, HOROZ 0.962). AUC (0.989) is lower than LR/SVM because neighbour-vote probabilities are less smooth. Prediction cost grows with the training size (the model stores all 10.8 k rows). |
| Naive Bayes | The lowest accuracy (89.6%), F1 (0.905) and MCC (0.874). The independence assumption is badly violated by the highly correlated size features, so evidence gets counted twice. This causes the most DERMASON→SIRA (70) and BARBUNYA→CALI (47) errors. Even so, its AUC is high (0.992): the probability *ranking* is good, but the decision thresholds are skewed. The power transform helped by making the features closer to Gaussian. Very fast to train. |
| Support Vector Machine | **Best on 5 of 6 metrics**: accuracy 93.6%, precision 0.948, recall 0.944, F1 0.946, MCC 0.922. The RBF kernel learns smooth non-linear boundaries between overlapping varieties. It has the best per-class F1 for BARBUNYA (0.940), SEKER (0.957) and SIRA (0.891), and the fewest BARBUNYA↔CALI errors (16, compared with 20–67 for the other models). Its AUC (0.992) is a little below LR because Platt calibration is needed to get probabilities. Training is the slowest of the five (≈25 s with grid search), but that is still cheap. |
| **Overall winner for this dataset** | **Support Vector Machine (RBF kernel)**: the highest Accuracy, Precision, Recall, F1 and MCC, with balanced performance across all seven classes. Logistic Regression is the best choice if well-ranked probabilities (AUC) or interpretability matter most. In every model the main source of error is the SIRA–DERMASON pair, which points to a limit in the features rather than the model. BOMBAY is classified perfectly by all five models. |

---

## Streamlit app features

- **CSV upload** of test data (sidebar), plus a one-click download/bundled copy of `test_data.csv`
- **Model selection dropdown** for the five classifiers
- **Evaluation metrics** (Accuracy, AUC, Precision, Recall, F1, MCC) as metric cards
- **Confusion matrix** (raw or row-normalised) and a **classification report**
- **Model comparison** tab that scores all five models on the uploaded file, with a bar chart
- **Predictions** tab with confidence, a misclassification filter and a CSV download
- Input validation (missing columns, non-numeric values, unknown labels). If no `Class`
  column is given, the app still makes predictions

## Run locally

```bash
pip install -r requirements.txt
python model/train_models.py      # optional: re-train and regenerate all artefacts
streamlit run app.py
```
