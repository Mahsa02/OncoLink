def f():
    if x:
        return 1
"""
Leak-free evaluation for OncoLink.

Every preprocessing step (imputation, scaling, gene selection, PCA) is fit
inside each training fold with a sklearn Pipeline, so test folds never
influence the transforms. Results are reported as mean +/- std over
stratified 5-fold cross-validation instead of a single train/test split.

Two tasks are evaluated:
  1. Classification of the `overall_survival` flag (alive vs. deceased at
     last follow-up) -- the label the app currently predicts.
  2. Survival analysis on (overall_survival_months, death event), scored
     with Harrell's C-index, which uses follow-up time instead of
     discarding it.

Run:  python evaluate_cv.py
Writes: outputs_metabric/model_results/cv_classification.csv
        outputs_metabric/model_results/cv_survival.csv
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = "data/METABRIC_RNA_Mutation.csv"
OUT_DIR = os.path.join("outputs_metabric", "model_results")
SEED = 42
N_FOLDS = 5

CLINICAL = [
    "age_at_diagnosis", "chemotherapy", "hormone_therapy", "radio_therapy",
    "tumor_size", "tumor_stage", "lymph_nodes_examined_positive",
    "er_status", "her2_status", "pr_status", "neoplasm_histologic_grade",
]
EXCLUDE = {"patient_id", "cancer_type", "overall_survival_months",
           "type_of_breast_surgery", "cohort", "overall_survival"}


# ---------------------------------------------------------------- data
def load():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = (df.columns.str.strip().str.lower().str.replace(" ", "_")
                  .str.replace("+", "plus").str.replace("-", "_"))
    df["overall_survival"] = pd.to_numeric(df["overall_survival"], errors="coerce")
    df["overall_survival_months"] = pd.to_numeric(df["overall_survival_months"], errors="coerce")
    df = df.dropna(subset=["overall_survival", "overall_survival_months"])

    for c in ["er_status", "her2_status", "pr_status"]:
        df[c] = df[c].astype(str).str.lower().str.strip().map(
            {"positive": 1, "pos": 1, "1": 1, "negative": 0, "neg": 0, "0": 0})

    clinical = [c for c in CLINICAL if c in df.columns]
    genes = [c for c in df.columns
             if c not in EXCLUDE and c not in clinical and not c.endswith("_mut")]
    for c in clinical + genes:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Column filter uses missingness only (no label information), same 80% rule as processing.py
    keep = lambda cols: [c for c in cols if df[c].notna().mean() >= 0.8]
    clinical, genes = keep(clinical), keep(genes)

    X = df[clinical + genes]
    y = df["overall_survival"].astype(int).values
    # METABRIC: overall_survival == 1 means alive, so death event is 0
    event = (y == 0)
    time = df["overall_survival_months"].values
    return X, y, event, time, clinical, genes


# ---------------------------------------------------------------- features
class TopVariance(BaseEstimator, TransformerMixin):
    """Keep the k highest-variance columns, computed on the training fold
    BEFORE scaling (after scaling every column has variance 1)."""
    def __init__(self, k=1000):
        self.k = k

    def fit(self, X, y=None):
        v = np.nanvar(np.asarray(X, dtype=float), axis=0)
        self.idx_ = np.argsort(v)[-min(self.k, len(v)):]
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.idx_]


def featurizer(kind, clinical, genes):
    clin = Pipeline([("impute", SimpleImputer(strategy="mean")),
                     ("scale", StandardScaler())])
    if kind == "Clinical Only":
        return ColumnTransformer([("clin", clin, clinical)])

    steps = [("impute", SimpleImputer(strategy="mean"))]
    if kind == "Top Variable Genes":
        steps += [("select", TopVariance(k=1000))]
    steps += [("scale", StandardScaler())]
    if kind == "PCA 20":
        steps += [("pca", PCA(n_components=20, random_state=SEED))]
    return ColumnTransformer([("clin", clin, clinical),
                              ("genes", Pipeline(steps), genes)])


def classifiers():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=0.1),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.5,
                                          random_state=SEED, n_jobs=-1, verbosity=0)
    except ImportError:
        print("XGBoost not installed, skipping.")
    return models


FEATURE_SETS = ["Clinical Only", "All Features", "Top Variable Genes", "PCA 20"]


# ---------------------------------------------------------------- task 1
def run_classification(X, y, clinical, genes):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for fs in FEATURE_SETS:
        for name, model in classifiers().items():
            pipe = Pipeline([("features", featurizer(fs, clinical, genes)),
                             ("model", clone(model))])
            s = cross_validate(pipe, X, y, cv=cv,
                               scoring=["roc_auc", "f1", "accuracy"])
            rows.append({
                "Model": name, "Features": fs,
                "ROC_AUC_mean": s["test_roc_auc"].mean(), "ROC_AUC_std": s["test_roc_auc"].std(),
                "F1_mean": s["test_f1"].mean(), "F1_std": s["test_f1"].std(),
                "Accuracy_mean": s["test_accuracy"].mean(), "Accuracy_std": s["test_accuracy"].std(),
            })
            print(f"{name:20s} | {fs:18s} | AUC {rows[-1]['ROC_AUC_mean']:.3f} "
                  f"± {rows[-1]['ROC_AUC_std']:.3f}")
    return pd.DataFrame(rows).sort_values("ROC_AUC_mean", ascending=False)


# ---------------------------------------------------------------- task 2
def run_survival(X, y, event, time, clinical, genes):
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.util import Surv
    except ImportError:
        print("scikit-survival not installed, skipping survival analysis.")
        return None

    Y = Surv.from_arrays(event=event, time=time)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    models = {
        "Cox (elastic net)": CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=[0.05], fit_baseline_model=False),
        "Random Survival Forest": RandomSurvivalForest(n_estimators=200, min_samples_leaf=15,
                                                       random_state=SEED, n_jobs=-1),
    }
    rows = []
    for fs in ["Clinical Only", "PCA 20"]:
        for name, model in models.items():
            scores = []
            for tr, te in cv.split(X, event):
                pipe = Pipeline([("features", featurizer(fs, clinical, genes)),
                                 ("model", clone(model))])
                pipe.fit(X.iloc[tr], Y[tr])
                scores.append(pipe.score(X.iloc[te], Y[te]))  # Harrell's C-index
            rows.append({"Model": name, "Features": fs,
                         "C_index_mean": np.mean(scores), "C_index_std": np.std(scores)})
            print(f"{name:22s} | {fs:13s} | C-index {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    return pd.DataFrame(rows).sort_values("C_index_mean", ascending=False)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y, event, time, clinical, genes = load()
    print(f"Patients: {len(X)} | clinical features: {len(clinical)} | genes: {len(genes)} "
          f"| alive: {y.mean():.1%}\n")

    print("== Classification (5-fold CV, leak-free) ==")
    cls = run_classification(X, y, clinical, genes)
    cls.to_csv(os.path.join(OUT_DIR, "cv_classification.csv"), index=False)

    print("\n== Survival analysis (5-fold CV, C-index) ==")
    surv = run_survival(X, y, event, time, clinical, genes)
    if surv is not None:
        surv.to_csv(os.path.join(OUT_DIR, "cv_survival.csv"), index=False)
