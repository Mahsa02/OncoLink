from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import importlib.util
import os
import subprocess
import sys
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(
    tempfile.gettempdir(), "oncolink-matplotlib"))
matplotlib.use("Agg")


try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as exc:
    XGBOOST_AVAILABLE = False
    print(f"XGBoost unavailable ({exc}) — skipping XGBoost experiments.")


SHAP_AVAILABLE = importlib.util.find_spec("shap") is not None
if not SHAP_AVAILABLE:
    print("SHAP not installed — skipping SHAP explainability.")


INPUT_DIR = "outputs_metabric"
RESULTS_DIR = os.path.join(INPUT_DIR, "model_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data():
    X_all = pd.read_csv(os.path.join(INPUT_DIR, "X_all_genes.csv"))
    X_top = pd.read_csv(os.path.join(INPUT_DIR, "X_top_variable_genes.csv"))
    X_pca_20 = pd.read_csv(os.path.join(INPUT_DIR, "X_pca_20.csv"))
    X_clinical = pd.read_csv(os.path.join(INPUT_DIR, "X_clinical.csv"))
    y = pd.read_csv(os.path.join(INPUT_DIR, "y_labels.csv")).iloc[:, 0]
    return X_all, X_top, X_pca_20, X_clinical, y


def make_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, feature_name):
    """
    Fit, evaluate, and save plots for one model/feature combination
    """
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n=== {model_name} | {feature_name} ===")
    print(f"Accuracy: {acc:.4f}  F1: {f1:.4f}  ROC AUC: {auc:.4f}")

    slug = f"{model_name.lower().replace(' ', '_')}_{feature_name.lower().replace(' ', '_')}"

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Confusion Matrix: {model_name} | {feature_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_{slug}.png"))
    plt.close()

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title(f"ROC Curve: {model_name} | {feature_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"roc_{slug}.png"))
    plt.close()

    return {"Model": model_name, "Features": feature_name, "Accuracy": acc, "F1": f1, "ROC_AUC": auc}


def save_feature_importance_fallback(model, X_test, reason):
    """
    Write SHAP-compatible artifacts from tree feature importances
    """
    feat_names = list(X_test.columns) if hasattr(X_test, "columns") else [
        f"f{i}" for i in range(X_test.shape[1])]
    importances = getattr(model, "feature_importances_",
                          np.zeros(len(feat_names)))
    importances = np.asarray(importances, dtype=float)

    if len(feat_names) != len(importances):
        feat_names = [f"f{i}" for i in range(len(importances))]

    np.save(os.path.join(RESULTS_DIR, "shap_values.npy"),
            importances.reshape(1, -1))

    with open(os.path.join(RESULTS_DIR, "shap_feature_names.json"), "w") as f:
        json.dump(feat_names, f)

    top_idx = np.argsort(importances)[-15:][::-1]

    plt.figure(figsize=(10, 6))
    plt.barh([feat_names[i] for i in top_idx]
             [::-1], importances[top_idx][::-1])
    plt.xlabel("Feature importance")
    plt.title("Feature importance fallback")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_summary.png"),
                bbox_inches="tight")
    plt.close()

    print(
        f"SHAP unavailable or unstable ({reason}). Saved feature-importance fallback artifacts.")


def compute_shap(model, X_train, X_test, feature_name):
    """
    Compute SHAP in a child process so native SHAP aborts cannot kill training
    """
    if not SHAP_AVAILABLE:
        save_feature_importance_fallback(
            model, X_test, "shap is not installed")
        return

    feat_names = list(X_test.columns) if hasattr(X_test, "columns") else [
        f"f{i}" for i in range(X_test.shape[1])]
    X_train_np = np.ascontiguousarray(X_train.values if hasattr(
        X_train, "values") else X_train, dtype=np.float64)
    X_test_np = np.ascontiguousarray(X_test.values if hasattr(
        X_test, "values") else X_test, dtype=np.float64)

    with tempfile.TemporaryDirectory(prefix="oncolink-shap-") as tmpdir:
        model_path = os.path.join(tmpdir, "model.pkl")
        train_path = os.path.join(tmpdir, "train.npy")
        test_path = os.path.join(tmpdir, "test.npy")
        names_path = os.path.join(tmpdir, "feature_names.json")
        script_path = os.path.join(tmpdir, "compute_shap_child.py")

        joblib.dump(model, model_path)
        np.save(train_path, X_train_np[: min(50, len(X_train_np))].copy())
        np.save(test_path, X_test_np[: min(30, len(X_test_np))].copy())

        with open(names_path, "w") as f:
            json.dump(feat_names, f)

        with open(script_path, "w") as f:
            f.write(
                """
import json
import os
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "oncolink-matplotlib"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

model_path, train_path, test_path, names_path, results_dir = sys.argv[1:]
model = joblib.load(model_path)
X_train = np.load(train_path)
X_test = np.load(test_path)

with open(names_path) as f:
    feat_names = json.load(f)

explainer = shap.TreeExplainer(model, data=X_train)
shap_vals = explainer.shap_values(X_test, check_additivity=False)

if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

shap_vals = np.asarray(shap_vals)

if shap_vals.ndim == 3 and shap_vals.shape[-1] == 2:
    shap_vals = shap_vals[:, :, 1]

np.save(os.path.join(results_dir, "shap_values.npy"), shap_vals)

with open(os.path.join(results_dir, "shap_feature_names.json"), "w") as f:
    json.dump(feat_names, f)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals, X_test, feature_names=feat_names, show=False, max_display=15)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "shap_summary.png"), bbox_inches="tight")
plt.close()
"""
            )

        result = subprocess.run(
            [sys.executable, script_path, model_path,
                train_path, test_path, names_path, RESULTS_DIR],
            capture_output=True,
            text=True,
        )

    if result.returncode == 0:
        print(f"SHAP values computed and saved for {feature_name}.")
    else:
        detail = (result.stderr or result.stdout or f"exit code {result.returncode}").strip(
        ).splitlines()[-1]
        save_feature_importance_fallback(model, X_test, detail)


def save_calibration_report(model, X_test, y_test, model_name, feature_name):
    """
    Check whether predicted probabilities can be read at face value:
    of patients given ~70% survival probability, do ~70% survive?
    Saves a reliability diagram and the Brier score.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    prob = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_test, prob)
    baseline = brier_score_loss(y_test, np.full_like(prob, np.mean(y_test)))

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    plt.plot(mean_pred, frac_pos, marker="o", label=f"{model_name} | {feature_name}")
    plt.xlabel("Predicted probability of being alive at follow-up")
    plt.ylabel("Observed fraction alive")
    plt.title(f"Calibration (Brier {brier:.3f}, baseline {baseline:.3f})")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "calibration.png"), bbox_inches="tight")
    plt.close()

    with open(os.path.join(RESULTS_DIR, "calibration.json"), "w") as f:
        json.dump({"brier": float(brier), "brier_baseline": float(baseline),
                   "mean_predicted": [float(v) for v in mean_pred],
                   "fraction_observed": [float(v) for v in frac_pos]}, f, indent=2)
    print(f"Calibration: Brier {brier:.4f} (predict-the-mean baseline {baseline:.4f})")


def train_survival_model(X, y, feature_name):
    """
    Fit a Random Survival Forest on the same feature set as the best classifier,
    using follow-up time, so the app can show a per-patient survival curve.
    """
    months_path = os.path.join(INPUT_DIR, "survival_months.csv")
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
    except ImportError:
        print("scikit-survival not installed; skipping survival model.")
        return
    if not os.path.exists(months_path):
        print("survival_months.csv not found; re-run processing.py to enable the survival model.")
        return

    months = pd.read_csv(months_path).iloc[:, 0].values
    X_np = np.asarray(X.values if hasattr(X, "values") else X, dtype=float)
    y_np = np.asarray(y)
    keep = ~np.isnan(months)
    # METABRIC: overall_survival == 1 means alive, so the death event is label 0
    target = Surv.from_arrays(event=(y_np[keep] == 0), time=months[keep])

    rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=15, random_state=42, n_jobs=-1)
    rsf.fit(X_np[keep], target)
    joblib.dump(rsf, os.path.join(RESULTS_DIR, "survival_model.pkl"))
    with open(os.path.join(RESULTS_DIR, "survival_model_info.json"), "w") as f:
        json.dump({"model_name": "Random Survival Forest", "feature_name": feature_name,
                   "n_patients": int(keep.sum())}, f, indent=2)
    print(f"Survival model (Random Survival Forest, {feature_name}) saved.")


def train_incremental_base():
    """
    Train and save an SGDClassifier for online/incremental updates
    """
    X_all = pd.read_csv(os.path.join(INPUT_DIR, "X_all_genes.csv"))
    y = pd.read_csv(os.path.join(INPUT_DIR, "y_labels.csv")).iloc[:, 0]

    X_train, X_test, y_train, y_test = make_split(X_all, y)

    sgd = SGDClassifier(loss="log_loss", random_state=42, max_iter=1000)
    sgd.fit(X_train, y_train)

    auc = roc_auc_score(y_test, sgd.predict_proba(X_test)[:, 1])
    print(f"\nIncremental (SGD) baseline ROC AUC: {auc:.4f}")

    joblib.dump(sgd, os.path.join(RESULTS_DIR, "incremental_model.pkl"))
    print("Incremental model saved.")


def run_experiments():
    X_all, X_top, X_pca_20, X_clinical, y = load_data()

    feature_sets = [
        ("Clinical Only", X_clinical),
        ("Top Features", X_top),
        ("PCA 20", X_pca_20),
        ("All Features", X_all),
    ]

    models = [
        ("Logistic Regression", LogisticRegression(
            max_iter=2000, C=0.1, random_state=42)),
        ("Random Forest", RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1)),
    ]

    if XGBOOST_AVAILABLE:
        models.append((
            "XGBoost",
            xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
        ))

    results = []
    best_model = None
    best_score = -1
    best_metadata = {}
    best_X_train = None
    best_X_test = None
    best_y_test = None
    best_X_full = None

    for feature_name, X in feature_sets:
        X_train, X_test, y_train, y_test = make_split(X, y)

        for model_name, model_template in models:
            model = clone(model_template)

            result = evaluate_model(
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                model_name=model_name,
                feature_name=feature_name,
            )

            results.append(result)

            if result["ROC_AUC"] > best_score:
                best_score = result["ROC_AUC"]
                best_model = model
                best_metadata = {"model_name": model_name,
                                 "feature_name": feature_name}
                best_X_train = X_train
                best_X_test = X_test
                best_y_test = y_test
                best_X_full = X

    results_df = pd.DataFrame(results).sort_values(
        by="ROC_AUC", ascending=False)

    joblib.dump(best_model, os.path.join(RESULTS_DIR, "best_model.pkl"))

    with open(os.path.join(RESULTS_DIR, "best_model_info.txt"), "w") as f:
        f.write(str(best_metadata))

    print(f"\nBest model: {best_metadata}")

    if hasattr(best_model, "feature_importances_") and best_X_test is not None:
        compute_shap(best_model, best_X_train, best_X_test,
                     best_metadata["feature_name"])

    results_df.to_csv(os.path.join(
        RESULTS_DIR, "model_comparison.csv"), index=False)

    save_calibration_report(best_model, best_X_test, best_y_test,
                            best_metadata["model_name"], best_metadata["feature_name"])
    train_survival_model(best_X_full, y, best_metadata["feature_name"])

    train_incremental_base()

    return results_df


if __name__ == "__main__":
    run_experiments()
