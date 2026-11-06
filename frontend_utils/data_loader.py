import csv
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs_metabric")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "model_results")

RAW_DATA_PATH = os.path.join(DATA_DIR, "METABRIC_RNA_Mutation.csv")
X_ALL_PATH = os.path.join(OUTPUTS_DIR, "X_all_genes.csv")
X_TOP_PATH = os.path.join(OUTPUTS_DIR, "X_top_variable_genes.csv")
X_PCA20_PATH = os.path.join(OUTPUTS_DIR, "X_pca_20.csv")
X_CLINICAL_PATH = os.path.join(OUTPUTS_DIR, "X_clinical.csv")
Y_LABELS_PATH = os.path.join(OUTPUTS_DIR, "y_labels.csv")

BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pkl")
BEST_MODEL_INFO_PATH = os.path.join(RESULTS_DIR, "best_model_info.txt")
MODEL_COMPARISON_PATH = os.path.join(RESULTS_DIR, "model_comparison.csv")
SHAP_VALUES_PATH = os.path.join(RESULTS_DIR, "shap_values.npy")
SHAP_NAMES_PATH = os.path.join(RESULTS_DIR, "shap_feature_names.json")
SURVIVAL_MODEL_PATH = os.path.join(RESULTS_DIR, "survival_model.pkl")
SURVIVAL_MODEL_INFO_PATH = os.path.join(RESULTS_DIR, "survival_model_info.json")
CALIBRATION_PATH = os.path.join(RESULTS_DIR, "calibration.json")

GENE_COLS_PATH = os.path.join(OUTPUTS_DIR, "gene_column_names.csv")
CLINICAL_COLS_PATH = os.path.join(OUTPUTS_DIR, "clinical_column_names.csv")
TOP_GENE_IDX_PATH = os.path.join(OUTPUTS_DIR, "top_gene_indices.csv")
SCALER_GENES_PATH = os.path.join(OUTPUTS_DIR, "scaler_genes.pkl")
SCALER_CLINICAL_PATH = os.path.join(OUTPUTS_DIR, "scaler_clinical.pkl")
PCA20_PATH = os.path.join(OUTPUTS_DIR, "pca_20.pkl")
PCA50_PATH = os.path.join(OUTPUTS_DIR, "pca_50.pkl")


@st.cache_data
def load_raw_data():
    if not os.path.exists(RAW_DATA_PATH):
        return None
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    df.columns = _clean_columns(df.columns)
    return df


@st.cache_data
def load_processed_features():
    data = {}
    paths = {
        "X_all": X_ALL_PATH,
        "X_top": X_TOP_PATH,
        "X_pca_20": X_PCA20_PATH,
        "X_clinical": X_CLINICAL_PATH,
        "y": Y_LABELS_PATH,
    }
    for key, path in paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            data[key] = df.iloc[:, 0] if key == "y" else df
        else:
            data[key] = None
    return data


@st.cache_resource
def load_best_model():
    if not os.path.exists(BEST_MODEL_PATH):
        return None
    return joblib.load(BEST_MODEL_PATH)


@st.cache_resource
def load_survival_model():
    """Random Survival Forest trained in model.py on the best classifier's feature set."""
    if not os.path.exists(SURVIVAL_MODEL_PATH):
        return None, {}
    info = {}
    if os.path.exists(SURVIVAL_MODEL_INFO_PATH):
        with open(SURVIVAL_MODEL_INFO_PATH) as f:
            info = json.load(f)
    return _safe_joblib_load(SURVIVAL_MODEL_PATH), info


@st.cache_data
def load_calibration():
    if not os.path.exists(CALIBRATION_PATH):
        return None
    with open(CALIBRATION_PATH) as f:
        return json.load(f)


def predict_survival_curve(survival_model, feature_row):
    """
    Return the patient's predicted survival curve as a DataFrame
    (Years, Survival Probability) plus point estimates at 5 and 10 years.
    """
    if survival_model is None or feature_row is None or len(feature_row) == 0:
        return None
    X = np.asarray(feature_row.values if hasattr(feature_row, "values") else feature_row, dtype=float)
    try:
        fn = survival_model.predict_survival_function(X[:1])[0]
    except ValueError:
        return None
    months = np.asarray(fn.x, dtype=float)
    probs = np.asarray(fn.y, dtype=float)

    def at(month):
        if month > months.max():
            return None
        return float(fn(month))

    curve = pd.DataFrame({"Years": months / 12.0, "Survival Probability": probs})
    return {"curve": curve, "five_year": at(60), "ten_year": at(120)}


@st.cache_data
def load_best_model_info():
    if not os.path.exists(BEST_MODEL_INFO_PATH):
        return {"model_name": "Unknown", "feature_name": "Unknown"}
    with open(BEST_MODEL_INFO_PATH) as f:
        content = f.read()
    try:
        return eval(content)
    except Exception:
        return {"model_name": "Unknown", "feature_name": "Unknown"}


@st.cache_data
def load_model_comparison():
    if not os.path.exists(MODEL_COMPARISON_PATH):
        return None
    return pd.read_csv(MODEL_COMPARISON_PATH)


@st.cache_data
def load_shap_data():
    if not os.path.exists(SHAP_VALUES_PATH) or not os.path.exists(SHAP_NAMES_PATH):
        return None, None
    try:
        shap_vals = np.load(SHAP_VALUES_PATH)
        with open(SHAP_NAMES_PATH) as f:
            feat_names = json.load(f)
        return shap_vals, feat_names
    except Exception:
        return None, None


@st.cache_data
def load_feature_metadata():
    gene_cols = _read_single_col_csv(GENE_COLS_PATH, "gene_name")
    clinical_cols = _read_single_col_csv(CLINICAL_COLS_PATH, "clinical_name")
    top_gene_idx = _read_numeric_csv(TOP_GENE_IDX_PATH)

    if not gene_cols and os.path.exists(X_ALL_PATH):
        x_all_cols = pd.read_csv(X_ALL_PATH, nrows=0).columns.tolist()
        clinical_cols = clinical_cols or pd.read_csv(X_CLINICAL_PATH, nrows=0).columns.tolist()
        gene_cols = [c for c in x_all_cols if c not in set(clinical_cols)]

    return {
        "gene_columns": gene_cols,
        "clinical_columns": clinical_cols,
        "top_gene_indices": top_gene_idx,
    }


@st.cache_resource
def load_preprocessors():
    assets = {
        "scaler_genes": _safe_joblib_load(SCALER_GENES_PATH),
        "scaler_clinical": _safe_joblib_load(SCALER_CLINICAL_PATH),
        "pca_20": _safe_joblib_load(PCA20_PATH),
        "pca_50": _safe_joblib_load(PCA50_PATH),
    }
    return assets


def get_feature_matrix_for_model(model_info, processed, model=None):
    resolved_name = resolve_feature_name_for_model(model_info, processed, model=model)
    if resolved_name == "PCA 20":
        return processed.get("X_pca_20")
    if resolved_name == "Top Features":
        return processed.get("X_top")
    if resolved_name == "Clinical Only":
        return processed.get("X_clinical")
    return processed.get("X_all")


def resolve_feature_name_for_model(model_info, processed, model=None):
    n_features = getattr(model, "n_features_in_", None) if model is not None else None

    x_all = processed.get("X_all")
    x_top = processed.get("X_top")
    x_pca_20 = processed.get("X_pca_20")
    x_clinical = processed.get("X_clinical")

    if n_features is not None:
        if x_pca_20 is not None and x_pca_20.shape[1] == n_features:
            return "PCA 20"
        if x_clinical is not None and x_clinical.shape[1] == n_features:
            return "Clinical Only"
        if x_top is not None and x_top.shape[1] == n_features and x_all is not None and x_all.shape[1] != n_features:
            return "Top Features"
        if x_all is not None and x_all.shape[1] == n_features:
            return "All Features"

    feature_name = str(model_info.get("feature_name", "")).lower()
    if "pca" in feature_name and "20" in feature_name:
        return "PCA 20"
    if "top" in feature_name:
        return "Top Features"
    if "clinical" in feature_name:
        return "Clinical Only"
    return "All Features"


def get_patient_profile(raw_df, index):
    if raw_df is None or index >= len(raw_df):
        return None
    row = raw_df.iloc[index]
    profile = {
        "Patient ID": row.get("patient_id", f"Patient-{index}"),
        "Age at Diagnosis": row.get("age_at_diagnosis", "N/A"),
        "Tumor Size (mm)": row.get("tumor_size", "N/A"),
        "Tumor Stage": row.get("tumor_stage", "N/A"),
        "Histologic Grade": row.get("neoplasm_histologic_grade", "N/A"),
        "ER Status": row.get("er_status", "N/A"),
        "HER2 Status": row.get("her2_status", "N/A"),
        "PR Status": row.get("pr_status", "N/A"),
        "Lymph Nodes Positive": row.get("lymph_nodes_examined_positive", "N/A"),
        "Hormone Therapy": _format_binary(row.get("hormone_therapy")),
        "Chemotherapy": _format_binary(row.get("chemotherapy")),
        "Radiation Therapy": _format_binary(row.get("radio_therapy")),
        "Cancer Type": row.get("cancer_type", "N/A"),
        "Surgery Type": row.get("type_of_breast_surgery", "N/A"),
        "_index": index,
    }
    return profile


def predict_patient(model, X, index):
    if model is None or X is None or index >= len(X):
        return None
    patient_features = X.iloc[[index]]
    return predict_feature_row(model, patient_features)


def predict_feature_row(model, feature_row):
    if model is None or feature_row is None or len(feature_row) == 0:
        return None
    try:
        pred = model.predict(feature_row)[0]
        prob = model.predict_proba(feature_row)[0]
    except ValueError:
        arr = feature_row.values
        pred = model.predict(arr)[0]
        prob = model.predict_proba(arr)[0]
    return {
        "prediction": int(pred),
        "label": "Lower Risk" if int(pred) == 1 else "Higher Risk",
        "probability_respond": float(prob[1]),
        "probability_non_respond": float(prob[0]),
        "confidence": float(max(prob)),
    }


def build_new_patient_features(profile, external_case = None) :
    metadata = load_feature_metadata()
    preprocessors = load_preprocessors()
    processed = load_processed_features()
    model_info = load_best_model_info()
    model = load_best_model()

    gene_cols = metadata.get("gene_columns", [])
    clinical_cols = metadata.get("clinical_columns", [])
    top_gene_indices = metadata.get("top_gene_indices", [])

    scaler_genes = preprocessors.get("scaler_genes")
    scaler_clinical = preprocessors.get("scaler_clinical")
    pca_20 = preprocessors.get("pca_20")
    pca_50 = preprocessors.get("pca_50")

    if scaler_genes is None or scaler_clinical is None:
        raise FileNotFoundError("Saved scalers not found. Run `python processing.py` first.")

    imputed_fields = []
    clinical_raw = []
    clinical_lookup = _profile_to_clinical_lookup(profile)

    clinical_means = list(getattr(scaler_clinical, "mean_", np.zeros(len(clinical_cols))))
    for i, col in enumerate(clinical_cols):
        value = clinical_lookup.get(col)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            value = float(clinical_means[i]) if i < len(clinical_means) else 0.0
            imputed_fields.append(col)
        clinical_raw.append(float(value))

    clinical_raw_df = pd.DataFrame([clinical_raw], columns=clinical_cols)
    clinical_scaled_arr = scaler_clinical.transform(clinical_raw_df)
    clinical_df = pd.DataFrame(clinical_scaled_arr, columns=clinical_cols)

    gene_raw = np.array(getattr(scaler_genes, "mean_", np.zeros(len(gene_cols))), dtype=float)
    matched_gene_count = 0
    genomic_mode = "cohort_mean_fallback"
    gene_inputs = _extract_gene_inputs(external_case)

    if gene_inputs:
        gene_index = {gene: idx for idx, gene in enumerate(gene_cols)}
        for gene_name, raw_value in gene_inputs.items():
            idx = gene_index.get(gene_name)
            if idx is None:
                continue
            gene_raw[idx] = raw_value
            matched_gene_count += 1
        if matched_gene_count > 0:
            genomic_mode = "matched_external_genes"

    gene_raw_df = pd.DataFrame([gene_raw], columns=gene_cols)
    gene_scaled_arr = scaler_genes.transform(gene_raw_df)
    gene_df = pd.DataFrame(gene_scaled_arr, columns=gene_cols)

    all_features = pd.concat([gene_df, clinical_df], axis=1)

    x_top_cols = processed.get("X_top").columns.tolist() if processed.get("X_top") is not None else []
    x_pca20_cols = processed.get("X_pca_20").columns.tolist() if processed.get("X_pca_20") is not None else []
    x_pca50_cols = []

    top_arr = gene_scaled_arr[:, top_gene_indices] if top_gene_indices else np.empty((1, 0))
    top_df = pd.DataFrame(np.concatenate([top_arr, clinical_scaled_arr], axis=1), columns=x_top_cols) if x_top_cols else None

    pca20_scores = pca_20.transform(gene_scaled_arr) if pca_20 is not None else np.empty((1, 0))
    pca20_df = pd.DataFrame(np.concatenate([pca20_scores, clinical_scaled_arr], axis=1), columns=x_pca20_cols) if x_pca20_cols else None

    if pca_50 is not None:
        pca50_scores = pca_50.transform(gene_scaled_arr)
        x_pca50_cols = [str(i) for i in range(pca50_scores.shape[1] + clinical_scaled_arr.shape[1])]
        pca50_df = pd.DataFrame(np.concatenate([pca50_scores, clinical_scaled_arr], axis=1), columns=x_pca50_cols)
    else:
        pca50_df = None

    resolved_feature_name = resolve_feature_name_for_model(model_info, processed, model=model)
    if resolved_feature_name == "PCA 20":
        model_features = pca20_df
    elif resolved_feature_name == "Top Features":
        model_features = top_df
    elif resolved_feature_name == "Clinical Only":
        model_features = clinical_df
    else:
        model_features = all_features

    if model_features is None:
        raise RuntimeError("Could not build the feature vector required by the saved model.")

    return {
        "model_features": model_features,
        "all_features": all_features,
        "clinical_features": clinical_df,
        "pca20_features": pca20_df,
        "imputed_fields": imputed_fields,
        "genomic_mode": genomic_mode,
        "matched_gene_count": matched_gene_count,
        "resolved_feature_name": resolved_feature_name,
    }


def get_chart_paths():
    charts = {"confusion": {}, "roc": {}}
    if not os.path.exists(RESULTS_DIR):
        return charts
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".png"):
            continue
        fpath = os.path.join(RESULTS_DIR, fname)
        if fname.startswith("confusion_"):
            charts["confusion"][fname.replace("confusion_", "").replace(".png", "")] = fpath
        elif fname.startswith("roc_"):
            charts["roc"][fname.replace("roc_", "").replace(".png", "")] = fpath
    return charts


def get_output_chart_paths():
    charts = {}
    for fname in ["class_distribution.png", "pca_explained_variance.png"]:
        fpath = os.path.join(OUTPUTS_DIR, fname)
        if os.path.exists(fpath):
            charts[fname.replace(".png", "")] = fpath
    return charts


def load_external_case_metadata(path = None):
    if not path or not os.path.exists(path):
        return None

    with open(path, newline="") as f:
        text = f.read()
    return parse_series_matrix_metadata(text)


def parse_uploaded_case_metadata(file_bytes):
    text = _decode_uploaded_text(file_bytes)
    if not text:
        return None
    return parse_series_matrix_metadata(text)


def parse_series_matrix_metadata(text):
    if not text:
        return None

    char_rows = []
    sample_rows = {}
    for line in text.splitlines():
        if line.startswith("!series_matrix_table_begin"):
            break
        if not line.startswith("!Sample_"):
            continue
        row = next(csv.reader([line], delimiter="\t"))
        key = row[0][1:]
        values = [cell.strip().strip('"') for cell in row[1:]]
        if key == "Sample_characteristics_ch1":
            char_rows.append(values)
        else:
            sample_rows[key] = values

    n_samples = len(sample_rows.get("Sample_geo_accession", []))
    if n_samples == 0:
        return None

    records = []
    for i in range(n_samples):
        rec = {
            "sample_title": _row_get(sample_rows, "Sample_title", i),
            "sample_accession": _row_get(sample_rows, "Sample_geo_accession", i),
            "sample_source_name": _row_get(sample_rows, "Sample_source_name_ch1", i),
        }
        for values in char_rows:
            raw = values[i] if i < len(values) else ""
            parsed = _parse_characteristic(raw)
            if parsed:
                rec[parsed[0]] = parsed[1]
        records.append(rec)

    df = pd.DataFrame(records)
    if not df.empty:
        df["display_name"] = df.apply(_external_case_label, axis=1)
    return df


def derive_profile_from_external_case(case_row) :
    if case_row is None:
        return {}
    case = dict(case_row)
    return {
        "patient_label": case.get("sample_title") or case.get("sample_id") or case.get("sample_accession") or "External case",
        "patient_source": "uploaded_file",
        "source_identifier": case.get("sample_id") or case.get("sample_title") or "",
        "sample_accession": case.get("sample_accession", ""),
        "age_at_diagnosis": _to_float(case.get("age_years")),
        "tumor_size": None,
        "tumor_stage": case.get("clinical_ajcc_stage") or case.get("clinical_t_stage") or "Unknown",
        "histologic_grade": _to_int(case.get("grade")),
        "er_status": _map_ext_status(case.get("er_status_ihc")),
        "her2_status": _map_ext_status(case.get("her2_status")),
        "pr_status": _map_ext_status(case.get("pr_status_ihc")),
        "lymph_nodes_positive": _map_nodal_status(case.get("clinical_nodal_status")),
        "chemotherapy": "Unknown",
        "hormone_therapy": "Unknown",
        "radiation_therapy": "Unknown",
        "cancer_type": "Breast Cancer",
        "surgery_type": "Unknown",
        "external_metadata": case,
    }


@st.cache_data
def load_external_expression_sample(sample_accession, path = None):
    if not path or not os.path.exists(path) or not sample_accession:
        return None

    with open(path) as f:
        text = f.read()
    return parse_series_matrix_expression(text, sample_accession)


def parse_uploaded_expression_sample(file_bytes, sample_accession):
    text = _decode_uploaded_text(file_bytes)
    if not text:
        return None
    return parse_series_matrix_expression(text, sample_accession)


def parse_series_matrix_expression(text, sample_accession):
    if not text or not sample_accession:
        return None
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("!series_matrix_table_begin"):
            header_idx = i + 1
            break
    if header_idx is None or header_idx >= len(lines):
        return None

    header = next(csv.reader([lines[header_idx]], delimiter="\t"))
    header = [cell.strip().strip('"') for cell in header]
    if sample_accession not in header or "ID_REF" not in header:
        return None

    try:
        sample_idx = header.index(sample_accession)
    except Exception:
        return None

    rows = []
    for line in lines[header_idx + 1 :]:
        if line.startswith("!series_matrix_table_end"):
            break
        row = next(csv.reader([line], delimiter="\t"))
        if len(row) <= sample_idx:
            continue
        probe = str(row[0]).strip().strip('"')
        val = pd.to_numeric(str(row[sample_idx]).strip().strip('"'), errors="coerce")
        if pd.isna(val):
            continue
        rows.append({"ID_REF": probe, "expression": float(val)})

    if not rows:
        return None
    return pd.DataFrame(rows)


def _format_binary(val):
    if pd.isna(val) if not isinstance(val, str) else False:
        return "N/A"
    try:
        return "Yes" if float(val) == 1 else "No"
    except (ValueError, TypeError):
        return str(val) if val else "N/A"


def _clean_columns(columns):
    return (
        columns.str.strip().str.lower()
        .str.replace(" ", "_")
        .str.replace("+", "plus")
        .str.replace("-", "_")
    )


def _read_single_col_csv(path, column_name) :
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    if column_name in df.columns:
        return df[column_name].dropna().astype(str).tolist()
    if df.shape[1] >= 1:
        return df.iloc[:, 0].dropna().astype(str).tolist()
    return []


def _read_numeric_csv(path) :
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    return [int(float(x)) for x in df.iloc[:, 0].dropna().tolist()]


def _safe_joblib_load(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def _profile_to_clinical_lookup(profile) :
    return {
        "age_at_diagnosis": _to_float(profile.get("age_at_diagnosis")),
        "chemotherapy": _to_binary_flag(profile.get("chemotherapy")),
        "hormone_therapy": _to_binary_flag(profile.get("hormone_therapy")),
        "radio_therapy": _to_binary_flag(profile.get("radiation_therapy")),
        "tumor_size": _to_float(profile.get("tumor_size")),
        "lymph_nodes_examined_positive": _to_float(profile.get("lymph_nodes_positive")),
        "er_status": _status_to_binary(profile.get("er_status")),
        "her2_status": _status_to_binary(profile.get("her2_status")),
        "pr_status": _status_to_binary(profile.get("pr_status")),
        "neoplasm_histologic_grade": _to_float(profile.get("histologic_grade")),
    }


def _extract_gene_inputs(external_case) :
    if not external_case:
        return {}
    gene_values = external_case.get("gene_expression")
    return gene_values if isinstance(gene_values, dict) else {}


def _status_to_binary(value):
    if value is None:
        return None
    val = str(value).strip().lower()
    if val in {"positive", "pos", "p", "1", "1.0"}:
        return 1.0
    if val in {"negative", "neg", "n", "0", "0.0"}:
        return 0.0
    return None


def _to_binary_flag(value):
    if value is None:
        return None
    val = str(value).strip().lower()
    if val in {"yes", "true", "1"}:
        return 1.0
    if val in {"no", "false", "0"}:
        return 0.0
    return None


def _to_float(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def _to_int(value):
    num = _to_float(value)
    return int(num) if num is not None else None


def _row_get(sample_rows, key, idx) :
    values = sample_rows.get(key, [])
    return values[idx] if idx < len(values) else ""


def _parse_characteristic(raw):
    text = str(raw).strip().strip('"')
    if not text or ":" not in text:
        return None
    prefix, value = text.split(":", 1)
    key = re.sub(r"[^a-z0-9]+", "_", prefix.strip().lower()).strip("_")
    value = value.strip()
    return (key, value) if key else None


def _external_case_label(row) :
    sample_id = row.get("sample_id") or row.get("sample_title") or row.get("sample_accession") or "Unknown"
    age = row.get("age_years") or "?"
    er = row.get("er_status_ihc") or "?"
    stage = row.get("clinical_ajcc_stage") or row.get("clinical_t_stage") or "?"
    return f"{sample_id} — Age {age}, ER {er}, Stage {stage}"


def _map_ext_status(value):
    val = str(value).strip().upper()
    if val == "P":
        return "Positive"
    if val == "N":
        return "Negative"
    return "Unknown"


def _map_nodal_status(value):
    val = str(value).strip().upper()
    mapping = {"N0": 0, "N1": 1, "N2": 4, "N3": 10}
    return mapping.get(val)


def _decode_uploaded_text(file_bytes):
    if not file_bytes:
        return None
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="ignore")
