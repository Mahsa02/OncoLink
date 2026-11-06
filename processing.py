import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR",os.path.join(tempfile.gettempdir(), "oncolink-matplotlib"))
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = "data/METABRIC_RNA_Mutation.csv"
OUTPUT_DIR = "outputs_metabric"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_column_names(df):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("+", "plus")
        .str.replace("-", "_")
    )
    return df


def map_binary(series, pos_vals, neg_vals):
    def convert(x):
        if pd.isna(x):
            return np.nan
        x = str(x).lower().strip()
        if x in pos_vals:
            return 1
        if x in neg_vals:
            return 0
        return np.nan
    return series.map(convert)


print("\n DATA INSPECTION ")

df = pd.read_csv(DATA_PATH, low_memory=False)
df = clean_column_names(df)

print("Raw Shape:", df.shape)
print(df.head())

target_col = "overall_survival"

df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(int)

print("\nClass Distribution:")
print(df[target_col].value_counts())


print("\n CLINICAL FEATURE PROCESSING ")

clinical_candidates = [
    "age_at_diagnosis",
    "chemotherapy",
    "hormone_therapy",
    "radio_therapy",
    "tumor_size",
    "tumor_stage",
    "lymph_nodes_examined_positive",
    "er_status",
    "her2_status",
    "pr_status",
    "neoplasm_histologic_grade",
]

clinical_cols = [c for c in clinical_candidates if c in df.columns]

for col in ["chemotherapy", "hormone_therapy", "radio_therapy"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in ["er_status", "her2_status", "pr_status"]:
    if col in df.columns:
        df[col] = map_binary(df[col],
            ["positive", "pos", "1"],
            ["negative", "neg", "0"]
        )

for col in clinical_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Clinical Features Used:", clinical_cols)

print("\n GENE FEATURE PROCESSING ")

metadata_exclude = {
    "patient_id", "cancer_type", "overall_survival_months",
    "type_of_breast_surgery", "cohort", target_col
}

mutation_cols = [c for c in df.columns if c.endswith("_mut")]

gene_cols = [
    c for c in df.columns
    if c not in metadata_exclude
    and c not in clinical_cols
    and c not in mutation_cols
]

for col in gene_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Number of Gene Features:", len(gene_cols))

# Follow-up time is kept aside (not a model input) so model.py can fit a survival model
survival_months = pd.to_numeric(df["overall_survival_months"], errors="coerce")

print("\n DATA CLEANING ")

threshold = int(0.8 * len(df))

keep_cols = [
    c for c in clinical_cols + gene_cols + [target_col]
    if df[c].notna().sum() >= threshold
]

df = df[keep_cols]

clinical_cols = [c for c in clinical_cols if c in df.columns]
gene_cols = [c for c in gene_cols if c in df.columns]

df[clinical_cols] = df[clinical_cols].fillna(df[clinical_cols].mean())
df[gene_cols] = df[gene_cols].fillna(df[gene_cols].mean())

df = df.drop_duplicates()

print("Cleaned Shape:", df.shape)


print("\n FEATURE MATRICES ")

X_genes = df[gene_cols]
X_clinical = df[clinical_cols]
X_all = pd.concat([X_genes, X_clinical], axis=1)
y = df[target_col]

print("Genes:", X_genes.shape)
print("Clinical:", X_clinical.shape)
print("Combined:", X_all.shape)


print("\n PREPROCESSING ")

scaler_genes = StandardScaler()
X_genes_scaled = scaler_genes.fit_transform(X_genes)

scaler_clinical = StandardScaler()
X_clinical_scaled = scaler_clinical.fit_transform(X_clinical)

X_all_scaled = np.concatenate([X_genes_scaled, X_clinical_scaled], axis=1)


print("\n FEATURE ENGINEERING ")

num_top = min(1000, X_genes.shape[1])
variances = np.var(X_genes_scaled, axis=0)
top_idx = np.argsort(variances)[-num_top:]

X_top_genes = X_genes_scaled[:, top_idx]
X_top = np.concatenate([X_top_genes, X_clinical_scaled], axis=1)

X_clinical_only = X_clinical_scaled

pca_20 = PCA(n_components=20, random_state=42)
pca_50 = PCA(n_components=50, random_state=42)
pca_var = PCA(n_components=0.95, random_state=42)

X_pca_20 = np.concatenate([pca_20.fit_transform(X_genes_scaled), X_clinical_scaled], axis=1)
X_pca_50 = np.concatenate([pca_50.fit_transform(X_genes_scaled), X_clinical_scaled], axis=1)
X_pca_var = np.concatenate([pca_var.fit_transform(X_genes_scaled), X_clinical_scaled], axis=1)

print("PCA 20 Variance:", pca_20.explained_variance_ratio_.sum())
print("PCA 50 Variance:", pca_50.explained_variance_ratio_.sum())
print("PCA 95% Components:", pca_var.n_components_)


print("\n SAVING OUTPUTS ")
# Save gene column names for downstream use
pd.Series(gene_cols).to_csv(f"{OUTPUT_DIR}/gene_column_names.csv", index=False, header=["gene_name"])
pd.Series(clinical_cols).to_csv(f"{OUTPUT_DIR}/clinical_column_names.csv", index=False, header=["clinical_name"])
pd.Series([str(i) for i in top_idx]).to_csv(f"{OUTPUT_DIR}/top_gene_indices.csv", index=False, header=["index"])

pd.DataFrame(X_all_scaled, columns=list(gene_cols) + list(clinical_cols)).to_csv(f"{OUTPUT_DIR}/X_all_genes.csv", index=False)
pd.DataFrame(X_top).to_csv(f"{OUTPUT_DIR}/X_top_variable_genes.csv", index=False)
pd.DataFrame(X_pca_20).to_csv(f"{OUTPUT_DIR}/X_pca_20.csv", index=False)
pd.DataFrame(X_pca_50).to_csv(f"{OUTPUT_DIR}/X_pca_50.csv", index=False)
pd.DataFrame(X_pca_var).to_csv(f"{OUTPUT_DIR}/X_pca_95_var.csv", index=False)
pd.DataFrame(X_clinical_scaled, columns=clinical_cols).to_csv(f"{OUTPUT_DIR}/X_clinical.csv", index=False)
y.to_csv(f"{OUTPUT_DIR}/y_labels.csv", index=False)
survival_months.loc[df.index].rename("overall_survival_months").to_csv(f"{OUTPUT_DIR}/survival_months.csv", index=False)

df[gene_cols + clinical_cols + [target_col]].to_csv(f"{OUTPUT_DIR}/raw_features_unscaled.csv", index=False)

joblib.dump(scaler_genes, f"{OUTPUT_DIR}/scaler_genes.pkl")
joblib.dump(scaler_clinical, f"{OUTPUT_DIR}/scaler_clinical.pkl")
joblib.dump(pca_20, f"{OUTPUT_DIR}/pca_20.pkl")
joblib.dump(pca_50, f"{OUTPUT_DIR}/pca_50.pkl")
joblib.dump(pca_var, f"{OUTPUT_DIR}/pca_95_var.pkl")


print("\n SUMMARY ")

summary = {
    "Total Samples": len(y),
    "Number of Gene Features": X_genes.shape[1],
    "Number of Clinical Features": X_clinical.shape[1],
    "Total Combined Features": X_all.shape[1],
    "Alive at follow-up (label=1)": int((y == 1).sum()),
    "Deceased (label=0)": int((y == 0).sum()),
    "PCA 95pct Components": int(pca_var.n_components_),
}

for k, v in summary.items():
    print(f"{k}: {v}")


with open(f"{OUTPUT_DIR}/dataset_summary.txt", "w") as f:
    for k, v in summary.items():
        f.write(f"{k}: {v}\n")

plt.figure()
y.value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.savefig(f"{OUTPUT_DIR}/class_distribution.png")
plt.close()

plt.figure()
plt.plot(np.cumsum(pca_var.explained_variance_ratio_))
plt.title("PCA Explained Variance")
plt.savefig(f"{OUTPUT_DIR}/pca_explained_variance.png")
plt.close()

print("\nProcessing complete.")
