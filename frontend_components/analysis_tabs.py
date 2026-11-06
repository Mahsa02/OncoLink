import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import inspect
import os
import shap

from frontend_utils.data_loader import (
    load_best_model, load_best_model_info, load_processed_features,
    get_feature_matrix_for_model, resolve_feature_name_for_model,
    load_shap_data, load_model_comparison,
    get_chart_paths, get_output_chart_paths, parse_uploaded_expression_sample,
    RESULTS_DIR, OUTPUTS_DIR,
)
from frontend_utils.new_patient_store import NewPatientStore


def render(current_evaluation = None):
    st.markdown("""
    <h2 style='color:#1a3a6b; margin-bottom:0.2rem;'>📊 Analysis & Model Insights</h2>
    <p style='color:#556; font-size:0.95rem; margin-top:0;'>
    Deeper analysis tools — feature attribution, gene expression, and model performance metrics.
    </p>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "🧪 Feature Importance & SHAP",
        "🧬 Gene Expression",
        "📈 Model Comparison",
    ])

    with tab1:
        _render_feature_importance()

    with tab2:
        _render_gene_expression(current_evaluation)

    with tab3:
        _render_model_comparison()

def _render_feature_importance():
    st.subheader("Feature Importance & SHAP Explainability")
    st.caption("Which genomic and clinical features most influence survival risk predictions.")

    model = load_best_model()
    model_info = load_best_model_info()
    processed = load_processed_features()

    if model is None:
        st.warning("Trained model not found. Run `python model.py` first.")
        return

    X = get_feature_matrix_for_model(model_info, processed, model=model)
    resolved_feature_name = resolve_feature_name_for_model(model_info, processed, model=model)
    if X is None:
        st.warning("Feature data not available.")
        return

    if resolved_feature_name != model_info.get("feature_name", "Unknown"):
        st.caption(
            f"Detected feature-set mismatch in saved metadata. Using the actual model input shape, resolved as: **{resolved_feature_name}**."
        )

    shap_tab, fi_tab = st.tabs(["SHAP Summary", "Feature Importances"])

    with shap_tab:
        _shap_section(model_info)

    with fi_tab:
        _fi_section(model, model_info, X, processed)


def _shap_section(model_info):
    shap_plot = os.path.join(RESULTS_DIR, "shap_summary.png")
    shap_vals, feat_names = load_shap_data()

    if shap_vals is None:
        st.info(
            "SHAP values not yet computed.\n\n"
            "Run `python model.py` with `shap` installed: `pip install shap`"
        )
        return

    st.markdown(
        "**SHAP** (SHapley Additive exPlanations) — red pushes toward Lower Risk (survival), "
        "blue pushes toward Higher Risk. Width shows magnitude."
    )

    if os.path.exists(shap_plot):
        st.image(shap_plot, caption="SHAP Summary — Top 20 Features", width=520)
    else:
        try:
            plt.figure(figsize=(7,5))
            shap.summary_plot(shap_vals, feature_names=feat_names, plot_type="bar", show=False, max_display=20)
            fig = plt.gcf()
            fig.set_size_inches(7,5)
            #plt.tight_layout()
            st.pyplot(fig, width="content")
            plt.close(fig)
        except ImportError:
            st.warning("Install `shap` to render the summary plot.")

    if feat_names:
        mean_abs = np.abs(shap_vals)
        if mean_abs.ndim == 3:
            mean_abs = mean_abs.mean(axis=2)
        mean_abs = mean_abs.mean(axis=0)
        top_idx = np.argsort(mean_abs)[-10:][::-1]
        feat_names = np.array(feat_names)
        top_df = pd.DataFrame({
            "Feature": [feat_names[i] for i in top_idx],
            "Mean |SHAP|": [round(float(mean_abs[i]), 5) for i in top_idx],
        })
        st.markdown("**Top 10 by mean |SHAP value|**")
        st.dataframe(top_df, width="stretch", hide_index=True)


def _fi_section(model, model_info, X, processed):
    if not hasattr(model, "feature_importances_"):
        st.info(
            f"Feature importances not available for {model_info.get('model_name','this model')}. "
            "Requires a tree-based model (Random Forest or XGBoost)."
        )
        return

    importances = model.feature_importances_
    feat_names = list(X.columns) if hasattr(X, "columns") else [f"F{i}" for i in range(len(importances))]

    imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=False)

    n = st.slider("Top features to display", 5, min(50, len(imp_df)), 20, 5, key="fi_slider")
    top_df = imp_df.head(n).copy()

    clinical_set = _get_clinical_set(X, processed)
    top_df["Type"] = top_df["Feature"].apply(lambda f: "Clinical" if f in clinical_set else "Gene / PCA")

    fig, ax = plt.subplots(figsize=(8, max(3.5, n * 0.25)))
    colors = top_df["Type"].map({"Clinical": "#2ecc71", "Gene / PCA": "#3498db"}).values
    ax.barh(range(len(top_df)), top_df["Importance"].values, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df["Feature"].values, fontsize=8)
    ax.set_xlabel("Importance Score", fontsize=9)
    ax.set_title(f"Top {n} Feature Importances — {model_info.get('model_name','Model')}", fontsize=10)
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#2ecc71", label="Clinical"), Patch(facecolor="#3498db", label="Gene / PCA")], loc="lower right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, width="content")
    plt.close()

    total_imp = imp_df["Importance"].sum()
    clin_imp = imp_df[imp_df["Feature"].isin(clinical_set)]["Importance"].sum()
    gene_imp = total_imp - clin_imp
    c1, c2 = st.columns(2)
    c1.metric("Clinical Feature Contribution", f"{clin_imp/total_imp:.1%}" if total_imp else "N/A")
    c2.metric("Gene / PCA Contribution", f"{gene_imp/total_imp:.1%}" if total_imp else "N/A")


def _get_clinical_set(X, processed):
    xc = processed.get("X_clinical")
    if xc is None:
        return set()
    if hasattr(xc, "columns"):
        return set(xc.columns)
    n = xc.shape[1]
    cols = list(X.columns) if hasattr(X, "columns") else []
    return set(cols[-n:]) if len(cols) > n else set(cols)

def _render_gene_expression(current_evaluation):
    st.subheader("Gene Expression Profile")

    from frontend_utils.data_loader import load_raw_data, load_feature_metadata

    raw_df = load_raw_data()
    if raw_df is None or raw_df.empty:
        st.warning("METABRIC dataset not found. Run `python processing.py` first.")
        return

    meta = load_feature_metadata()
    gene_cols = meta.get("gene_columns", [])
    available_gene_cols = [c for c in gene_cols if c in raw_df.columns]

    if not available_gene_cols:
        st.info("No gene expression columns found in the dataset.")
        return

    st.caption(
        f"Explore per-patient gene expression from the METABRIC cohort ({len(raw_df)} patients, "
        f"{len(available_gene_cols)} gene features). Select a patient and configure the view below."
    )

    patient_ids = raw_df.get("patient_id")
    if patient_ids is None:
        patient_ids = pd.Series([f"Patient-{i}" for i in range(len(raw_df))])

    col_sel, col_n, col_sort = st.columns([2, 1, 1])
    with col_sel:
        idx = st.selectbox(
            "Select patient",
            range(len(raw_df)),
            format_func=lambda i: f"{patient_ids.iloc[i]}  (idx {i})",
            key="ge_patient_idx",
        )
    with col_n:
        n_show = st.slider("Top N genes", 10, min(100, len(available_gene_cols)), 25, 5, key="ge_n_slider")
    with col_sort:
        sort_order = st.radio("Sort by", ["Highest", "Lowest", "Absolute"], horizontal=True, key="ge_sort")

    col_viz, _ = st.columns([1, 2])
    with col_viz:
        viz = st.radio("Chart type", ["Bar Chart", "Heatmap"], horizontal=True, key="ge_viz")

    patient_row = raw_df.iloc[idx]
    gene_data = patient_row[available_gene_cols].apply(pd.to_numeric, errors="coerce").dropna()
    patient_label = str(patient_ids.iloc[idx])

    if gene_data.empty:
        st.warning("No numeric gene expression data available for this patient.")
        return

    if sort_order == "Highest":
        top_genes = gene_data.nlargest(n_show)
    elif sort_order == "Lowest":
        top_genes = gene_data.nsmallest(n_show)
    else:
        top_genes = gene_data.abs().nlargest(n_show)
        top_genes = gene_data[top_genes.index]

    if viz == "Bar Chart":
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in top_genes.values]
        fig, ax = plt.subplots(figsize=(8, max(3.5, n_show * 0.22)))
        ax.barh(range(len(top_genes)), top_genes.values, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(top_genes)))
        ax.set_yticklabels(top_genes.index, fontsize=7)
        ax.set_xlabel("Expression Level", fontsize=9)
        ax.set_title(f"Top {n_show} Genes — {patient_label} ({sort_order})", fontsize=10)
        ax.invert_yaxis()
        ax.axvline(0, color="#999", linewidth=0.5, linestyle="--")
        plt.tight_layout()
        st.pyplot(fig, width="content")
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(10, 2.2))
        im = ax.imshow(top_genes.values.reshape(1, -1), aspect="auto", cmap="RdYlGn", interpolation="nearest")
        ax.set_xticks(range(len(top_genes)))
        ax.set_xticklabels(top_genes.index, rotation=90, fontsize=6)
        ax.set_yticks([0])
        ax.set_yticklabels([patient_label], fontsize=9)
        ax.set_title(f"Gene Expression Heatmap — Top {n_show} ({sort_order})", fontsize=10)
        plt.colorbar(im, ax=ax, label="Expression", shrink=0.8)
        plt.tight_layout()
        st.pyplot(fig, width="content")
        plt.close()

   
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Gene Features", f"{len(gene_data)}")
    c2.metric("Mean Expression", f"{gene_data.mean():.3f}")
    c3.metric("Std Deviation", f"{gene_data.std():.3f}")
    c4.metric("Max Expression", f"{gene_data.max():.3f}")

def _render_model_comparison():
    st.subheader("Model Comparison Dashboard")
    st.caption("Performance metrics across all trained model and feature-set combinations.")

    comparison_df = load_model_comparison()
    model_info = load_best_model_info()
    model = load_best_model()
    processed = load_processed_features()

    if comparison_df is None:
        st.warning("Model comparison results not found. Run `python model.py` first.")
        return

    best_name = model_info.get("model_name", "")
    best_feat = resolve_feature_name_for_model(model_info, processed, model=model)
    st.success(
        f"**Best Model:** {best_name} with {best_feat} features "
        f"(ROC AUC: {comparison_df.iloc[0]['ROC_AUC']:.4f})",
        icon="🏆",
    )

    disp = comparison_df.copy()
    disp.insert(0, "Rank", range(1, len(disp) + 1))
    for col in ["Accuracy", "F1", "ROC_AUC"]:
        if col in disp.columns:
            disp[col] = disp[col].apply(lambda x: f"{x:.4f}")
    st.dataframe(disp, width="stretch", hide_index=True)

    charts = get_chart_paths()
    output_charts = get_output_chart_paths()

    if charts["roc"]:
        st.markdown("#### ROC Curves")
        keys = sorted(charts["roc"].keys())
        n_cols = min(3, len(keys))
        cols = st.columns(n_cols)
        for i, k in enumerate(keys):
            with cols[i % n_cols]:
                _safe_image(charts["roc"][k], k.replace("_", " ").title())

    if charts["confusion"]:
        st.markdown("#### Confusion Matrices")
        keys = sorted(charts["confusion"].keys())
        n_cols = min(3, len(keys))
        cols = st.columns(n_cols)
        for i, k in enumerate(keys):
            with cols[i % n_cols]:
                _safe_image(charts["confusion"][k], k.replace("_", " ").title())

    if output_charts:
        st.markdown("#### Dataset Overview")
        n_cols = min(3, len(output_charts))
        cols = st.columns(n_cols)
        for i, (name, path) in enumerate(output_charts.items()):
            with cols[i % n_cols]:
                _safe_image(path, name.replace("_", " ").title())


def _safe_image(path, label):
    if not os.path.exists(path):
        st.caption(f"Chart not found: {os.path.basename(path)}")
        return
    try:
        st.image(path, caption=label, **_img_kwargs())
    except Exception as e:
        st.caption(f"Could not render {os.path.basename(path)}: {e}")


def _img_kwargs():
    params = inspect.signature(st.image).parameters
    return {"width": "stretch"} if "width" in params else {"use_column_width": True}
