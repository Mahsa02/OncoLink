import streamlit as st
import pandas as pd
from frontend_utils.agent_bridge import is_groq_available, generate_prediction_explanation
from frontend_utils.data_loader import (
    build_new_patient_features,
    derive_profile_from_external_case,
    get_patient_profile,
    load_best_model,
    load_best_model_info,
    load_model_comparison,
    parse_uploaded_case_metadata,
    predict_feature_row,
    load_processed_features,
    resolve_feature_name_for_model,
    load_survival_model,
    load_calibration,
    predict_survival_curve,
)
from frontend_utils.new_patient_store import NewPatientStore


def render(raw_df):
    _render_header()

    current_eval = st.session_state.get("current_evaluation")
    if current_eval and st.session_state.get("_eval_show_results", False):
        if st.button("← Enter Another Patient", key="reset_eval"):
            st.session_state["_eval_show_results"] = False
            st.session_state.pop("current_evaluation", None)
            st.rerun()
            return

    if current_eval and st.session_state.get("_eval_show_results", False):
        _render_results_view(raw_df, current_eval)
        return


    new_eval = _render_new_patient_intake()
    if new_eval and new_eval.get("evaluation_id"):
        st.session_state["_eval_show_results"] = True
        st.rerun()


def _render_results_view(raw_df, current_eval):
    if "feature_bundle" not in current_eval:
        try:
            selected_case = current_eval.get("profile", {}).get("external_metadata") or None
            feature_bundle = build_new_patient_features(current_eval["profile"], external_case=selected_case)
            current_eval["feature_bundle"] = feature_bundle
            st.session_state["current_evaluation"] = current_eval
        except Exception as exc:
            st.error(f"Could not restore the saved evaluation context: {exc}")
            return

    profile = current_eval["profile"]
    prediction = current_eval["prediction_result"]
    feature_bundle = current_eval["feature_bundle"]
    evaluation_id = current_eval["evaluation_id"]

    _render_profile_card(profile, current_eval)

    if feature_bundle.get("imputed_fields"):
        human_fields = ", ".join(_pretty_name(name) for name in feature_bundle["imputed_fields"])
        st.info(f"Some model inputs were unavailable and were imputed from historical cohort averages: {human_fields}.")

    if feature_bundle.get("genomic_mode") != "matched_external_genes":
        source_note = "uploaded probe IDs could not be mapped into the METABRIC training gene space" if profile.get("patient_source") == "uploaded_file" else "no patient-specific genomic assay was supplied"
        st.caption(
            f"Genomic input note: {source_note}, so OncoLink used a cohort-mean genomic baseline and the physician-entered clinical fields for this evaluation."
        )

    tab_pred, tab_explain, tab_compare = st.tabs([
        "📊 Predicted Survival Risk",
        "🧠 Grounded Clinical Explanation",
        "🔍 Compare Against Historical Patients",
    ])

    with tab_pred:
        _render_prediction(prediction, feature_bundle.get("model_features"))

    with tab_explain:
        summary = _render_explanation(profile, prediction, feature_bundle, evaluation_id)

    with tab_compare:
        _render_similar_patients(raw_df, profile, feature_bundle, evaluation_id, None)


def _render_header():
    col_title, col_badge = st.columns([3, 1])
    with col_title:
        st.markdown(
            """
            <h1 style='margin-bottom:0.1rem; color:#1a3a6b;'>🩺 OncoLink</h1>
            <p style='color:#556; font-size:1rem; margin-top:0;'>
            Evaluate a new patient, compare against historical patients, and later record the patient’s real treatment outcome.
            </p>
            """,
            unsafe_allow_html=True,
        )
    with col_badge:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("🟢 Groq AI active" if is_groq_available() else "🟡 Static mode — add GROQ_API_KEY to .env")

    st.markdown(
        """
        <div class='disclaimer'>
        ⚠️ OncoLink supports physician review of historical analogs and model outputs.
        It is not a standalone diagnostic or treatment device and must be interpreted alongside full clinical evaluation.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_new_patient_intake():
    store = NewPatientStore()
    model = load_best_model()
    model_info = load_best_model_info()

    st.markdown("### Evaluate New Patient")
    st.caption("You can enter a new patient manually or attach a patient file for prefill, then compare that case against historical METABRIC patients.")

    source_mode = st.radio("New patient intake source", ["Manual Entry", "Attach Patient File"], horizontal=True)
    selected_case = None
    uploaded_file_bytes = None
    source_token = "manual"

    if source_mode == "Attach Patient File":
        uploaded_file = st.file_uploader(
            "Attach patient file",
            type=["txt", "tsv", "csv"],
            help="Upload a patient file such as the GSE25066 series matrix to prefill a new-patient evaluation. The file is used only for this evaluation session unless you later submit a confirmed outcome.",
            key="new_patient_upload",
        )
        if uploaded_file is not None:
            uploaded_file_bytes = uploaded_file.getvalue()
            external_df = parse_uploaded_case_metadata(uploaded_file_bytes)
            if external_df is None or external_df.empty:
                st.warning("The uploaded file could not be parsed into patient records.")
            else:
                selected_label = st.selectbox(
                    "Select a patient record from the uploaded file",
                    external_df["display_name"].tolist(),
                    key="uploaded_case_pick",
                )
                selected_case = external_df[external_df["display_name"] == selected_label].iloc[0].to_dict()
                source_token = selected_case.get("sample_accession") or selected_case.get("sample_title") or "upload"
                with st.expander("Attached file details used for prefill", expanded=False):
                    st.json({k: v for k, v in selected_case.items() if k not in {"display_name"}})
                st.info(
                    "The uploaded patient file is not added to the historical training dataset or similarity index. "
                    "It is used only to prefill and support this new-patient evaluation."
                )
        else:
            st.caption("Attach a file to prefill the new patient form. You can still overwrite any field before evaluation.")

    defaults = _default_profile(selected_case)

    if model is None:
        st.error("Trained model not found. Run `python model.py` first.")
        return None

    with st.form("new_patient_evaluation_form"):
        left, right = st.columns(2)

        with left:
            patient_label = st.text_input(
                "Patient Label / MRN Alias",
                value=defaults["patient_label"],
                key=f"patient_label_{source_token}",
                help="Use a de-identified clinical label for this evaluation record.",
            )
            age = st.number_input(
                "Age at Diagnosis",
                min_value=18.0,
                max_value=100.0,
                value=_safe_number_input_value(defaults.get("age_at_diagnosis"), 60.0),
                step=1.0,
                key=f"age_{source_token}",
            )
            tumor_size = st.text_input(
                "Tumor Size (mm)",
                value=_text_or_blank(defaults.get("tumor_size")),
                key=f"tumor_size_{source_token}",
            )
            lymph_nodes = st.text_input(
                "Lymph Nodes Positive",
                value=_text_or_blank(defaults.get("lymph_nodes_positive")),
                key=f"nodes_{source_token}",
            )
            tumor_stage = st.text_input(
                "Tumor Stage",
                value=str(defaults.get("tumor_stage", "Unknown")),
                key=f"stage_{source_token}",
            )
            grade = st.selectbox(
                "Histologic Grade",
                options=["Unknown", "1", "2", "3"],
                index=_select_index(["Unknown", "1", "2", "3"], defaults.get("histologic_grade")),
                key=f"grade_{source_token}",
            )

        with right:
            er_status = st.selectbox(
                "ER Status",
                options=["Positive", "Negative", "Unknown"],
                index=_select_index(["Positive", "Negative", "Unknown"], defaults.get("er_status", "Unknown")),
                key=f"er_{source_token}",
            )
            her2_status = st.selectbox(
                "HER2 Status",
                options=["Positive", "Negative", "Unknown"],
                index=_select_index(["Positive", "Negative", "Unknown"], defaults.get("her2_status", "Unknown")),
                key=f"her2_{source_token}",
            )
            pr_status = st.selectbox(
                "PR Status",
                options=["Positive", "Negative", "Unknown"],
                index=_select_index(["Positive", "Negative", "Unknown"], defaults.get("pr_status", "Unknown")),
                key=f"pr_{source_token}",
            )
            chemotherapy = st.selectbox(
                "Chemotherapy Context",
                options=["Unknown", "Yes", "No"],
                index=_select_index(["Unknown", "Yes", "No"], defaults.get("chemotherapy", "Unknown")),
                key=f"chemo_{source_token}",
            )
            hormone_therapy = st.selectbox(
                "Hormone Therapy Context",
                options=["Unknown", "Yes", "No"],
                index=_select_index(["Unknown", "Yes", "No"], defaults.get("hormone_therapy", "Unknown")),
                key=f"hormone_{source_token}",
            )
            radiation_therapy = st.selectbox(
                "Radiation Therapy Context",
                options=["Unknown", "Yes", "No"],
                index=_select_index(["Unknown", "Yes", "No"], defaults.get("radiation_therapy", "Unknown")),
                key=f"radiation_{source_token}",
            )

        extra_left, extra_right = st.columns(2)
        with extra_left:
            cancer_type = st.text_input(
                "Cancer Type",
                value=str(defaults.get("cancer_type", "Breast Cancer")),
                key=f"cancer_type_{source_token}",
            )
        with extra_right:
            surgery_type = st.text_input(
                "Surgery Type",
                value=str(defaults.get("surgery_type", "Unknown")),
                key=f"surgery_{source_token}",
            )

        submitted = st.form_submit_button("Evaluate New Patient", type="primary")

    if not submitted:
        return None

    profile = {
        "patient_label": patient_label.strip() or defaults["patient_label"],
        "patient_source": "uploaded_file" if selected_case else "manual",
        "source_identifier": (selected_case or {}).get("sample_title", ""),
        "sample_accession": (selected_case or {}).get("sample_accession", ""),
        "age_at_diagnosis": age,
        "tumor_size": _safe_float(tumor_size),
        "tumor_stage": tumor_stage.strip() or "Unknown",
        "histologic_grade": None if grade == "Unknown" else int(grade),
        "er_status": er_status,
        "her2_status": her2_status,
        "pr_status": pr_status,
        "lymph_nodes_positive": _safe_float(lymph_nodes),
        "chemotherapy": chemotherapy,
        "hormone_therapy": hormone_therapy,
        "radiation_therapy": radiation_therapy,
        "cancer_type": cancer_type.strip() or "Breast Cancer",
        "surgery_type": surgery_type.strip() or "Unknown",
        "external_metadata": selected_case or {},
    }

    try:
        feature_bundle = build_new_patient_features(profile, external_case=selected_case)
        prediction = predict_feature_row(model, feature_bundle["model_features"])
    except Exception as exc:
        st.error(f"Could not evaluate this patient: {exc}")
        return None

    if prediction is None:
        st.error("Prediction failed for the entered patient.")
        return None

    evaluation_id = store.save_evaluation(
        patient_label=profile["patient_label"],
        profile=profile,
        prediction_result=prediction,
        all_features=feature_bundle["all_features"],
        source_type=profile["patient_source"],
        source_identifier=profile.get("source_identifier", ""),
        sample_accession=profile.get("sample_accession", ""),
        genomic_mode=feature_bundle.get("genomic_mode", "cohort_mean_fallback"),
        matched_gene_count=feature_bundle.get("matched_gene_count", 0),
        imputed_fields=feature_bundle.get("imputed_fields", []),
    )

    physician = st.session_state.get("physician")
    if physician:
        try:
            from auth import save_patient
            save_patient(
                physician_id=physician["id"],
                evaluation_id=evaluation_id,
                patient_label=profile["patient_label"],
                profile=profile,
                prediction=prediction,
            )
        except Exception:
            pass 

    current_eval = {
        "evaluation_id": evaluation_id,
        "profile": profile,
        "prediction_result": prediction,
        "feature_bundle": feature_bundle,
        "model_info": model_info,
    }
    if uploaded_file_bytes:
        current_eval["uploaded_file_bytes"] = uploaded_file_bytes
    st.session_state["current_evaluation"] = current_eval
    st.success(f"Saved evaluation `{evaluation_id}` for {profile['patient_label']}.")
    return current_eval


def _render_profile_card(profile, current_eval):
    age = _fmt(profile.get("age_at_diagnosis"))
    stage = _fmt(profile.get("tumor_stage"))
    grade = _fmt(profile.get("histologic_grade"))
    size = _fmt(profile.get("tumor_size"))
    nodes = _fmt(profile.get("lymph_nodes_positive"))
    label = profile.get("patient_label", current_eval.get("evaluation_id", "New Patient"))

    st.markdown(
        f"""
        <div class='patient-card'>
            <h3>{label} &nbsp;·&nbsp; <span style='font-weight:400;'>Age {age}</span></h3>
            <div class='sub'>Tumor Stage {stage} &nbsp;·&nbsp; Grade {grade} &nbsp;·&nbsp; Size {size} mm &nbsp;·&nbsp; Lymph nodes positive: {nodes}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Clinical Receptor Status**")
        st.markdown(f"- ER: {_receptor_badge(profile.get('er_status'))}", unsafe_allow_html=True)
        st.markdown(f"- HER2: {_receptor_badge(profile.get('her2_status'))}", unsafe_allow_html=True)
        st.markdown(f"- PR: {_receptor_badge(profile.get('pr_status'))}", unsafe_allow_html=True)
    with c2:
        st.markdown("**Entered Treatment Context**")
        st.markdown(f"- Chemotherapy: {_tx_badge(profile.get('chemotherapy'))}", unsafe_allow_html=True)
        st.markdown(f"- Hormone Therapy: {_tx_badge(profile.get('hormone_therapy'))}", unsafe_allow_html=True)
        st.markdown(f"- Radiation: {_tx_badge(profile.get('radiation_therapy'))}", unsafe_allow_html=True)
    with c3:
        st.markdown("**Pathology & Record Source**")
        st.markdown(f"- Surgery: {_fmt(profile.get('surgery_type'))}")
        st.markdown(f"- Cancer Type: {_fmt(profile.get('cancer_type'))}")
        st.markdown(f"- Source: {_source_label(profile.get('patient_source'))}")


def _render_prediction(prediction, feature_row=None):
    model = load_best_model()
    model_info = load_best_model_info()
    processed = load_processed_features()
    resolved_feature_name = resolve_feature_name_for_model(model_info, processed, model=model)
    col_badge, col_prob, col_conf, col_model = st.columns([2, 1.5, 1.5, 2])

    with col_badge:
        badge_cls = "pred-responder" if prediction["prediction"] == 1 else "pred-non-responder"
        icon = "✅" if prediction["prediction"] == 1 else "❌"
        st.markdown(f"<span class='{badge_cls}'>{icon} {prediction['label']}</span>", unsafe_allow_html=True)
        st.caption("Predicted outcome for this new patient")

    with col_prob:
        st.metric("Survival Probability", f"{prediction['probability_respond']:.1%}")

    with col_conf:
        conf = prediction["confidence"]
        conf_label = "High" if conf > 0.8 else "Moderate" if conf > 0.6 else "Low"
        st.metric("Model Confidence", f"{conf_label} ({conf:.0%})")

    with col_model:
        st.metric("Model", model_info.get("model_name", "N/A"))
        st.caption(f"Features: {resolved_feature_name}")

    prob = prediction["probability_respond"]
    bar_pct = int(prob * 100)
    bar_color = "#38a169" if prob >= 0.5 else "#e53e3e"
    st.markdown(
        f"""
        <div style='margin: 0.6rem 0 0.2rem 0;'>
            <div style='display:flex; justify-content:space-between; font-size:0.8rem; color:#666; margin-bottom:3px;'>
                <span>Higher Risk</span><span>Lower Risk</span>
            </div>
            <div style='background:#f0f0f0; border-radius:8px; height:10px; overflow:hidden;'>
                <div style='width:{bar_pct}%; background:{bar_color}; height:100%; border-radius:8px; transition:width 0.5s;'></div>
            </div>
            <div style='text-align:center; font-size:0.78rem; color:#555; margin-top:3px;'>
                {prob:.1%} predicted probability of being alive at follow-up
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if prob > 0.7:
        st.success("The model estimates a higher likelihood of survival for this new patient.", icon="📈")
    elif prob > 0.5:
        st.info("The model estimates a moderate likelihood of survival. Review this together with tumor biology and treatment intent.", icon="📊")
    elif prob > 0.3:
        st.warning("The model estimates a lower likelihood of survival. Historical analogs and alternative strategies merit close review.", icon="📉")
    else:
        st.error("The model estimates a substantially lower likelihood of survival. Historical outcomes suggest a more challenging course.", icon="⚠️")

    _render_survival_curve(feature_row)
    _render_calibration_note()
    st.caption("Research prototype trained on the METABRIC cohort. Not validated for clinical use.")


def _render_survival_curve(feature_row):
    """Per-patient survival curve from the Random Survival Forest (uses follow-up time)."""
    survival_model, info = load_survival_model()
    if survival_model is None or feature_row is None:
        return
    result = predict_survival_curve(survival_model, feature_row)
    if result is None:
        return

    st.markdown("**Predicted survival over time**")
    c1, c2 = st.columns(2)
    fmt = lambda v: f"{v:.0%}" if v is not None else "—"
    c1.metric("Estimated 5-year survival", fmt(result["five_year"]))
    c2.metric("Estimated 10-year survival", fmt(result["ten_year"]))
    st.line_chart(result["curve"], x="Years", y="Survival Probability", height=240)
    st.caption(
        f"{info.get('model_name', 'Survival model')} trained on {info.get('n_patients', 'the')} METABRIC patients "
        "using follow-up time. Curves describe similar historical patients, not a guarantee for this patient."
    )


def _render_calibration_note():
    cal = load_calibration()
    if not cal:
        return
    st.caption(
        f"Probability calibration on held-out patients: Brier score {cal['brier']:.3f} "
        f"(predicting the cohort average scores {cal['brier_baseline']:.3f}; lower is better)."
    )


def _render_explanation(
    profile,
    prediction,
    feature_bundle,
    evaluation_id,
):
    summary = _get_similarity_summary(feature_bundle, evaluation_id, k=10)
    model_df = load_model_comparison()
    top_models = model_df.head(3).to_dict("records") if model_df is not None else []

    if is_groq_available():
        st.caption("🟢 Grounded explanation generated after retrieving similar historical patients and current model performance")
    else:
        st.caption("🟡 Static explanation with retrieved evidence summary. Configure GROQ_API_KEY for AI-generated grounded text.")

    cache_key = f"explanation_{evaluation_id}"
    retrieval_context = {
        "similarity_summary": summary,
        "top_models": top_models,
    }

    if cache_key in st.session_state and not _is_usable_explanation(st.session_state.get(cache_key)):
        st.session_state.pop(cache_key, None)

    if cache_key not in st.session_state:
        with st.spinner("Generating grounded clinical explanation..."):
            explanation = generate_prediction_explanation(prediction, profile, retrieval_context=retrieval_context)
        st.session_state[cache_key] = explanation
    else:
        explanation = st.session_state[cache_key]

    st.markdown(explanation)

    with st.expander("Why this explanation is grounded", expanded=False):
        st.markdown(
            """
            OncoLink first retrieves:
            1. The most similar historical METABRIC patients for this newly entered case
            2. Their observed survival outcomes (alive / deceased at follow-up)
            3. The current model-performance table used by the app

            The explanatory text is then generated from those retrieved facts rather than from free-form prompting alone.
            """
        )

    if st.button("Regenerate Explanation", key=f"regen_{evaluation_id}"):
        with st.spinner("Regenerating grounded clinical explanation..."):
            explanation = generate_prediction_explanation(prediction, profile, retrieval_context=retrieval_context)
        st.session_state[cache_key] = explanation
        st.rerun()

    return summary


def _is_usable_explanation(explanation) :
    if not explanation or not str(explanation).strip():
        return False
    normalized = str(explanation).strip().lower()
    return normalized not in {"no response generated.", "no response generated"}


def _render_similar_patients(raw_df, profile, feature_bundle, evaluation_id, default_summary):
    try:
        from similarity_engine import get_engine

        engine = get_engine()
    except Exception as exc:
        st.warning(f"Similarity engine unavailable: {exc}")
        return

    if engine.index is None or feature_bundle.get("pca20_features") is None:
        st.warning("Similarity index or PCA-20 patient representation is unavailable. Run `python processing.py` first.")
        return

    max_k = max(1, min(50, len(engine.X_embed) if engine.X_embed is not None else 15))
    col_k, col_info = st.columns([2, 3])
    with col_k:
        k = st.slider(
            "Most Similar Prior Cases to Show",
            min_value=1,
            max_value=max_k,
            value=min(10, max_k),
            step=1,
            key=f"sim_k_{evaluation_id}",
        )
    with col_info:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(f"Searching across {len(engine.X_embed):,} historical patients ranked in the saved PCA-20 similarity space.")

    summary = default_summary if default_summary and default_summary.get("total") == k else _get_similarity_summary(feature_bundle, evaluation_id, k=k)

    if summary["total"] == 0:
        st.warning("No similar historical patients were found.")
        return

    rate = summary.get("response_rate")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Similar Cases", summary["total"])
    m2.metric("Alive", summary["responders"])
    m3.metric("Deceased", summary["non_responders"])
    m4.metric("Historical Survival Rate", f"{rate}%" if rate is not None else "—")

    if rate is not None:
        if rate >= 65:
            st.success(f"{rate}% of the selected similar cohort were alive at last follow-up.", icon="📊")
        elif rate >= 40:
            st.warning(f"{rate}% of the selected similar cohort were alive at last follow-up, suggesting mixed historical outcomes.", icon="📊")
        else:
            st.error(f"Only {rate}% of the selected similar cohort were alive at last follow-up, suggesting a difficult historical pattern.", icon="📊")

    st.markdown("**Most Similar Prior Cases**")
    st.caption("Select a historical case to review full available clinical details.")

    patients = summary.get("patients", [])
    rows = []
    for i, patient in enumerate(patients):
        rows.append(
            {
                "#": i + 1,
                "Historical Patient": f"#{patient['index']}",
                "Similarity": f"{patient['similarity_pct']}%",
                "Outcome": "✅ Alive at follow-up" if patient["outcome"] == 1 else "❌ Deceased",
                "Age": patient.get("age", "—"),
                "ER": patient.get("er_status", "—"),
                "HER2": patient.get("her2_status", "—"),
                "Stage": patient.get("tumor_stage", "—"),
                "Treatment History": patient.get("treatments", "—"),
            }
        )

    df_display = pd.DataFrame(rows)
    st.dataframe(df_display, width="stretch", hide_index=True)

    select_map = {
        f"#{p['index']} — Similarity {p['similarity_pct']}% — {p['outcome_label']}": p
        for p in patients
    }
    selected_label = st.selectbox(
        "Review a similar patient in detail",
        options=list(select_map.keys()),
        key=f"sim_detail_{evaluation_id}",
    )
    if selected_label:
        _render_similar_patient_detail(raw_df, profile, select_map[selected_label])

    with st.expander("How similarity is calculated", expanded=False):
        st.markdown(
            """
            The similarity engine compares the newly entered patient against the historical cohort in the same
            PCA-20 feature space used by the saved model. Distances are converted into a bounded similarity score
            so physicians can inspect the closest prior molecular and clinical analogs.
            """
        )


def _render_similar_patient_detail(raw_df, current_profile, sim_patient):
    idx = sim_patient["index"]
    outcome_label = "✅ Alive at follow-up" if sim_patient["outcome"] == 1 else "❌ Deceased"
    sim_pct = sim_patient["similarity_pct"]

    st.markdown(
        f"""
        <div style='background:#f8faff; border:1px solid #c5d3f0; border-radius:10px; padding:1rem 1.5rem; margin-top:0.5rem;'>
            <div style='font-weight:700; font-size:1rem; color:#1a3a6b; margin-bottom:0.3rem;'>
                Prior Case #{idx} &nbsp;·&nbsp; Similarity {sim_pct}% &nbsp;·&nbsp; {outcome_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if raw_df is None or idx >= len(raw_df):
        st.json(sim_patient)
        return

    hist_profile = get_patient_profile(raw_df, idx)
    if hist_profile is None:
        st.json(sim_patient)
        return

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**Demographics & Tumor**")
        st.markdown(f"- Age: {_fmt(hist_profile['Age at Diagnosis'])}")
        st.markdown(f"- Tumor Size: {_fmt(hist_profile['Tumor Size (mm)'])} mm")
        st.markdown(f"- Stage: {_fmt(hist_profile['Tumor Stage'])}")
        st.markdown(f"- Grade: {_fmt(hist_profile['Histologic Grade'])}")
        st.markdown(f"- Lymph Nodes+: {_fmt(hist_profile['Lymph Nodes Positive'])}")
    with d2:
        st.markdown("**Receptor Status**")
        st.markdown(f"- ER: {_receptor_badge(hist_profile['ER Status'])}", unsafe_allow_html=True)
        st.markdown(f"- HER2: {_receptor_badge(hist_profile['HER2 Status'])}", unsafe_allow_html=True)
        st.markdown(f"- PR: {_receptor_badge(hist_profile['PR Status'])}", unsafe_allow_html=True)
    with d3:
        st.markdown("**Treatment Received**")
        st.markdown(f"- Chemotherapy: {_tx_badge(hist_profile['Chemotherapy'])}", unsafe_allow_html=True)
        st.markdown(f"- Hormone Therapy: {_tx_badge(hist_profile['Hormone Therapy'])}", unsafe_allow_html=True)
        st.markdown(f"- Radiation: {_tx_badge(hist_profile['Radiation Therapy'])}", unsafe_allow_html=True)
        st.markdown(f"- Surgery: {_fmt(hist_profile['Surgery Type'])}")

    st.caption(
        f"Current evaluation: {current_profile.get('patient_label', 'New patient')} · Historical case #{idx} had {sim_pct}% similarity (outcome: {sim_patient['outcome_label'].lower()})."
    )


def _get_similarity_summary(feature_bundle, evaluation_id, k):
    cache_key = f"sim_summary_{evaluation_id}_{k}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    from similarity_engine import get_engine

    engine = get_engine()
    query_vector = feature_bundle["pca20_features"].iloc[0].values
    summary = engine.get_similar_outcomes_summary_for_vector(query_vector, k=k)
    st.session_state[cache_key] = summary
    return summary


def _default_profile(selected_case):
    if selected_case:
        defaults = derive_profile_from_external_case(selected_case)
    else:
        defaults = {
            "patient_label": "New Patient",
            "age_at_diagnosis": 60.0,
            "tumor_size": None,
            "tumor_stage": "Unknown",
            "histologic_grade": None,
            "er_status": "Unknown",
            "her2_status": "Unknown",
            "pr_status": "Unknown",
            "lymph_nodes_positive": None,
            "chemotherapy": "Unknown",
            "hormone_therapy": "Unknown",
            "radiation_therapy": "Unknown",
            "cancer_type": "Breast Cancer",
            "surgery_type": "Unknown",
        }
    return defaults


def _select_index(options, value):
    value = str(value) if value is not None else "Unknown"
    return options.index(value) if value in options else 0


def _text_or_blank(value):
    return "" if value is None else str(value)


def _safe_float(value):
    try:
        text = str(value).strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None


def _safe_number_input_value(value, fallback):
    parsed = _safe_float(value)
    if parsed is None:
        return float(fallback)
    return float(parsed)


def _pretty_name(name):
    mapping = {
        "age_at_diagnosis": "Age",
        "chemotherapy": "Chemotherapy",
        "hormone_therapy": "Hormone Therapy",
        "radio_therapy": "Radiation Therapy",
        "tumor_size": "Tumor Size",
        "lymph_nodes_examined_positive": "Lymph Nodes Positive",
        "er_status": "ER Status",
        "her2_status": "HER2 Status",
        "pr_status": "PR Status",
        "neoplasm_histologic_grade": "Histologic Grade",
    }
    return mapping.get(name, name.replace("_", " ").title())


def _source_label(value):
    return str(value or "manual").replace("_", " ").title()


def _fmt(val) :
    if val is None or str(val).lower() in ("nan", "n/a", "none", "", "unknown"):
        return "—"
    try:
        f = float(val)
        return str(int(f)) if f == int(f) else f"{f:.1f}"
    except Exception:
        return str(val)


def _receptor_badge(val) :
    v = str(val).strip().lower()
    if v in ("positive", "pos", "p", "1", "1.0"):
        return "<span style='color:#155724; font-weight:600;'>Positive</span>"
    if v in ("negative", "neg", "n", "0", "0.0"):
        return "<span style='color:#721c24; font-weight:600;'>Negative</span>"
    return f"<span style='color:#555;'>{val or 'Unknown'}</span>"


def _tx_badge(val) :
    v = str(val).strip().lower()
    if v == "yes":
        return "<span style='color:#155724;'>Yes</span>"
    if v == "no":
        return "<span style='color:#721c24;'>No</span>"
    return "<span style='color:#555;'>Unknown</span>"
