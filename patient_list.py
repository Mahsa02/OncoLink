import pandas as pd
import streamlit as st
from datetime import datetime
from auth import (
    get_patients_for_physician,
    get_patient_by_eval_id,
    update_patient_outcome,
    update_patient_profile,
    delete_patient,
    get_outcome_pending_count,
)
from outcome_store import OutcomeStore
from frontend_utils.data_loader import build_new_patient_features, load_raw_data

RETRAIN_THRESHOLD = 10 


def render():
    physician = st.session_state.get("physician")
    if not physician:
        st.warning("Not authenticated.")
        return
    if st.session_state.get("_view_patient_eval_id"):
        _render_patient_detail_view()
        return

    st.markdown("""
    <h2 style='color:#1a3a6b; margin-bottom:0.2rem;'>🗂️ My Patients</h2>
    <p style='color:#556; font-size:0.95rem; margin-top:0;'>
    All the patients you have evaluated. You can confirm real-world outcomes here and once 10 new outcomes are
    confirmed, the prediction model updates automatically.
    </p>
    """, unsafe_allow_html=True)

    patients = get_patients_for_physician(physician["id"])

    if not patients:
        st.info("No patients evaluated yet. Go to **Evaluate New Patient** to get started.")
        return

    _render_stats(patients, physician["id"])
    st.markdown("---")
    _render_patient_cards(patients)


def _render_stats(patients, physician_id):
    total = len(patients)
    with_outcome = sum(1 for p in patients if p.get("true_outcome") is not None)
    pending_outcomes = total - with_outcome
    confirmed_this_cycle = get_outcome_pending_count(physician_id)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", total)
    c2.metric("Outcomes Confirmed", with_outcome)
    c3.metric("Awaiting Outcome", pending_outcomes)
    c4.metric("Until Next Retrain", f"{confirmed_this_cycle}/{RETRAIN_THRESHOLD}")

    if RETRAIN_THRESHOLD > 0:
        pct = min(confirmed_this_cycle / RETRAIN_THRESHOLD, 1.0)
        st.markdown(
            f"""
            <div style='margin:0.6rem 0 0.2rem 0;'>
                <div style='font-size:0.8rem; color:#555; margin-bottom:4px;'>
                    Model update progress — {confirmed_this_cycle} of {RETRAIN_THRESHOLD} new confirmed outcomes
                </div>
                <div style='background:#e9ecef; border-radius:8px; height:8px; overflow:hidden;'>
                    <div style='width:{int(pct*100)}%; background:{"#1a5276" if pct < 1.0 else "#27ae60"};
                         height:100%; border-radius:8px; transition:width 0.5s;'></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if pct >= 1.0:
            st.success("Threshold reached — model will update on next outcome submission.", icon="🔄")


def _render_patient_cards(patients):
    col_f1, col_f2, col_f3 = st.columns([2, 1.5, 1.5])
    with col_f1:
        search = st.text_input("🔍 Search patients", placeholder="Patient name…", label_visibility="collapsed")
    with col_f2:
        outcome_filter = st.selectbox(
            "Outcome filter",
            ["All", "Pending Outcome", "Confirmed"],
            label_visibility="collapsed",
        )
    with col_f3:
        pred_filter = st.selectbox(
            "Prediction filter",
            ["All Predictions", "Lower Risk", "Higher Risk"],
            label_visibility="collapsed",
        )

    filtered = list(patients)
    if search:
        filtered = [p for p in filtered if search.lower() in p["patient_label"].lower()]
    if outcome_filter == "Pending Outcome":
        filtered = [p for p in filtered if p.get("true_outcome") is None]
    elif outcome_filter == "Confirmed":
        filtered = [p for p in filtered if p.get("true_outcome") is not None]
    if pred_filter == "Lower Risk":
        filtered = [p for p in filtered if (p.get("prediction") or {}).get("prediction") == 1]
    elif pred_filter == "Higher Risk":
        filtered = [p for p in filtered if (p.get("prediction") or {}).get("prediction") == 0]

    st.markdown(f"<p style='font-size:0.82rem; color:#8a9ab8;'>{len(filtered)} patient(s) shown</p>", unsafe_allow_html=True)
    for row_start in range(0, len(filtered), 2):
        cols = st.columns(2)
        for col_idx in range(2):
            idx = row_start + col_idx
            if idx >= len(filtered):
                break
            with cols[col_idx]:
                _render_single_card(filtered[idx])


def _render_single_card(patient):
    eval_id = patient["evaluation_id"]
    pred = patient.get("prediction") or {}
    profile = patient.get("profile") or {}
    outcome_val = patient.get("true_outcome")

    prob = pred.get("probability_respond", 0)
    if pred.get("prediction") == 1:
        pred_badge = f"<span class='badge-resp'>Lower Risk ({prob:.0%} survival)</span>"
    elif pred.get("prediction") == 0:
        pred_badge = f"<span class='badge-non'>Higher Risk ({prob:.0%} survival)</span>"
    else:
        pred_badge = "<span class='badge-pending'>No Prediction</span>"

    if outcome_val == 1:
        outcome_badge = "<span class='badge-confirmed'>Outcome: Alive</span>"
    elif outcome_val == 0:
        outcome_badge = "<span class='badge-non'>Outcome: Deceased</span>"
    else:
        outcome_badge = "<span class='badge-pending'>Outcome: Pending</span>"

    er = profile.get("er_status", "—")
    her2 = profile.get("her2_status", "—")
    stage = profile.get("tumor_stage", "—")

    st.markdown(
        f"""
        <div class='result-card'>
            <div class='card-header-row'>
                <div>
                    <div class='card-title'>{patient['patient_label']}</div>
                    <div class='card-date'>Evaluated {_fmt_date(patient.get('created_at'))}</div>
                </div>
            </div>
            <div class='card-badges'>
                {pred_badge}
                {outcome_badge}
            </div>
            <div class='card-detail-row'><strong>ER / HER2:</strong> {er} / {her2}</div>
            <div class='card-detail-row'><strong>Stage:</strong> {stage}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2, b3, b4 = st.columns([2, 2, 1, 1])
    with b1:
        if st.button("View Details", key=f"view_{eval_id}", width="stretch"):
            st.session_state["_view_patient_eval_id"] = eval_id
            st.rerun()
    with b2:
        if st.button("Update Outcome", key=f"outcome_btn_{eval_id}", width="stretch"):
            st.session_state["_outcome_modal_eval_id"] = eval_id
            st.rerun()
    with b3:
        if st.button("✏️", key=f"edit_btn_{eval_id}", width="stretch", help="Edit patient"):
            st.session_state["_edit_modal_eval_id"] = eval_id
            st.rerun()
    with b4:
        if st.button("🗑️", key=f"del_btn_{eval_id}", width="stretch", help="Delete patient"):
            st.session_state["_delete_modal_eval_id"] = eval_id
            st.rerun()
    if st.session_state.get("_outcome_modal_eval_id") == eval_id:
        _render_outcome_dialog(patient)
    if st.session_state.get("_edit_modal_eval_id") == eval_id:
        _render_edit_dialog(patient)
    if st.session_state.get("_delete_modal_eval_id") == eval_id:
        _render_delete_dialog(patient)


def _render_outcome_dialog(patient):
    eval_id = patient["evaluation_id"]
    profile = patient.get("profile") or {}
    pred = patient.get("prediction") or {}
    existing_outcome = patient.get("true_outcome")

    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,#f0f4ff,#e8f0fe); border:1px solid #c5d3f0;
             border-radius:12px; padding:1.2rem 1.5rem; margin:0.5rem 0 1rem 0;'>
            <div style='font-weight:700; font-size:0.95rem; color:#1a3a6b; margin-bottom:0.3rem;'>
                Submit / Update Confirmed Outcome
            </div>
            <div style='color:#556; font-size:0.82rem;'>
                {patient['patient_label']} &nbsp;·&nbsp;
                Prediction: <strong>{pred.get('label','—')}</strong> ({pred.get('probability_respond',0):.0%})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if existing_outcome is not None:
        label = "Alive at follow-up" if existing_outcome == 1 else "Deceased"
        st.info(f"An outcome is already recorded: **{label}**. Submitting again will update the record.")

    col_left, col_right = st.columns(2)
    with col_left:
        true_outcome = st.radio(
            "Confirmed real-world outcome",
            options=[1, 0],
            format_func=lambda x: "✅ Alive at follow-up" if x == 1 else "❌ Deceased",
            index=0 if existing_outcome != 0 else 1,
            key=f"outcome_radio_{eval_id}",
        )
    with col_right:
        notes = st.text_area(
            "Clinical notes (optional)",
            value=patient.get("outcome_notes") or "",
            placeholder="e.g., Completed 6-cycle regimen; partial response confirmed…",
            max_chars=600,
            height=100,
            key=f"notes_{eval_id}",
        )

    btn_submit, btn_cancel = st.columns(2)
    with btn_submit:
        if st.button("Submit Outcome", type="primary", key=f"submit_{eval_id}", width="stretch"):
            with st.spinner("Recording outcome…"):
                db_result = update_patient_outcome(eval_id, true_outcome, notes)
                if not db_result["success"]:
                    st.error(f"Failed to save: {db_result['error']}")
                    return

                pred_val = pred.get("prediction")
                pred_prob = pred.get("probability_respond")
                store = OutcomeStore()
                store_result = store.record_outcome(
                    patient_index=None,
                    evaluation_id=eval_id,
                    true_outcome=true_outcome,
                    predicted_outcome=pred_val,
                    predicted_probability=pred_prob,
                    notes=notes,
                )

            if store_result["success"]:
                st.session_state.pop("_outcome_modal_eval_id", None)
                if store_result.get("retrained"):
                    st.success(store_result["message"], icon="🔄")
                else:
                    pending = store_result.get("new_count", 0)
                    remaining = max(0, RETRAIN_THRESHOLD - pending)
                    st.success(f"Outcome recorded. {remaining} more needed before next model update.", icon="💾")
                st.rerun()
            else:
                st.error(f"Outcome store error: {store_result.get('message','')}")
    with btn_cancel:
        if st.button("Cancel", key=f"cancel_{eval_id}", width="stretch"):
            st.session_state.pop("_outcome_modal_eval_id", None)
            st.rerun()


def _render_edit_dialog(patient):
    eval_id = patient["evaluation_id"]
    profile = patient.get("profile") or {}

    st.markdown(
        f"""<div class='modal-overlay'>
            <div class='modal-title'>✏️ Edit Patient — {patient['patient_label']}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    with st.form(key=f"edit_form_{eval_id}"):
        left, right = st.columns(2)
        with left:
            new_label = st.text_input("Patient Label", value=patient["patient_label"], key=f"ed_label_{eval_id}")
            new_age = st.number_input("Age at Diagnosis", value=int(_safe_num(profile.get("age_at_diagnosis"), 50)), min_value=0, max_value=120, key=f"ed_age_{eval_id}")
            new_stage = st.text_input("Tumor Stage", value=str(profile.get("tumor_stage", "")), key=f"ed_stage_{eval_id}")
            new_grade = st.text_input("Histologic Grade", value=str(profile.get("histologic_grade", "")), key=f"ed_grade_{eval_id}")
        with right:
            new_er = st.selectbox("ER Status", ["Positive", "Negative", "Unknown"], index=_status_idx(profile.get("er_status")), key=f"ed_er_{eval_id}")
            new_her2 = st.selectbox("HER2 Status", ["Positive", "Negative", "Unknown"], index=_status_idx(profile.get("her2_status")), key=f"ed_her2_{eval_id}")
            new_pr = st.selectbox("PR Status", ["Positive", "Negative", "Unknown"], index=_status_idx(profile.get("pr_status")), key=f"ed_pr_{eval_id}")
            new_size = st.number_input("Tumor Size (mm)", value=float(_safe_num(profile.get("tumor_size"), 0)), min_value=0.0, step=1.0, key=f"ed_size_{eval_id}")

        col_save, col_cancel = st.columns(2)
        with col_save:
            save = st.form_submit_button("Save Changes", type="primary", width="stretch")
        with col_cancel:
            cancel = st.form_submit_button("Cancel", width="stretch")

    if save:
        updated_profile = dict(profile)
        updated_profile["age_at_diagnosis"] = new_age
        updated_profile["tumor_stage"] = new_stage.strip() or profile.get("tumor_stage", "")
        updated_profile["histologic_grade"] = new_grade.strip() or profile.get("histologic_grade", "")
        updated_profile["er_status"] = new_er
        updated_profile["her2_status"] = new_her2
        updated_profile["pr_status"] = new_pr
        updated_profile["tumor_size"] = new_size

        result = update_patient_profile(eval_id, new_label.strip() or patient["patient_label"], updated_profile)
        if result["success"]:
            st.session_state.pop("_edit_modal_eval_id", None)
            st.success("Patient updated successfully.", icon="✅")
            st.rerun()
        else:
            st.error(f"Update failed: {result.get('error', '')}")

    if cancel:
        st.session_state.pop("_edit_modal_eval_id", None)
        st.rerun()


def _render_delete_dialog(patient):
    eval_id = patient["evaluation_id"]

    st.markdown(
        f"""<div class='delete-confirm'>
            <div class='modal-title'>🗑️ Delete Patient</div>
            <p style='font-size:0.88rem; color:#555; margin:0;'>
                Are you sure you want to permanently delete
                <strong>{patient['patient_label']}</strong>? This action cannot be undone.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

    col_del, col_cancel = st.columns(2)
    with col_del:
        if st.button("Delete Permanently", key=f"confirm_del_{eval_id}", type="primary", width="stretch"):
            result = delete_patient(eval_id)
            if result["success"]:
                st.session_state.pop("_delete_modal_eval_id", None)
                st.success("Patient deleted.", icon="🗑️")
                st.rerun()
            else:
                st.error(f"Delete failed: {result.get('error', '')}")
    with col_cancel:
        if st.button("Cancel", key=f"cancel_del_{eval_id}", width="stretch"):
            st.session_state.pop("_delete_modal_eval_id", None)
            st.rerun()


def _render_patient_detail_view():
    eval_id = st.session_state.get("_view_patient_eval_id")
    patient = get_patient_by_eval_id(eval_id)

    if st.button("← Back to My Patients", key="back_to_list"):
        st.session_state.pop("_view_patient_eval_id", None)
        st.rerun()
        return

    if patient is None:
        st.error("Patient record not found.")
        return

    profile = patient.get("profile") or {}
    pred = patient.get("prediction") or {}
    outcome_val = patient.get("true_outcome")

    age = _fmt(profile.get("age_at_diagnosis"))
    stage = _fmt(profile.get("tumor_stage"))
    grade = _fmt(profile.get("histologic_grade"))
    size = _fmt(profile.get("tumor_size"))
    nodes = _fmt(profile.get("lymph_nodes_positive"))

    st.markdown(
        f"""
        <div class='patient-card'>
            <h3>{patient['patient_label']} &nbsp;·&nbsp; <span style='font-weight:400;'>Age {age}</span></h3>
            <div class='sub'>
                Tumor Stage {stage} &nbsp;·&nbsp; Grade {grade} &nbsp;·&nbsp;
                Size {size} mm &nbsp;·&nbsp; Lymph nodes positive: {nodes}
            </div>
            <div style='margin-top:0.5rem; font-size:0.82rem; color:#8a9ab8;'>
                Evaluated {_fmt_date(patient.get('created_at'))}
            </div>
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
        st.markdown("**Treatment Context**")
        st.markdown(f"- Chemotherapy: {_tx_badge(profile.get('chemotherapy'))}", unsafe_allow_html=True)
        st.markdown(f"- Hormone Therapy: {_tx_badge(profile.get('hormone_therapy'))}", unsafe_allow_html=True)
        st.markdown(f"- Radiation: {_tx_badge(profile.get('radiation_therapy'))}", unsafe_allow_html=True)
    with c3:
        st.markdown("**Pathology**")
        st.markdown(f"- Surgery: {_fmt(profile.get('surgery_type'))}")
        st.markdown(f"- Cancer Type: {_fmt(profile.get('cancer_type'))}")

    st.markdown("---")
    _render_saved_patient_analysis_tabs(profile, pred, eval_id)

    st.markdown("---")
    st.markdown("<div class='section-header'>Confirmed Outcome</div>", unsafe_allow_html=True)

    if outcome_val == 1:
        st.success("Confirmed: **Alive at last follow-up**.", icon="✅")
    elif outcome_val == 0:
        st.error("Confirmed: **Deceased**.", icon="❌")
    else:
        st.warning("Outcome has not been confirmed yet.", icon="⏳")

    if patient.get("outcome_notes"):
        st.markdown(
            f"""
            <div style='margin-top:0.9rem; font-size:1rem; line-height:1.6; color:#31333f;'>
                <strong>Notes:</strong> {patient['outcome_notes']}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_saved_patient_analysis_tabs(profile,prediction,evaluation_id):
    if not prediction:
        st.info("No prediction is available for this patient.")
        return

    try:
        import evaluate_patient as evaluate_patient_view

        feature_bundle = build_new_patient_features(
            profile,
            external_case=profile.get("external_metadata") or None,
        )
        raw_df = load_raw_data()
    except Exception as exc:
        st.warning(f"Could not restore the full analysis sections for this saved patient: {exc}")
        return

    st.markdown("<div class='section-header'>Predicted Survival Risk</div>", unsafe_allow_html=True)
    evaluate_patient_view._render_prediction(prediction, feature_bundle.get("model_features") if feature_bundle else None)

    st.markdown("---")
    st.markdown("<div class='section-header'>Grounded Clinical Explanation</div>", unsafe_allow_html=True)
    evaluate_patient_view._render_explanation(
        profile,
        prediction,
        feature_bundle,
        evaluation_id,
    )

    st.markdown("---")
    st.markdown("<div class='section-header'>Compare Against Historical Patients</div>", unsafe_allow_html=True)
    evaluate_patient_view._render_similar_patients(raw_df, profile, feature_bundle, evaluation_id, None)


def _fmt_date(iso):
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%b %d, %Y")
    except Exception:
        return iso[:10] if iso else "—"


def _fmt(val):
    if val is None or str(val).lower() in ("nan", "n/a", "none", "", "unknown"):
        return "—"
    try:
        f = float(val)
        return str(int(f)) if f == int(f) else f"{f:.1f}"
    except Exception:
        return str(val)


def _receptor_badge(val):
    v = str(val).strip().lower()
    if v in ("positive", "pos", "p", "1", "1.0"):
        return "<span style='color:#155724; font-weight:600;'>Positive</span>"
    if v in ("negative", "neg", "n", "0", "0.0"):
        return "<span style='color:#721c24; font-weight:600;'>Negative</span>"
    return f"<span style='color:#555;'>{val or 'Unknown'}</span>"


def _tx_badge(val):
    v = str(val).strip().lower()
    if v == "yes":
        return "<span style='color:#155724;'>Yes</span>"
    if v == "no":
        return "<span style='color:#721c24;'>No</span>"
    return "<span style='color:#555;'>Unknown</span>"


def _safe_num(val, default=0):
    try:
        if val is None or str(val).strip().lower() in ("", "nan", "none", "unknown", "n/a"):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def _status_idx(val):
    v = str(val).strip().lower() if val else ""
    if v in ("positive", "pos", "p", "1", "1.0"):
        return 0
    if v in ("negative", "neg", "n", "0", "0.0"):
        return 1
    return 2
