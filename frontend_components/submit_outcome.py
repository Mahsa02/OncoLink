import pandas as pd
import streamlit as st

from frontend_utils.new_patient_store import NewPatientStore
from outcome_store import OutcomeStore


def render(current_evaluation = None):
    st.markdown(
        """
        <h2 style='color:#1a3a6b; margin-bottom:0.2rem;'>📝 Submit Confirmed Outcome</h2>
        <p style='color:#556; font-size:0.95rem; margin-top:0;'>
        Use this page after a real treatment course is complete. Confirmed outcomes are stored against the evaluated case and feed the incremental learning loop.
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("How the feedback loop works", expanded=False):
        st.markdown(
            """
            ```
            Physician confirms real-world outcome
                     |
                     v
            outcome_store.py  ->  appends to outcome_feedback.csv
                     |
                     v
              pending_count >= 5?
                Yes -> IncrementalLearner.update(new_patient_features, true_outcome)
                          |
                          v
                   SGDClassifier.partial_fit()
                          |
                          v
                cumulative_samples >= 20?
                  Yes -> promote to best_model.pkl
            ```
            Confirmed outcomes are batched before an update so new evidence enters the model in controlled increments.
            """
        )

    st.markdown("---")

    store = OutcomeStore()
    eval_store = NewPatientStore()
    eval_df = eval_store.list_evaluations()

    if eval_df.empty:
        st.info("No saved new-patient evaluations are available yet. Run an evaluation first.")
        return

    current_evaluation = current_evaluation or st.session_state.get("current_evaluation")
    default_eval_id = current_evaluation.get("evaluation_id") if current_evaluation else eval_store.latest_evaluation_id()
    options = eval_df["evaluation_id"].astype(str).tolist()
    default_index = options.index(default_eval_id) if default_eval_id in options else 0

    selected_eval_id = st.selectbox(
        "Select an evaluated patient record",
        options=options,
        index=default_index,
        format_func=lambda eid: _format_eval_label(eval_store.get_evaluation(eid)),
    )

    selected_eval = eval_store.get_evaluation(selected_eval_id)
    if selected_eval is None:
        st.warning("The selected evaluation could not be loaded.")
        return

    existing = _get_existing(store, selected_eval_id)
    if existing is not None:
        outcome_str = "Alive at follow-up" if existing == 1 else "Deceased"
        st.info(f"A confirmed outcome is already recorded for this evaluation: **{outcome_str}**")
        st.caption("Submitting again will append a corrected or updated record.")

    profile = selected_eval.get("profile", {})
    prediction = selected_eval.get("prediction", {})

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Evaluated Patient Record**")
        st.markdown(f"Evaluation ID: **{selected_eval_id}**")
        st.markdown(f"Patient Label: **{selected_eval.get('patient_label', 'N/A')}**")
        st.markdown(f"Source: **{selected_eval.get('source_type', 'manual').title()}**")
        st.markdown(f"Tumor Stage: **{profile.get('tumor_stage', 'Unknown')}**")
        st.markdown(f"ER / HER2 / PR: **{profile.get('er_status', 'Unknown')} / {profile.get('her2_status', 'Unknown')} / {profile.get('pr_status', 'Unknown')}**")
        if prediction:
            st.markdown(f"Original Prediction: **{prediction.get('label', 'N/A')}**")
            st.markdown(f"Predicted Probability: **{prediction.get('probability_respond', 0):.1%}**")
            st.markdown(f"Model Confidence: **{prediction.get('confidence', 0):.1%}**")

    with col_right:
        st.markdown("**Confirmed Real-World Outcome**")
        true_outcome = st.radio(
            "Submit final observed outcome",
            options=[1, 0],
            format_func=lambda x: "✅ Alive at last follow-up" if x == 1 else "❌ Deceased",
            key=f"outcome_radio_{selected_eval_id}",
        )

    notes = st.text_area(
        "Clinical notes (optional)",
        placeholder="e.g., Completed neoadjuvant regimen with partial response at 8 weeks and confirmed residual disease at surgery...",
        key=f"notes_{selected_eval_id}",
        max_chars=800,
        height=100,
    )

    if st.button("Submit Confirmed Outcome", type="primary", key=f"submit_{selected_eval_id}"):
        with st.spinner("Recording confirmed outcome and checking for model update..."):
            result = store.record_outcome(
                patient_index=None,
                evaluation_id=selected_eval_id,
                true_outcome=true_outcome,
                predicted_outcome=prediction.get("prediction") if prediction else None,
                predicted_probability=prediction.get("probability_respond") if prediction else None,
                notes=notes,
            )

        if result["success"]:
            if result["retrained"]:
                st.success(result["message"], icon="🔄")
            else:
                pending = result["new_count"]
                st.success(
                    f"{result['message']} {pending}/5 new confirmed outcomes until the next incremental update.",
                    icon="💾",
                )
        else:
            st.error(f"Failed to record outcome: {result['message']}")

    st.markdown("---")
    st.markdown("**Feedback System Status**")

    stats = store.get_feedback_stats()
    pending = stats.get("pending_retrain", 0)
    threshold = stats.get("retrain_threshold", 5)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Submitted", stats["total"])
    m2.metric("Alive", stats["responders"])
    m3.metric("Deceased", stats["non_responders"])
    m4.metric("Until Next Update", f"{pending}/{threshold}")

    if threshold > 0:
        pct = min(pending / threshold, 1.0)
        st.markdown(
            f"""
            <div style='margin:0.5rem 0;'>
                <div style='font-size:0.8rem; color:#555; margin-bottom:3px;'>Progress toward next model update ({pending}/{threshold})</div>
                <div style='background:#e9ecef; border-radius:8px; height:8px; overflow:hidden;'>
                    <div style='width:{int(pct*100)}%; background:#1a5276; height:100%; border-radius:8px; transition:width 0.5s;'></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("View All Submitted Outcomes", expanded=False):
        df = store.get_all_feedback()
        if df.empty:
            st.caption("No outcomes submitted yet.")
        else:
            display_cols = [
                "timestamp",
                "evaluation_id",
                "patient_index",
                "true_outcome",
                "predicted_outcome",
                "predicted_probability",
                "notes",
            ]
            available = [c for c in display_cols if c in df.columns]
            st.dataframe(df[available].tail(25), width="stretch", hide_index=True)

    with st.expander("Model Update History", expanded=False):
        try:
            from incremental_learner import IncrementalLearner

            learner = IncrementalLearner()
            history = learner.get_update_history()
            if not history:
                st.caption("No model updates yet — submit 5 confirmed outcomes to trigger the first update.")
            else:
                hist_df = pd.DataFrame(history)
                st.dataframe(hist_df, width="stretch", hide_index=True)
                st.caption(f"Total patient outcomes incorporated into the incremental model: **{learner.total_samples_seen()}**")
        except Exception as exc:
            st.caption(f"Update history unavailable: {exc}")


def _get_existing(store, evaluation_id) :
    df = store.get_all_feedback()
    if df.empty or "evaluation_id" not in df.columns:
        return None
    match = df[df["evaluation_id"].astype(str) == str(evaluation_id)]
    if match.empty:
        return None
    return int(match.iloc[-1]["true_outcome"])


def _format_eval_label(evaluation):
    if not evaluation:
        return "Unavailable evaluation"
    prediction = evaluation.get("prediction", {})
    label = prediction.get("label", "Pending")
    prob = prediction.get("probability_respond", 0)
    return f"{evaluation.get('patient_label', evaluation.get('evaluation_id', 'Case'))} — {label} ({prob:.0%})"
