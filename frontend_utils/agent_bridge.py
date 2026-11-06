import streamlit as st
import agent as _agent


def is_groq_available() :
    """Check whether the Groq API key is configured."""
    return _agent.is_groq_available()

def is_ollama_available() :
    return is_groq_available()


def generate_prediction_explanation(prediction_result,patient_profile,retrieval_context = None,) :
    """
    Generate an AI explanation for a patient prediction
    """
    if not is_groq_available():
        return _static_explanation(prediction_result, patient_profile, retrieval_context)
    try:
        explanation = _agent.explain_prediction(
            prediction_result,
            patient_profile,
            retrieval_context=retrieval_context,
        )
        if _is_bad_agent_response(explanation):
            return (
                _static_explanation(prediction_result, patient_profile, retrieval_context)
                + "\n\n*(AI did not return usable text, so OncoLink used the retrieved evidence summary.)*"
            )
        return explanation
    except Exception as e:
        return _static_explanation(prediction_result, patient_profile, retrieval_context) + f"\n\n*(AI error: {e})*"


def generate_decision_support(prediction_result, model_info, model_comparison_df=None) :
    """
    Generate clinical decision support text
    """
    if not is_groq_available():
        return _static_decision_support(prediction_result, model_info)
    try:
        return _agent.generate_decision_support(prediction_result, model_info)
    except Exception as e:
        return _static_decision_support(prediction_result, model_info) + f"\n\n*(AI error: {e})*"


def query_groq(prompt) :
    if not is_groq_available():
        return None
    try:
        return _agent.run_agent(prompt)
    except Exception:
        return None


def _is_bad_agent_response(text) :
    if not text or not str(text).strip():
        return True
    normalized = str(text).strip().lower()
    return (
        normalized == "no response generated."
        or normalized == "no response generated"
        or normalized.startswith("groq api error:")
    )


# Fall back
def _static_explanation(pred, profile, retrieval_context=None) :
    prob = pred["probability_respond"]
    conf = pred["confidence"]
    er = profile.get("ER Status", "N/A")
    her2 = profile.get("HER2 Status", "N/A")
    if er == "N/A":
        er = profile.get("er_status", "N/A")
    if her2 == "N/A":
        her2 = profile.get("her2_status", "N/A")

    if prob > 0.7:
        likelihood = "a higher likelihood"
    elif prob > 0.5:
        likelihood = "a moderate likelihood"
    elif prob > 0.3:
        likelihood = "a lower likelihood"
    else:
        likelihood = "a substantially lower likelihood"

    conf_text = "high" if conf > 0.8 else "moderate" if conf > 0.6 else "low"

    base = (
        f"**Prediction Summary:** The model estimates {likelihood} of survival "
        f"(probability: {prob:.1%}).\n\n"
        f"**Confidence:** {conf_text.capitalize()} confidence ({conf:.1%}). "
        f"{'The model is reasonably certain about this prediction.' if conf > 0.7 else 'Some uncertainty exists — review additional clinical data.'}\n\n"
        f"**Key Factors:** Receptor status (ER: {er}, HER2: {her2}) and gene expression "
        f"patterns are among the strongest predictors of survival in this model.\n\n"
        f"**Limitations:** This prediction is based on the METABRIC training dataset. "
        f"Configure a Groq API key in `.env` for AI-powered explanations with similar patient retrieval.\n\n"
        f"**Disclaimer:** This is decision support only. All treatment decisions must involve qualified clinicians."
    )
    if not retrieval_context:
        return base

    sim = retrieval_context.get("similarity_summary", {})
    if sim.get("total"):
        base += (
            f"\n\n**Retrieved Similar Cohort:** {sim['responders']} of {sim['total']} similar historical patients "
            f"were alive at last follow-up ({sim.get('response_rate', 'N/A')}% survival rate)."
        )
    return base


def _static_decision_support(pred, model_info) :
    prob = pred["probability_respond"]
    conf = pred["confidence"]
    model_name = model_info.get("model_name", "Unknown")
    feature_name = model_info.get("feature_name", "Unknown")

    return (
        f"**Model Performance:** {model_name} using {feature_name} features achieved the highest "
        f"ROC AUC in validation testing.\n\n"
        f"**Clinical Consideration:** The predicted survival probability of {prob:.1%} "
        f"{'indicates a lower-risk profile' if prob > 0.5 else 'indicates a higher-risk profile that warrants close review'}.\n\n"
        f"**Recommended Actions:**\n"
        f"- Correlate with pathology and imaging findings\n"
        f"- Consider genomic profiling (e.g., Oncotype DX)\n"
        f"- Discuss with multidisciplinary tumor board\n"
        f"- Factor in patient preferences and comorbidities\n\n"
        f"**Note:** Configure `GROQ_API_KEY` in `.env` for AI-generated guidance with similar patient context.\n\n"
        f"**Disclaimer:** This tool provides supplementary analytical insight only."
    )
