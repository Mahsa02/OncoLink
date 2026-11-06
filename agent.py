
import os
import json
import time
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

GROQ_TIMEOUT = int(os.getenv("GROQ_TIMEOUT_SECONDS", "120"))
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2"))

GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "700"))

RESULTS_PATH = "outputs_metabric/model_results/model_comparison.csv"

MAX_RETRIES = 3
BACKOFF_SECONDS = 3



def _get_client() :
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_key_here":
        return None
    return OpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        timeout=GROQ_TIMEOUT,
    )


def is_groq_available() :
    return bool(GROQ_API_KEY and GROQ_API_KEY != "your_groq_key_here")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_similar_patients",
            "description": "Retrieve similar past patients and outcomes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_index": {"type": "integer"},
                    "k": {"type": "integer", "default": 5},
                },
                "required": ["patient_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_performance",
            "description": "Retrieve model performance table.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _execute_tool(name, args) :
    if name == "get_similar_patients":
        return _tool_similar_patients(args.get("patient_index", 0), args.get("k", 5))
    elif name == "get_model_performance":
        return _tool_model_performance()
    return "Unknown tool"


def _tool_similar_patients(patient_index, k = 5) :
    try:
        from similarity_engine import get_engine
        engine = get_engine()
        summary = engine.get_similar_outcomes_summary(patient_index, k=k)

        if summary["total"] == 0:
            return "Similarity index not built."

        patients = summary.get("patients", [])[:3]

        lines = [
            f"Similar patients: {summary['total']}",
            f"Alive at follow-up: {summary['responders']}",
            f"Deceased: {summary['non_responders']}",
        ]

        for p in patients:
            lines.append(
                f"Patient {p['index']} — {p['outcome_label']} "
                f"(similarity {p['similarity_pct']}%)"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"Error: {e}"


def _tool_model_performance() :
    try:
        df = pd.read_csv(RESULTS_PATH)
        df = df.head(5).round(3)

        return df.to_string(index=False)

    except Exception as e:
        return f"Error: {e}"

def _trim_messages(messages, max_chars=8000):
    """Prevent huge context windows"""
    total = 0
    trimmed = []

    for m in reversed(messages):
        content = m.get("content", "")
        total += len(content)

        if total > max_chars:
            break

        trimmed.append(m)

    return list(reversed(trimmed))



def run_agent(user_prompt, system_prompt = None, max_turns = 4) :
    client = _get_client()
    if client is None:
        return _static_fallback()

    if system_prompt is None:
        system_prompt = (
            "You are an oncology AI assistant. Be concise, clinical, and precise. "
            "The model predicts overall survival status (alive vs. deceased at last follow-up) from the METABRIC cohort; it is not a treatment-response model. "
            "Use tools when helpful. Limit responses to under 150 words."
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt[:2000]},  # 🔥 truncate prompt
    ]

    for _ in range(max_turns):

        messages = _trim_messages(messages)

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=GROQ_TEMPERATURE,
                    max_tokens=GROQ_MAX_TOKENS,
                )
                break

            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < MAX_RETRIES - 1:
                    wait = BACKOFF_SECONDS * (attempt + 1)
                    print(f"Rate limit hit. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    return f"Groq API error: {e}"

        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content or ""

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except:
                args = {}

            result = _execute_tool(tc.function.name, args)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result[:1500],  # 🔥 trim tool output
            })

    return "No response generated."


def run_direct(user_prompt, system_prompt = None) :
    client = _get_client()
    if client is None:
        return _static_fallback()

    if system_prompt is None:
        system_prompt = (
            "You are an oncology AI assistant. Be concise, clinical, and precise. "
            "The model predicts overall survival status (alive vs. deceased at last follow-up) from the METABRIC cohort; it is not a treatment-response model. "
            "Use only the evidence provided in the prompt. Do not invent patient outcomes."
        )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt[:6000]},
                ],
                temperature=GROQ_TEMPERATURE,
                max_tokens=GROQ_MAX_TOKENS,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < MAX_RETRIES - 1:
                wait = BACKOFF_SECONDS * (attempt + 1)
                print(f"Rate limit hit. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return f"Groq API error: {e}"

    return "No response generated."


def explain_prediction(prediction_result,patient_profile,retrieval_context = None,) :
    context_text = ""
    if retrieval_context:
        context_text = f"\nRetrieved evidence:\n{json.dumps(retrieval_context, indent=2, default=str)}\n"

    prompt = (
        f"Patient:\n{patient_profile}\n\n"
        f"Prediction: {prediction_result}\n\n"
        f"{context_text}\n"
        "Explain clinical meaning, similar patients, model-performance context, and limitations. "
        "Use only the retrieved evidence provided; do not invent patient outcomes. "
        "Keep the explanation under 180 words."
    )
    if retrieval_context:
        return run_direct(prompt)
    return run_agent(prompt)


def generate_decision_support(prediction_result, model_info) :
    prompt = (
        f"Prediction: {prediction_result}\n"
        f"Model: {model_info}\n\n"
        "Give clinical guidance in under 120 words."
    )
    return run_agent(prompt)


def explain_model_results(question) :
    return run_agent(f"Answer this clinical ML question: {question}")

def _static_fallback() :
    return (
        "AI assistant unavailable (Groq not configured or rate limited).\n"
        "Use model outputs directly."
    )
if __name__ == "__main__":
    print("Groq available:", is_groq_available())
