"""
integ.py
--------
Loads trained models and metadata, performs anomaly detection on a sample input,
explains via SHAP, and optionally queries Groq LLM for reasoning.
Outputs structured JSON.
"""

import os
import json
import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# =============================
# STEP 0: Load Environment
# =============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

try:
    from groq import GroqClient
    groq_client = GroqClient(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception:
    groq_client = None


# =============================
# STEP 1: Load Trained Models
# =============================
def load_all_models():
    targets = ["Shaft Power", "RPM", "Scavenge Temp"]
    models, metadata = {}, {}
    base_dir = Path(__file__).resolve().parent / "models"

    for label in targets:
        model_path = base_dir / f"{label.replace(' ', '_')}_rf_model.pkl"
        meta_path = base_dir / f"{label.replace(' ', '_')}_metadata.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")

        models[label] = joblib.load(model_path)
        metadata[label] = joblib.load(meta_path)

    return models, metadata


# =============================
# STEP 2: Define Static Input
# =============================
sample = {
    'Main Engine Average Revolutions Per Minutes [RPM] - MainEngine1': 5200.0,
    'Main Engine Torsion Meter - MainEngine1': 500.0,
    'Main Engine Runtime - MainEngine1': 12.0,
    'Cargo Weight [MT]': 22000.0,
    'Main Engine Consumption [MT] - VLS': 2.0,
    'Main Engine Average Shaft Power [kW] - MainEngine1': 25000.0,
    'Main Engine Air Temperature Inlet To Turbocharger [°C] - MainEngine1': 40.0,
    'Main Engine Ambient Air Pressure [bar] - MainEngine1': 1.01,
    'Main Engine SFOC': 267.31079,
    'Exh. Uptake Temperature [°C]': 300.0,
    'Speed Logged [KN]': 15.0,
    'LT Water Temperature In to Scavenge Cooler [°C]': 35.0,
    'ME Jacket Water Outlet Temperature [°C]': 75.0,
    'Main Engine Scavenge Air Temperature [°C] - MainEngine1': 40.0
}


# =============================
# STEP 3: Run Predictions
# =============================
def analyze_sample(sample: dict):
    models, metadata = load_all_models()
    sample_df = pd.DataFrame([sample])

    key_map = {
        'Shaft Power': 'Main Engine Average Shaft Power [kW] - MainEngine1',
        'RPM': 'Main Engine Average Revolutions Per Minutes [RPM] - MainEngine1',
        'Scavenge Temp': 'Main Engine Scavenge Air Temperature [°C] - MainEngine1'
    }

    results = []

    for label in models:
        model = models[label]
        features = metadata[label]['features']
        threshold = metadata[label]['threshold']

        X = sample_df[features]
        predicted = model.predict(X)[0]
        actual = sample.get(key_map[label])

        if actual is None:
            continue

        diff = abs(predicted - actual)
        anomaly = diff > threshold

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        top_idx = np.argmax(np.abs(shap_values[0]))
        top_feature = features[top_idx]
        shap_value = shap_values[0][top_idx]

        # Severity level
        if diff > 2 * threshold:
            severity = "High"
        elif diff > threshold:
            severity = "Medium"
        else:
            severity = "Low"

        # Groq LLM reasoning
        prompt = (
            f"Anomaly detected in {label}: Predicted={predicted:.2f}, Actual={actual:.2f}, "
            f"Diff={diff:.2f}, Threshold={threshold:.2f}. "
            f"Top contributing factor: {top_feature} (SHAP={shap_value:.2f}). "
            "Give a 1-sentence summary explaining why this anomaly might occur."
        )

        if groq_client:
            try:
                resp = groq_client.request(prompt)
                summary = getattr(resp, 'text', str(resp))
            except Exception:
                summary = f"The anomaly in {label} is likely due to unusual {top_feature} values."
        else:
            summary = f"The anomaly in {label} is likely due to unusual {top_feature} values."

        results.append({
            "column": label,
            "predicted": round(float(predicted), 2),
            "actual": round(float(actual), 2),
            "anomaly": bool(anomaly),
            "severity": severity,
            "top_feature": top_feature,
            "shap_value": round(float(shap_value), 2),
            "reason_summary": summary
        })

    return results


# =============================
# STEP 4: Execute & Output
# =============================
if __name__ == "__main__":
    output = analyze_sample(sample)
    print(json.dumps(output, indent=4))
