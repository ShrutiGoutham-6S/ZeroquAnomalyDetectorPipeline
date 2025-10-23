"""
model_trainer.py
----------------
Trains RandomForest models for vessel performance anomaly detection
and saves model + metadata files into /models folder.
"""

import os
import json
import joblib
import shap
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# =============================
# STEP 1: Load & Prepare Data
# =============================
FILE_PATH = "SOL_RELIANCE.xlsx"
SHEET_NAME = "Report"
DATE_COL = "Reporting Period [UTC]"

df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, skiprows=[0,1,2,3,4,6])
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
df = df.sort_values(DATE_COL).dropna(subset=[DATE_COL])

recent_data = df[df[DATE_COL] >= (df[DATE_COL].max() - pd.Timedelta(days=90))]
print(f"Data window: {recent_data[DATE_COL].min().date()} → {recent_data[DATE_COL].max().date()} ({len(recent_data)} rows)")

# =============================
# STEP 2: Define Targets & Features
# =============================
targets_features = {
    'Shaft Power': {
        'target': 'Main Engine Average Shaft Power [kW] - MainEngine1',
        'features': [
            'Main Engine Average Revolutions Per Minutes [RPM] - MainEngine1',
            'Main Engine Torsion Meter - MainEngine1',
            'Main Engine Runtime - MainEngine1',
            'Cargo Weight [MT]',
            'Main Engine Consumption [MT] - VLS',
            'Speed Logged [KN]'
        ]
    },
    'RPM': {
        'target': 'Main Engine Average Revolutions Per Minutes [RPM] - MainEngine1',
        'features': [
            'Main Engine Torsion Meter - MainEngine1',
            'Main Engine Runtime - MainEngine1',
            'Main Engine Average Shaft Power [kW] - MainEngine1',
            'Cargo Weight [MT]',
            'Speed Logged [KN]'
        ]
    },
    'Scavenge Temp': {
        'target': 'Main Engine Scavenge Air Temperature [°C] - MainEngine1',
        'features': [
            'Main Engine Air Temperature Inlet To Turbocharger [°C] - MainEngine1',
            'Main Engine Ambient Air Pressure [bar] - MainEngine1',
            'Main Engine SFOC',
            'Main Engine Average Revolutions Per Minutes [RPM] - MainEngine1',
            'ME Jacket Water Outlet Temperature [°C]',
            'Main Engine Average Shaft Power [kW] - MainEngine1',
            'Speed Logged [KN]'
        ]
    }
}

# =============================
# STEP 3: Train Models
# =============================
results_summary = {}
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

for label, tf in targets_features.items():
    print(f"\n===== {label.upper()} MODEL TRAINING =====")
    required_cols = tf['features'] + [tf['target']]
    data_model = recent_data.dropna(subset=required_cols)

    X = data_model[tf['features']]
    y = data_model[tf['target']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_split=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    errors = np.abs(y_test - y_pred)

    # Define IQR threshold for anomalies
    q1, q3 = np.percentile(errors, [25, 75])
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr

    print(f"MAE: {mae:.3f} | R²: {r2:.3f} | Threshold: {threshold:.3f}")

    # Save model and metadata
    model_file = models_dir / f"{label.replace(' ', '_')}_rf_model.pkl"
    meta_file = models_dir / f"{label.replace(' ', '_')}_metadata.pkl"

    joblib.dump(model, model_file)
    joblib.dump({
        "features": tf['features'],
        "threshold": threshold
    }, meta_file)

    print(f"✅ Saved {label} model → {model_file.name}")
    print(f"✅ Saved {label} metadata → {meta_file.name}")

print("\nAll models trained and saved successfully.")
