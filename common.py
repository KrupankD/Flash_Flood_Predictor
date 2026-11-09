"""Shared helpers used by both train.py and app.py."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "haor_flood_events_v1.csv"
MODEL_DIR = ROOT / "models"
MODEL_NAMES = ["Logistic Regression", "Random Forest"]
METRICS_PATH = ROOT / "models" / "metrics.json"
CV_RESULTS_PATH = ROOT / "models" / "cv_results.csv"
IMPORTANCE_PATH = ROOT / "models" / "feature_importance.csv"

TARGET = "flood_label"

# Columns removed before training, with the reason:
#  - event_id, date : identifiers, not predictive
#  - source         : free text from the labelling step; it leaks the answer
#                     (e.g. "FFWC dry season" is always 0)
#  - slope, twi     : constant for the whole haor study area (no information)
DROP_COLS = ["event_id", "date", "source", "slope", "twi"]

# Model inputs, in a fixed order. vv_vh_ratio is derived from VV / VH.
FEATURES = [
    "VV", "VH", "vv_vh_ratio", "upstream_vv", "ndwi",
    "rainfall", "rain_24h", "rain_96h",
    "soil_moisture", "temp", "temp_anomaly", "wind",
]

# Friendly names + help text for the Streamlit form
FEATURE_INFO = {
    "VV": ("Sentinel-1 VV backscatter (dB)", "Radar signal over the haor. Lower (more negative) = smoother surface, e.g. open water."),
    "VH": ("Sentinel-1 VH backscatter (dB)", "Cross-polarised radar signal over the haor."),
    "upstream_vv": ("Upstream VV, Barak River (dB)", "Radar signal at the upstream reference area (Silchar, India)."),
    "ndwi": ("NDWI (Sentinel-2)", "Water index. Higher = more surface water. Often missing under cloud cover."),
    "rainfall": ("Rainfall (mm)", "Rainfall on the event date."),
    "rain_24h": ("Rainfall, last 24 h (mm)", "ERA5-Land accumulation."),
    "rain_96h": ("Rainfall, last 96 h (mm)", "ERA5-Land accumulation over 4 days."),
    "soil_moisture": ("Soil moisture", "ERA5-Land antecedent soil moisture."),
    "temp": ("Air temperature (°C)", ""),
    "temp_anomaly": ("Temperature anomaly (°C)", "Departure from the usual temperature for the time of year."),
    "wind": ("Wind speed (m/s)", ""),
}


def model_path(name: str) -> Path:
    return MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib")


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def make_xy(df: pd.DataFrame):
    """Return (X, y) with only the model features, in a fixed order."""
    X = df.drop(columns=[c for c in DROP_COLS + [TARGET] if c in df.columns]).copy()
    X["vv_vh_ratio"] = X["VV"] / X["VH"]  # same formula as the dataset column
    return X[FEATURES], df[TARGET]


def risk_level(p: float) -> str:
    if p < 0.35:
        return "Low"
    if p < 0.65:
        return "Moderate"
    return "High"


def row_from_inputs(values: dict, ndwi_available: bool = True) -> pd.DataFrame:
    """Build a one-row DataFrame for prediction from the form values."""
    row = dict(values)
    row["vv_vh_ratio"] = row["VV"] / row["VH"]
    if not ndwi_available:
        row["ndwi"] = np.nan
    return pd.DataFrame([row])[FEATURES]
