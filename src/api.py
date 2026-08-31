"""
Bearing Predictive Maintenance API
-----------------------------------
FastAPI application that serves a machine learning model for bearing fault
classification based on time-domain and frequency-domain vibration features.
"""

import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

try:
    import joblib
except Exception:
    joblib = None

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Feature Columns
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "b1_rms", "b1_kurtosis", "b1_skewness",
    "peak_freq_hz", "spectral_centroid", "spectral_energy",
    "top_freq_1_hz", "top_amp_1", "top_freq_2_hz", "top_amp_2",
    "top_freq_3_hz", "top_amp_3", "top_freq_4_hz", "top_amp_4",
    "top_freq_5_hz", "top_amp_5",
]


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_pkl(name):
    """Load a serialised artifact from the models directory."""
    p = ROOT / "models" / name
    if joblib is not None:
        try:
            return joblib.load(p)
        except Exception:
            pass
    with open(p, "rb") as f:
        return pickle.load(f)


MODEL   = load_pkl("predictive_model.pkl")
SCALER  = load_pkl("scaler.pkl")
ENCODER = load_pkl("label_encoder.pkl")

print("All machine learning artifacts successfully loaded.")

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="Bearing Predictive Maintenance API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Model
# ---------------------------------------------------------------------------
class BearingFeatures(BaseModel):
    b1_rms: float; b1_kurtosis: float; b1_skewness: float
    peak_freq_hz: float; spectral_centroid: float; spectral_energy: float
    top_freq_1_hz: float; top_amp_1: float
    top_freq_2_hz: float; top_amp_2: float
    top_freq_3_hz: float; top_amp_3: float
    top_freq_4_hz: float; top_amp_4: float
    top_freq_5_hz: float; top_amp_5: float


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def as_dict(m):
    """Convert a Pydantic model instance to a plain dictionary."""
    return m.model_dump() if hasattr(m, "model_dump") else m.dict()


def predict_frame(feat_dict):
    """Scale features, run inference, and decode the predicted label."""
    X  = pd.DataFrame([feat_dict], columns=FEATURE_COLUMNS)
    Xs = SCALER.transform(X)

    raw   = MODEL.predict(Xs)[0]
    label = raw if isinstance(raw, str) else ENCODER.inverse_transform([raw])[0]

    try:
        conf = float(max(MODEL.predict_proba(Xs)[0])) * 100
    except Exception:
        conf = 100.0

    return str(label), conf


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the monitoring dashboard HTML page."""
    return FileResponse(ROOT / "monitor.html", media_type="text/html")


@app.get("/csv")
async def csv_data():
    """Return the processed feature CSV file for the frontend dashboard."""
    return FileResponse(
        ROOT / "data" / "processed" / "final_bearing_features_processing.csv",
        media_type="text/csv",
    )


@app.get("/health")
async def health():
    """Health check endpoint — confirms the model is loaded and ready."""
    return {
        "status": "ok",
        "model_loaded": True,
        "expected_features": len(FEATURE_COLUMNS),
    }


@app.post("/predict")
async def predict(f: BearingFeatures):
    """Predict the bearing condition for a single set of features."""
    label, conf = predict_frame(as_dict(f))
    return {"status": label, "confidence": round(conf, 2)}


@app.post("/predict_batch")
async def predict_batch(rows: list[BearingFeatures]):
    """Predict bearing conditions for a batch of feature sets."""
    X  = pd.DataFrame([as_dict(r) for r in rows], columns=FEATURE_COLUMNS)
    Xs = SCALER.transform(X)

    raw    = MODEL.predict(Xs)
    labels = [
        str(l)
        for l in (raw if isinstance(raw[0], str) else ENCODER.inverse_transform(raw))
    ]

    try:
        confs = [
            round(float(c) * 100, 2)
            for c in MODEL.predict_proba(Xs).max(axis=1)
        ]
    except Exception:
        confs = [100.0] * len(rows)

    return {"status": labels, "confidence": confs}
