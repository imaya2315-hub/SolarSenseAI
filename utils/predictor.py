import os
import joblib
import numpy as np
import tensorflow as tf

# ----------------------------------------------------------
# Paths
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "hybrid_cnn_lstm_solar.keras")
SCALER_PATH = os.path.join(BASE_DIR, "solar_scaler.pkl")

_model = None
_scaler = None


# ----------------------------------------------------------
# Load Model
# ----------------------------------------------------------
def get_model():
    global _model

    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)

    return _model


# ----------------------------------------------------------
# Load Scaler
# ----------------------------------------------------------
def get_scaler():
    global _scaler

    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)

    return _scaler


# ----------------------------------------------------------
# Convert normalized prediction -> original scale
# ----------------------------------------------------------
def inverse_prediction(raw_prediction):

    scaler = get_scaler()

    # Scaler was fitted on:
    # Temperature
    # Wind Speed
    # GHI
    # Active Power

    dummy = np.zeros((1, 4))
    dummy[0, 3] = raw_prediction

    original = scaler.inverse_transform(dummy)[0, 3]

    return float(original)


# ----------------------------------------------------------
# Confidence
# ----------------------------------------------------------
def calculate_confidence(raw_prediction):

    confidence = 85 + raw_prediction * 12

    confidence = max(60, min(99, confidence))

    return round(confidence, 1)


# ----------------------------------------------------------
# Single Prediction
# ----------------------------------------------------------
def predict_single(X):

    model = get_model()

    raw_prediction = float(
        model.predict(X, verbose=0)[0][0]
    )

    power_watts = inverse_prediction(raw_prediction)

    # Convert to kW for display
    power_kw = power_watts / 1000.0

    return {

        "predicted_power": round(power_kw, 2),

        "predicted_power_watts": round(power_watts, 2),

        "confidence": calculate_confidence(raw_prediction),

        "raw_output": round(raw_prediction, 6)

    }


# ----------------------------------------------------------
# Batch Prediction
# ----------------------------------------------------------
def predict_batch(X):

    model = get_model()

    raw = model.predict(
        X,
        verbose=0
    ).flatten()

    predictions = []

    for value in raw:

        watts = inverse_prediction(float(value))

        predictions.append(watts / 1000.0)

    return np.array(predictions)


# ----------------------------------------------------------
# Feature Importance
# ----------------------------------------------------------
FEATURE_NAMES = [
    "Temperature",
    "Wind Speed",
    "GHI",
    "Previous Active Power"
]

FEATURE_UNITS = [
    "°C",
    "m/s",
    "W/m²",
    "kW"
]


def get_feature_importance(X):

    model = get_model()

    base = float(
        model.predict(X, verbose=0)[0][0]
    )

    delta = 0.05

    importance = []

    for i in range(4):

        modified = X.copy()

        modified[:, :, i] = np.clip(
            modified[:, :, i] + delta,
            0,
            1
        )

        pred = float(
            model.predict(
                modified,
                verbose=0
            )[0][0]
        )

        importance.append(
            abs(pred - base)
        )

    maximum = max(importance)

    if maximum == 0:
        maximum = 1

    results = []

    for i in range(4):

        results.append({

            "name": FEATURE_NAMES[i],

            "unit": FEATURE_UNITS[i],

            "importance": round(
                importance[i] / maximum,
                4
            )

        })

    results.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    return results