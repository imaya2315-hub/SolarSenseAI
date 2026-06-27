import numpy as np
import pandas as pd

# ── Feature metadata ──────────────────────────────────────────────────────────
FEATURE_NAMES = ["Temperature", "Wind Speed", "GHI", "Previous Active Power"]
FEATURE_UNITS = ["°C", "m/s", "W/m²", "kW"]

# The model InputLayer expects 24 timesteps × 4 features
SEQ_LEN = 24

# Normalisation bounds
FEATURE_BOUNDS = {
    "Temperature": (-10.0, 50.0),
    "Wind Speed": (0.0, 20.0),
    "GHI": (0.0, 1200.0),
    "Previous Active Power": (0.0, 500.0),
}


def _minmax_scale(value: float, low: float, high: float) -> float:
    if high == low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def scale_features(
    temp: float,
    wind: float,
    ghi: float,
    prev_power: float
) -> np.ndarray:

    scaled = np.array([
        _minmax_scale(temp, *FEATURE_BOUNDS["Temperature"]),
        _minmax_scale(wind, *FEATURE_BOUNDS["Wind Speed"]),
        _minmax_scale(ghi, *FEATURE_BOUNDS["GHI"]),
        _minmax_scale(prev_power, *FEATURE_BOUNDS["Previous Active Power"]),
    ], dtype=np.float32)

    tiled = np.tile(scaled, (SEQ_LEN, 1))

    return tiled.reshape(1, SEQ_LEN, 4)


def validate_manual_input(form):

    keys = [
        "temperature",
        "wind_speed",
        "ghi",
        "prev_power"
    ]

    labels = [
        "Temperature",
        "Wind Speed",
        "GHI",
        "Previous Active Power"
    ]

    result = {}

    for key, label in zip(keys, labels):

        raw = form.get(key, "").strip()

        if not raw:
            return None, f"{label} is required."

        try:
            result[key] = float(raw)

        except ValueError:
            return None, f"{label} must be numeric."

    if not (-50 <= result["temperature"] <= 60):
        return None, "Temperature must be between -50°C and 60°C."

    if not (0 <= result["wind_speed"] <= 50):
        return None, "Wind Speed must be between 0 and 50 m/s."

    if not (0 <= result["ghi"] <= 1500):
        return None, "GHI must be between 0 and 1500."

    if not (0 <= result["prev_power"] <= 1000):
        return None, "Previous Active Power must be between 0 and 1000."

    return result, None


def parse_csv(filepath):

    try:
        df = pd.read_csv(filepath)

    except Exception as exc:
        return None, str(exc)

    alias = {

        "temperature": "Temperature",
        "temp": "Temperature",
        "t": "Temperature",

        "wind_speed": "Wind Speed",
        "wind": "Wind Speed",
        "windspeed": "Wind Speed",

        "ghi": "GHI",
        "irradiance": "GHI",

        "prev_power": "Previous Active Power",
        "previous_power": "Previous Active Power",
        "previous_active_power": "Previous Active Power",
        "active_power": "Previous Active Power",
        "power": "Previous Active Power",
    }

    df.columns = [
        alias.get(c.lower().strip(), c)
        for c in df.columns
    ]

    missing = [

        c

        for c in FEATURE_NAMES

        if c not in df.columns

    ]

    if missing:
        return None, f"Missing columns: {missing}"

    df = df[FEATURE_NAMES]

    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()

    if len(df) < SEQ_LEN:

        return None, f"Minimum {SEQ_LEN} rows required."

    return df.reset_index(drop=True), None


def scale_dataframe(df):

    scaled = np.column_stack([

        df["Temperature"].map(
            lambda v: _minmax_scale(
                v,
                *FEATURE_BOUNDS["Temperature"]
            )
        ),

        df["Wind Speed"].map(
            lambda v: _minmax_scale(
                v,
                *FEATURE_BOUNDS["Wind Speed"]
            )
        ),

        df["GHI"].map(
            lambda v: _minmax_scale(
                v,
                *FEATURE_BOUNDS["GHI"]
            )
        ),

        df["Previous Active Power"].map(
            lambda v: _minmax_scale(
                v,
                *FEATURE_BOUNDS["Previous Active Power"]
            )
        )

    ]).astype(np.float32)

    windows = []

    for i in range(len(scaled) - SEQ_LEN + 1):
        windows.append(
            scaled[i:i + SEQ_LEN]
        )

    return np.array(windows)


def windowed_rows(df):

    return df.iloc[
        SEQ_LEN - 1:
    ].reset_index(drop=True)