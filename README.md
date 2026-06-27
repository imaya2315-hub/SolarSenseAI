# SolarSense AI

Explainable solar power prediction using a hybrid CNN-LSTM model, served with Flask.

Enter a single weather reading or upload a CSV, get a predicted power output in kWh, and see a SHAP-style breakdown of which input drove the prediction — plus a downloadable PDF report or CSV of results.

## Features

- Manual prediction (Temperature, Wind Speed, GHI, Previous Active Power) with synced slider/number inputs
- Batch prediction via CSV upload (up to 500 rows)
- Feature-importance explanation via input-perturbation analysis (SHAP-style, no `shap` package required)
- Interactive charts (Plotly.js) plus server-rendered matplotlib charts for the PDF report
- Downloadable PDF report (single prediction) and CSV export (batch)
- Responsive Bootstrap 5 UI

## Project structure

```
SolarSenseAI/
├── app.py                       # Flask routes
├── requirements.txt
├── hybrid_cnn_lstm_solar.keras  # trained model (lazy-loaded)
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   ├── about.html
│   ├── 404.html
│   └── 500.html
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   ├── sample_data.csv
│   ├── uploads/                # user-uploaded CSVs land here
│   └── plots/                  # generated chart PNGs + PDF reports
└── utils/
    ├── preprocess.py           # validation, scaling, CSV parsing
    ├── predictor.py            # model loading + inference + feature importance
    ├── shap_plot.py            # matplotlib chart generation
    └── report.py               # PDF report generation (ReportLab)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:5000`.

> **Model input shape:** `hybrid_cnn_lstm_solar.keras` was inspected directly (`config.json` inside the `.keras` archive) and its `InputLayer` is built with `batch_shape: [null, 24, 4]` — it's a **24-step lookback window**, not a single timestep. The app handles this as follows:
> - **Manual entry** has only one reading, so that reading is tiled across all 24 timesteps as a steady-state approximation (clearly labelled in the UI). It's an estimate, not a true time-series forecast.
> - **CSV upload** is treated as a real, time-ordered sequence. A sliding window of 24 consecutive rows is built for every valid window, and each window's prediction is reported against its *last* row — so a 30-row file yields 7 predictions, not 30. The CSV upload form enforces a minimum of 24 rows.
>
> If your model was trained with a different scaler, feature order, or window length, update `SEQ_LEN` and `FEATURE_BOUNDS` / `scale_features()` / `scale_dataframe()` in `utils/preprocess.py` to match.

## Deploying to Render

1. Push this project to a GitHub repo.
2. On Render: **New → Web Service** → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Add an environment variable `SECRET_KEY` with a random value.

TensorFlow is a large dependency — Render's free tier may be slow to build/cold-start. If that's a problem, consider `tensorflow-cpu` in `requirements.txt` instead of `tensorflow`.

## Bugs found and fixed while assembling this project

This codebase was generated across an earlier session and re-uploaded to continue. While wiring it together, three real bugs surfaced and were fixed:

1. **`utils/preprocess.py` had a `SyntaxError`** — several minus signs in the temperature bounds were the Unicode character `−` (U+2212) instead of the ASCII `-` (U+002D), which Python doesn't accept as a numeric/unary operator. Fixed by normalising to ASCII hyphens.
2. **Batch CSV predictions crashed with `TypeError: Object of type int64 is not JSON serializable`** — pandas returns numpy scalar types (`int64`/`float64`) from `DataFrame` rows, and `round()` on those returns the same numpy type, which `json.dumps()` (used to stash results in the session) can't serialize. Fixed by casting to `float()` before rounding in `app.py`.
3. **The model expects a 24-timestep window, but the original preprocessing built 1-timestep tensors** — see "Model input shape" above. This silently would have produced a shape-mismatch error (or, depending on TF version, a confusing broadcast) the first time a real prediction ran. Fixed by tiling (manual entry) and sliding-window batching (CSV) as described above.

## Other notes

- The app lazy-loads the Keras model on the first prediction request, so Flask itself starts instantly.
- If the model can't be loaded (missing TensorFlow/Keras, or an incompatible file), the form shows a clear error instead of crashing.

