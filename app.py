"""
app.py – SolarSense AI  – Explainable Solar Power Prediction
Flask application entry point.
"""

import os
import json
import csv
import io
import traceback
from datetime import datetime
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, redirect, url_for, flash, session,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "solarsense-dev-secret-2025")

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
UPLOAD_DIR  = BASE_DIR / "static" / "uploads"
PLOT_DIR    = BASE_DIR / "static" / "plots"
for d in (UPLOAD_DIR, PLOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    # ── Manual entry ──────────────────────────────────────────────────────────
    if request.form.get("mode") == "manual":
        from utils.preprocess import validate_manual_input, scale_features, FEATURE_NAMES
        from utils.predictor  import predict_single, get_feature_importance
        from utils.shap_plot  import feature_importance_bar, prediction_gauge, input_radar

        values, error = validate_manual_input(request.form)
        if error:
            flash(error, "danger")
            return render_template("predict.html")

        X = scale_features(
            values["temperature"],
            values["wind_speed"],
            values["ghi"],
            values["prev_power"],
        )

        try:
            result     = predict_single(X)
            importances = get_feature_importance(X)
        except RuntimeError as exc:
            flash(str(exc), "danger")
            return render_template("predict.html")

        # Generate charts
        imp_fname   = feature_importance_bar(importances)
        gauge_fname = prediction_gauge(result["predicted_power"], result["confidence"])

        # Normalised values for radar
        from utils.preprocess import FEATURE_BOUNDS
        norm_vals = [
            min(1.0, max(0.0, (values["temperature"]  - FEATURE_BOUNDS["Temperature"][0])            / (FEATURE_BOUNDS["Temperature"][1]            - FEATURE_BOUNDS["Temperature"][0]))),
            min(1.0, max(0.0, (values["wind_speed"]   - FEATURE_BOUNDS["Wind Speed"][0])             / (FEATURE_BOUNDS["Wind Speed"][1]             - FEATURE_BOUNDS["Wind Speed"][0]))),
            min(1.0, max(0.0, (values["ghi"]          - FEATURE_BOUNDS["GHI"][0])                    / (FEATURE_BOUNDS["GHI"][1]                    - FEATURE_BOUNDS["GHI"][0]))),
            min(1.0, max(0.0, (values["prev_power"]   - FEATURE_BOUNDS["Previous Active Power"][0])  / (FEATURE_BOUNDS["Previous Active Power"][1]  - FEATURE_BOUNDS["Previous Active Power"][0]))),
        ]
        radar_fname = input_radar(norm_vals, FEATURE_NAMES)

        session["last_result"] = json.dumps({
            "result":      result,
            "inputs":      values,
            "importances": importances,
            "imp_plot":    imp_fname,
        })

        return render_template(
            "result.html",
            result       = result,
            inputs       = values,
            importances  = importances,
            imp_plot     = imp_fname,
            gauge_plot   = gauge_fname,
            radar_plot   = radar_fname,
            batch        = False,
            timestamp    = datetime.now().strftime("%d %b %Y %H:%M"),
        )

    # ── CSV upload ────────────────────────────────────────────────────────────
    if request.form.get("mode") == "csv":
        if "csv_file" not in request.files:
            flash("No file selected.", "danger")
            return render_template("predict.html")

        f = request.files["csv_file"]
        if f.filename == "" or not allowed_file(f.filename):
            flash("Please upload a valid .csv file.", "danger")
            return render_template("predict.html")

        filepath = UPLOAD_DIR / f.filename
        f.save(filepath)

        from utils.preprocess import parse_csv, scale_dataframe, windowed_rows, FEATURE_NAMES, SEQ_LEN
        from utils.predictor  import predict_batch, get_feature_importance
        from utils.shap_plot  import batch_timeline, feature_importance_bar

        df, error = parse_csv(str(filepath))
        if error:
            flash(error, "danger")
            return render_template("predict.html")

        X_batch  = scale_dataframe(df)        # (n_windows, SEQ_LEN, 4)
        aligned  = windowed_rows(df)           # rows matching each window's prediction

        try:
            powers = predict_batch(X_batch).tolist()
        except RuntimeError as exc:
            flash(str(exc), "danger")
            return render_template("predict.html")

        # Use first window for importances
        imp_fname   = feature_importance_bar(get_feature_importance(X_batch[:1]))
        batch_fname = batch_timeline(powers)

        summary = {
            "count":    len(powers),
            "total":    round(sum(powers), 3),
            "mean":     round(sum(powers) / len(powers), 3),
            "max":      round(max(powers), 3),
            "min":      round(min(powers), 3),
        }

        rows = []
        for i, (power, (_, row)) in enumerate(zip(powers, aligned.iterrows()), 1):
            rows.append({
                "index":       i,
                "csv_row":     i + SEQ_LEN - 1,   # 1-indexed row in the original CSV
                "temperature": round(float(row["Temperature"]), 2),
                "wind_speed":  round(float(row["Wind Speed"]), 2),
                "ghi":         round(float(row["GHI"]), 2),
                "prev_power":  round(float(row["Previous Active Power"]), 2),
                "prediction":  round(float(power), 4),
            })

        session["last_batch"] = json.dumps({
            "rows":       rows,
            "summary":    summary,
            "batch_plot": batch_fname,
            "imp_plot":   imp_fname,
        })

        return render_template(
            "result.html",
            batch       = True,
            rows        = rows,
            summary     = summary,
            batch_plot  = batch_fname,
            imp_plot    = imp_fname,
            timestamp   = datetime.now().strftime("%d %b %Y %H:%M"),
        )

    flash("Unknown submission mode.", "danger")
    return render_template("predict.html")


@app.route("/download/report")
def download_report():
    """Generate and serve a PDF report for the last single prediction."""
    raw = session.get("last_result")
    if not raw:
        flash("No prediction to report. Run a prediction first.", "warning")
        return redirect(url_for("predict"))

    data = json.loads(raw)

    from utils.report import generate_report

    imp_abs = str(PLOT_DIR / data.get("imp_plot", "")) if data.get("imp_plot") else None
    fname   = generate_report(
        result        = data["result"],
        inputs        = data["inputs"],
        importance_img= imp_abs,
    )
    return send_from_directory(PLOT_DIR, fname, as_attachment=True,
                               download_name="SolarSense_Report.pdf")


@app.route("/download/csv")
def download_csv():
    """Return the batch predictions as a downloadable CSV."""
    raw = session.get("last_batch")
    if not raw:
        flash("No batch results to download.", "warning")
        return redirect(url_for("predict"))

    data = json.loads(raw)
    rows = data["rows"]

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["index", "csv_row", "temperature", "wind_speed", "ghi",
                    "prev_power", "prediction"],
    )
    writer.writeheader()
    writer.writerows(rows)

    from flask import Response
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=SolarSense_Batch.csv"},
    )


@app.route("/about")
def about():
    return render_template("about.html")


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
