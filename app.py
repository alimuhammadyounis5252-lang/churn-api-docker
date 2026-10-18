import json
import os

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

model = joblib.load("churn_pipeline.joblib")

# Decision threshold chosen in the notebook (Phase 10). Can be changed at runtime: docker run -e THRESHOLD=0.2 ...
THRESHOLD = float(os.getenv("THRESHOLD", "0.10"))

# Same logic as prepare_features() in the Project Retain notebook
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]
ENGINEERED = ["tenure_group", "num_addons", "fiber_no_support", "auto_pay"]
FEATURE_COLUMNS = list(model.feature_names_in_)
RAW_COLUMNS = [c for c in FEATURE_COLUMNS if c not in ENGINEERED]


def prepare_features(raw):
    df = raw.copy()
    df = df.drop(columns=["customerID", "Churn"], errors="ignore")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                                labels=["0-12", "13-24", "25-48", "49-72"])
    df["num_addons"] = (df[ADDONS] == "Yes").sum(axis=1)
    df["fiber_no_support"] = ((df["InternetService"] == "Fiber optic") &
                              (df["TechSupport"] == "No")).astype(int)
    df["auto_pay"] = df["PaymentMethod"].str.contains("automatic").astype(int)

    return df[FEATURE_COLUMNS]


OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}

# Pre-filled example: a risky new customer (from the notebook)
SAMPLE = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 75.5, "TotalCharges": 0,
}

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Project Retain - Churn API</title>
<style>
body{font-family:Arial,sans-serif;max-width:720px;margin:30px auto;padding:0 15px}
label{display:block;margin-top:10px;font-weight:bold;font-size:14px}
input,select{width:100%;padding:6px;box-sizing:border-box}
button{margin-top:18px;padding:10px 22px;font-size:16px;cursor:pointer}
pre{background:#f3f3f3;padding:12px;border-radius:6px;white-space:pre-wrap}
#headline{font-size:20px;font-weight:bold;margin:10px 0}
</style></head><body>
<h2>Project Retain: Customer Churn Prediction</h2>
<p>Fill in the customer details and press Predict. The four engineered features are built by the API.</p>
<div id="form">Loading...</div>
<button onclick="predict()">Predict</button>
<h3>Result</h3><div id="headline"></div><pre id="out">-</pre>
<script>
const SAMPLE = __SAMPLE__;
const OPTIONS = __OPTIONS__;
let cols = [];
fetch('/features').then(r => r.json()).then(d => {
  cols = d.features;
  document.getElementById('form').innerHTML = cols.map(c => {
    const v = (c in SAMPLE) ? SAMPLE[c] : '';
    if (c in OPTIONS) {
      return '<label>' + c + '</label><select id="f_' + c + '">' +
        OPTIONS[c].map(o => '<option' + (String(o) === String(v) ? ' selected' : '') + '>' + o + '</option>').join('') +
        '</select>';
    }
    return '<label>' + c + '</label><input id="f_' + c + '" value="' + v + '">';
  }).join('');
});
function predict() {
  const body = {};
  cols.forEach(c => {
    const v = document.getElementById('f_' + c).value.trim();
    if (v !== '') body[c] = /^-?\\d+(\\.\\d+)?$/.test(v) ? Number(v) : v;
  });
  fetch('/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)})
    .then(r => r.json()).then(d => {
      document.getElementById('out').textContent = JSON.stringify(d, null, 2);
      document.getElementById('headline').textContent = d.error ? 'Error' :
        'Churn probability: ' + (d.churn_probability * 100).toFixed(1) + '%  |  Contact this customer: ' + (d.contact ? 'YES' : 'NO');
    });
}
</script></body></html>"""


@app.route("/")
def home():
    html = PAGE.replace("__SAMPLE__", json.dumps(SAMPLE)).replace("__OPTIONS__", json.dumps(OPTIONS))
    return Response(html, mimetype="text/html")


@app.route("/health")
def health():
    return jsonify(status="ok", model=type(model).__name__, threshold=THRESHOLD)


@app.route("/features")
def features():
    return jsonify(features=RAW_COLUMNS, engineered_by_api=ENGINEERED)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if isinstance(data, dict) and isinstance(data.get("customer"), dict):
            data = data["customer"]

        data = dict(data)
        if data.get("TotalCharges") in (None, ""):
            data["TotalCharges"] = " "   # blank for brand-new customers, becomes 0 (as in the notebook)

        missing = [c for c in RAW_COLUMNS if c not in data]
        if missing:
            return jsonify(error="Missing fields: " + ", ".join(missing)), 400

        raw = pd.DataFrame([{c: data[c] for c in RAW_COLUMNS}])
        X_new = prepare_features(raw)
        proba = float(model.predict_proba(X_new)[0, 1])

        built = X_new.iloc[0][ENGINEERED].to_dict()
        built = {k: (v.item() if hasattr(v, "item") else str(v)) for k, v in built.items()}

        return jsonify(
            churn_probability=round(proba, 3),
            contact=proba >= THRESHOLD,
            threshold=THRESHOLD,
            engineered_features=built,
        )
    except Exception as e:
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
