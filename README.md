# Project Retain: Churn Prediction API in Docker

**Author:** Ali Muhammad Younis
**Docker Hub:** [`ali5252/churn-api`](https://hub.docker.com/r/ali5252/churn-api)
**Stack:** Python 3.11, Flask, scikit-learn 1.6.1, pandas, Docker (Windows, Docker Desktop)

A telecom customer churn model (trained in the Project Retain notebook) wrapped in a small Flask API and packaged as a Docker image. It has a browser form and a JSON endpoint. The API builds the four engineered features itself, so a caller only sends the 19 raw customer columns.

---

## Table of Contents

1. [What it does](#1-what-it-does)
2. [Project structure](#2-project-structure)
3. [How it works](#3-how-it-works)
4. [Build and run](#4-build-and-run)
5. [Using the API](#5-using-the-api)
6. [Test results](#6-test-results)
7. [Problems I hit and how I fixed them](#7-problems-i-hit-and-how-i-fixed-them)
8. [Push to Docker Hub](#8-push-to-docker-hub)
9. [Limitations](#9-limitations)
10. [Checklist](#10-checklist)

---

## 1. What it does

Given one customer's details, the API returns the chance that the customer will cancel next month, and whether the retention team should contact them.

- **Model:** Logistic Regression (C = 1) inside one scikit-learn pipeline (imputing, scaling, one-hot and ordinal encoding). Built with scikit-learn **1.6.1**.
- **Decision threshold:** **0.10**, chosen in the notebook from cross-validated cost (a missed churner costs $600, a wasted contact costs $60, so contacting pays off above 10%).
- **Data:** IBM Telco Customer Churn sample (7,043 customers, 26.5% churned).
- **Notebook results at threshold 0.10:** cross-validated ROC-AUC 0.846, recall 0.95, precision 0.41 on the test set.

## 2. Project structure

```
churn-api/
├── app.py                  Flask API and browser form
├── Dockerfile              Image recipe
├── requirements.txt        Pinned libraries (scikit-learn==1.6.1)
├── churn_pipeline.joblib   Trained pipeline from the notebook
└── .dockerignore
```

**Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Same scikit-learn version the model was trained with (1.6.1), installed first so it is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model and code come after the slow install
COPY churn_pipeline.joblib app.py ./

EXPOSE 5000
CMD ["python", "app.py"]
```

**requirements.txt**

```
flask
scikit-learn==1.6.1
joblib
pandas
```

## 3. How it works

1. `app.py` loads `churn_pipeline.joblib` once at startup.
2. The model's own `feature_names_in_` gives the 23 columns it expects. Four of them are engineered, so the API asks only for the other 19.
3. For each request the API runs the same `prepare_features()` logic as the notebook:

| Feature | Rule |
|---|---|
| `tenure_group` | `tenure` cut into `0-12`, `13-24`, `25-48`, `49-72` |
| `num_addons` | Number of `Yes` among OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| `fiber_no_support` | 1 if InternetService is Fiber optic **and** TechSupport is No |
| `auto_pay` | 1 if PaymentMethod contains "automatic" |

4. A blank `TotalCharges` (new customers) becomes 0, as in the notebook.
5. The pipeline returns `predict_proba`. The customer is flagged `contact: true` when the probability is at least the threshold.

The threshold can be changed without rebuilding, with an environment variable:

```bash
docker run -d --name churnapi -p 5000:5000 -e THRESHOLD=0.3 ali5252/churn-api:v1
```

## 4. Build and run

```bash
docker build -t ali5252/churn-api:v1 .
docker run -d --name churnapi -p 5000:5000 ali5252/churn-api:v1
docker ps
docker logs churnapi
```

`-p 5000:5000` maps port 5000 on the host to port 5000 in the container. `app.py` listens on `0.0.0.0` so the port is reachable from outside the container.

Stop and clean up:

```bash
docker rm -f churnapi
```

## 5. Using the API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Browser form with dropdowns, pre-filled with an example customer |
| `/features` | GET | The 19 raw columns the API expects |
| `/predict` | POST | JSON in, churn probability out |
| `/health` | GET | Status, model type and threshold |

Open **http://localhost:5000**, choose the customer details, and press **Predict**.

If a field is missing, the API returns a clear error instead of silently filling a default:

```
Missing fields: MonthlyCharges, SeniorCitizen, ...
```

## 6. Test results

Two customers from the notebook, sent through the container:

| | Risky customer | Safe customer |
|---|---|---|
| tenure | 2 | 60 |
| Contract | Month-to-month | Two year |
| InternetService | Fiber optic | DSL |
| TechSupport | No | Yes |
| PaymentMethod | Electronic check | Credit card (automatic) |
| MonthlyCharges / TotalCharges | 75.5 / blank | 55 / 3300 |
| **Churn probability** | **66.5%** | **1.1%** |
| **Contact?** | **Yes** | **No** |

Engineered features built by the API:

| | tenure_group | num_addons | fiber_no_support | auto_pay |
|---|---|---|---|---|
| Risky | 0-12 | 0 | 1 | 0 |
| Safe | 49-72 | 1 | 0 | 1 |

Example response:

```json
{
  "churn_probability": 0.665,
  "contact": true,
  "engineered_features": {
    "auto_pay": "0",
    "fiber_no_support": "1",
    "num_addons": "0",
    "tenure_group": "0-12"
  },
  "threshold": 0.1
}
```

## 7. Problems I hit and how I fixed them

**1. The model would not load on my own PC.**
Loading `churn_pipeline.joblib` locally gave `InconsistentVersionWarning` and then
`AttributeError: module 'sklearn.compose._column_transformer' has no attribute '_RemainderColsList'`.
The model was saved with scikit-learn 1.6.1, but my PC had 1.9.1. This is the "works on my machine" problem Docker solves. The image pins `scikit-learn==1.6.1`, and the same file loads without any warning inside the container.

**2. Training-serving skew.**
My first API version only asked for the 19 raw columns, so the four engineered features arrived empty and the pipeline's imputer filled them with the most common value. The prediction looked fine but was only partly based on the real customer. The fix was to copy `prepare_features()` from the notebook into the API so the same features are built at serving time as at training time.

**3. `Found unknown categories [None] in column 1 during transform`.**
Empty form boxes were passed to the model as `None`. The `SimpleImputer` only treats `NaN` as missing, so the encoder saw `None` as an unknown category. Missing values are now handled explicitly: the API rejects missing fields with a clear message, and a blank `TotalCharges` is turned into 0.

**4. Layer caching.**
Because `requirements.txt` is copied and installed before the code, changing `app.py` rebuilds in seconds and the `pip install` layer shows `CACHED`.

## 8. Push to Docker Hub

```bash
docker push ali5252/churn-api:v1
```

The repository `ali5252/churn-api` with tag `v1` appears on Docker Hub. To confirm the image works from the cloud, remove the local copy and pull it again:

```bash
docker rm -f churnapi
docker rmi ali5252/churn-api:v1
docker pull ali5252/churn-api:v1
docker run -d --name churnapi -p 5000:5000 ali5252/churn-api:v1
```

Anyone can then run it with:

```bash
docker run -d -p 5000:5000 ali5252/churn-api:v1
```

## 9. Limitations

- Flask's built-in server is for development only. A production deployment would use a WSGI server such as gunicorn.
- There is no authentication or input logging.
- From the model card: the model flags about 62% of customers at threshold 0.10, so a team with less capacity should contact only the highest scores. It was trained on one company's sample data with no dates, so it was not tested over time.
- The model uses gender and SeniorCitizen. Fairness across groups was not tested, so it should be checked before real use. It must not be used to refuse service or change prices.

## 10. Checklist

- [x] Churn pipeline loaded inside Docker with the original scikit-learn version
- [x] Flask API with browser form and JSON endpoint
- [x] Four engineered features built by the API from the notebook logic
- [x] Threshold 0.10, changeable with `-e THRESHOLD`
- [x] Image built and run with port mapping (`-p 5000:5000`)
- [x] Two test customers give sensible results (66.5% and 1.1%)
- [x] Image pushed to Docker Hub (`ali5252/churn-api:v1`)
- [ ] Image removed locally and pulled again from Docker Hub
- [ ] Compare the container output with the notebook's `predict_churn()` for the same two customers
