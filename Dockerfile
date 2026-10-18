FROM python:3.11-slim

WORKDIR /app

# Same scikit-learn version the model was trained with (1.6.1), installed first so it is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model and code come after the slow install
COPY churn_pipeline.joblib app.py ./

EXPOSE 5000
CMD ["python", "app.py"]
