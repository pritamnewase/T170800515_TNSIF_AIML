FROM python:3.11-slim

WORKDIR /app

COPY app.py .
COPY heart_disease_model.joblib .

RUN pip install --no-cache-dir fastapi uvicorn pandas scikit-learn==1.6.1 joblib

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
