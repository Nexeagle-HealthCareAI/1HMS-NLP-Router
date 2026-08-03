FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py Model_1_Final.py Voice_revised.py specialty_mapping.py model_meta.json symptom_specialist_classifier.joblib big.model ./

EXPOSE 5003
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5003"]
