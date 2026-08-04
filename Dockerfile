FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY nlp_brain/ nlp_brain/
COPY api/ api/
COPY specialty_mapping.py model_meta.json symptom_specialist_classifier.joblib ./

EXPOSE 5003
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "5003"]
