FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Model_1_Doctor_Dekho.py app.py specialty_mapping.py Hinglish_Symptoms_Reference_v2.csv model_meta.json ./

EXPOSE 5003
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5003"]
