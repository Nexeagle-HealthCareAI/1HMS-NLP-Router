FROM python:3.12-slim

WORKDIR /app

# ffmpeg: POST /route-symptom-audio normalizes uploaded audio (webm/opus from
# a browser's MediaRecorder, m4a, etc.) to WAV via pydub before running
# speech recognition -- see speech/transcription.py's _to_wav().
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY nlp_brain/ nlp_brain/
COPY api/ api/
COPY speech/ speech/
COPY location_brain/ location_brain/
COPY location_API/ location_API/
COPY specialty_mapping.py model_meta.json symptom_specialist_classifier.joblib ./

EXPOSE 5003
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "5003"]
