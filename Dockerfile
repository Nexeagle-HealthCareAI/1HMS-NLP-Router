FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
# fasttext in requirements.txt is an optional extra of /chat; skip it in the image.
RUN grep -vE '^(fasttext-wheel)' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt

COPY hms_router/ hms_router/
COPY Datasets/ Datasets/
# Held-out test set: removed from training and used to score every rebuild (rag/evaluation.py).
COPY data_pipeline/validation_set.csv data_pipeline/validation_set.csv
COPY api.py .

# Train the /route-symptom model at build time so containers start in seconds and every replica
# serves the same model version. ~40 s.
RUN python -m hms_router.rag.model_bundle build

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

# HMS_PORT: the VMs run with --network host on 5003, the port every caller already uses (deploy-router.yml).
ENV HMS_PORT=5005
EXPOSE 5005
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/ready' % os.environ.get('HMS_PORT', '5005'), timeout=4).status == 200 else 1)"
CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port \"${HMS_PORT:-5005}\" --proxy-headers --forwarded-allow-ips 127.0.0.1"]
