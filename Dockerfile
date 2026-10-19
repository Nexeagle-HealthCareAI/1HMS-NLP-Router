FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
# chromadb and fasttext in requirements.txt are optional extras of /chat; skip them in the image.
RUN grep -vE '^(chromadb|fasttext-wheel)' requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt

COPY hms_router/ hms_router/
COPY Datasets/ Datasets/
COPY api.py .

# Train the /route-symptom model at build time so containers start in seconds and every replica
# serves the same model version. ~40 s.
RUN python -m hms_router.rag.model_bundle build

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 5005
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5005/ready', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5005", "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1"]
