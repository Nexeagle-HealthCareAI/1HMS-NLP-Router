# Standalone image for location_API -- a separate service (city/pincode/
# coordinate lookup) exposed to OTHER applications, not consumed by the NLP
# router. Deployed independently via .github/workflows/deploy-location-api.yml;
# see ../Dockerfile for the (unrelated) NLP router image.
FROM python:3.12-slim

WORKDIR /app

COPY location_API/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# location_brain must land at /app/location_brain (a top-level package) to
# match main.py's `from location_brain.Pincode_final import ...` -- NOT
# nested under location_API/, which is just this repo's on-disk layout.
#
# Only the two CSVs main.py's PincodeFinder actually loads (Indian_Cities_
# Database.csv, pincode-dataset.csv) are copied -- Cities_Towns_District_
# State_India.csv, Ind_adm2_Points.csv, and pincodes_gov.csv are committed
# to the repo but unreferenced by any code path here (verified: nothing
# under location_brain/ imports or opens them). Together they're ~68MB;
# copying them into the image would be pure bloat for zero functionality.
COPY location_API/location_brain/*.py location_brain/
COPY location_API/location_brain/Indian_Cities_Database.csv location_API/location_brain/pincode-dataset.csv location_brain/
COPY location_API/main.py ./

EXPOSE 5004
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5004"]
