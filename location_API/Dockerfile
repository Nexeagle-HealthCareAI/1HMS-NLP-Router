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
# The 4 CSVs PincodeFinder actually loads: the original two (Indian_Cities_
# Database.csv, pincode-dataset.csv) plus the two GET/POST /locate needs
# (pincodes_gov.csv -- the ~165K-row official India Post directory, WITH
# coordinates; Cities_Towns_District_State_India.csv -- ~5,193 towns).
# Ind_adm2_Points.csv (~45MB, raw district-boundary polygon vertices, no
# pincode/office/city name attached) is still NOT copied -- it's strictly
# less useful than pincodes_gov.csv's richer points for the same nearest-
# neighbor use case, and would need real polygon geometry (e.g. shapely)
# to use correctly as actual point-in-polygon reverse geocoding rather
# than a nearest-vertex approximation. Re-evaluate if a future feature
# specifically needs true polygon containment.
COPY location_API/location_brain/*.py location_brain/
COPY location_API/location_brain/Indian_Cities_Database.csv \
     location_API/location_brain/pincode-dataset.csv \
     location_API/location_brain/pincodes_gov.csv \
     location_API/location_brain/Cities_Towns_District_State_India.csv \
     location_brain/
COPY location_API/main.py ./

EXPOSE 5004
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5004"]
