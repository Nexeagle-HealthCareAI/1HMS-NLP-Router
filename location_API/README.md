# Location API

Standalone city / pincode / coordinate lookup service for Indian locations —
**exposed to other applications**, not consumed by anything in this repo's
NLP router (`nlp_brain`/`api`/`voice`/`speech`). Deployed independently: its
own [Dockerfile](Dockerfile), own CI/CD
([`deploy-location-api.yml`](../.github/workflows/deploy-location-api.yml)),
own port (**5004**, vs. the NLP router's 5003), same Dev/Prod VMs.

Built with the same SOLID layering throughout
([`location_brain/`](location_brain/)): `repositories/` (one class per CSV,
no business logic) → `services/` (one class per capability, depends only on
narrow `interfaces/`) → `finder.py` (a facade wiring it all together) →
`main.py` (FastAPI, HTTP only).

## Endpoints

| Endpoint | Input | Use when... |
|---|---|---|
| `GET /health` | — | Liveness/readiness probe. |
| `GET/POST /locate` | pincode, `"lat,lon"`, **or** free-text city/town/district — auto-detected | **Start here.** Broadest coverage, single unified response shape regardless of input kind. See below. |
| `GET/POST /find-pincode` | city name (+ optional state) | You specifically need the older `PincodeResponse` shape, or a state-scoped pincode search (`/locate` doesn't take a state filter). |
| `GET/POST /coordinates` | city name | You specifically need `CoordinatesResponse`'s shape. Only covers the curated ~213-city list — `/locate` covers far more. |
| `GET /search` | partial city/town/district name | **Autocomplete-style search as the user types.** Same broad data coverage as `/locate`, but returns a ranked LIST of typed candidates instead of one best-effort answer. See below. |

All endpoints are rate-limited: 60 requests/minute per client IP.

### `GET /search` — autocomplete across every dataset

```
GET /search?city=mumbai&limit=5
```

Searches the curated ~213 cities, the ~19K pincode-dataset districts, the
~5,193-town list, and the official government pincode directory's districts
all at once (the same coverage `/locate` draws on), merged into one ranked,
deduplicated list. Ranking: exact match > prefix match > substring match >
fuzzy typo match; ties broken city-before-town-before-district, then
alphabetically. `limit` (default 10, max 50) caps how many come back.

Each match is typed and enriched with a small sample of pincodes/coordinates
when known:
```json
{
  "query": "mumbai",
  "matches": [
    {
      "name": "Mumbai", "type": "city", "state": "Maharashtra", "district": null,
      "pincodes": ["400001", "400002", "400003", "400004", "400005"],
      "coordinates": {"latitude": 18.987807, "longitude": 72.836447}
    },
    {
      "name": "Mumbai Suburban", "type": "district", "state": "Maharashtra", "district": null,
      "pincodes": ["400010", "400024", "400029", "400042", "400043"],
      "coordinates": {"latitude": 19.068833, "longitude": 72.877783}
    }
  ]
}
```
`type` is `"city"` (curated, has its own coordinates), `"town"` (broader
list, district/state only), or `"district"`. Same name can legitimately
appear more than once with different types (e.g. "Mumbai" the city and
"Mumbai" the district) — these are kept distinct, not merged. `pincodes` is
capped at 5 per match (a full list is what `/locate` or `/find-pincode` are
for); `coordinates` prefers a city's own curated lat/long and falls back to
the government dataset's when the city has none.

### `GET/POST /locate` — the smart, unified endpoint

```
GET /locate?q=400070
GET /locate?q=19.0760,72.8777
GET /locate?q=Ganganagar
POST /locate  {"query": "..."}
```

Detects the query type automatically:
- **6 digits** → treated as a pincode, looked up directly.
- **`"lat,lon"`** (two comma-separated numbers) → reverse geocoded to the
  nearest known post office (great-circle/haversine distance, vectorized
  over ~165K points — this doesn't get slower as the dataset grows).
- **Anything else** → treated as free text and matched against both the
  curated ~213-city list (has coordinates) and the ~5,193-town list
  (broader coverage, no coordinates), then cross-referenced against the
  government pincode directory for pincodes/district/state.

Response shape (same for every query type):
```json
{
  "query": "400070",
  "queryType": "pincode",
  "found": true,
  "matched": "Mumbai Suburban",
  "district": "Mumbai Suburban",
  "state": "Maharashtra",
  "pincodes": ["400070"],
  "coordinates": {"latitude": 19.068833, "longitude": 72.877783},
  "details": [ { "officeName": "...", "officeType": "...", "district": "...",
                 "state": "...", "pincode": "...", "latitude": ..., "longitude": ... } ],
  "suggestions": [],
  "message": ""
}
```
`details` is capped at 20 entries (a large district can have hundreds of
post offices) — `pincodes` has the full deduped list regardless.

**Known data-quality caveat:** the government pincode dataset (below) has
occasional bad coordinates — verified one Madhya Pradesh record whose
listed lat/long exactly coincides with a real Mumbai location. `/locate`'s
coordinate lookup returns the top 5 nearest (see `details`), not just 1,
as a partial mitigation — compare candidates rather than blindly trusting
`matched` alone for high-stakes use.

## Data sources

| File | Rows | Used by |
|---|---|---|
| `Indian_Cities_Database.csv` | ~213 | `/search`, `/find-pincode`, `/coordinates`, `/locate` (text) — the only source with coordinates for named cities |
| `pincode-dataset.csv` | ~19K | `/search`, `/find-pincode` (district-level, no coordinates) |
| `pincodes_gov.csv` | ~165K | `/search`, `/locate` (pincode + coordinates lookups) — official India Post directory, post-office-level, **with coordinates** |
| `Cities_Towns_District_State_India.csv` | ~5,193 | `/search`, `/locate` (text) — much broader town coverage than the curated cities list |
| `Ind_adm2_Points.csv` | ~785K | **Not used.** Raw district-boundary polygon vertices, no pincode/office/city name attached — see the Dockerfile comment for why this wasn't wired in. |

## Local development

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 5004
```

## Testing

Covered by the main repo's suite (`tests/location_API/`) — see
[docs/testing.md](../docs/testing.md). Run just this service's tests:
```bash
pytest tests/location_API -v
```
