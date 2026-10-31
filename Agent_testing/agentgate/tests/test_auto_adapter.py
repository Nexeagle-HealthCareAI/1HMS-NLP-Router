from agentgate.adapters import AutoAgent


def test_auto_discovers_fastapi_app_and_maps_request_and_response(tmp_path):
    source = tmp_path / "api.py"
    source.write_text(
        "from fastapi import FastAPI\n"
        "from pydantic import BaseModel\n"
        "app = FastAPI()\n"
        "class RequestBody(BaseModel):\n"
        "    query: str\n"
        "    topK: int = 5\n"
        "@app.post('/route-symptom')\n"
        "def route(req: RequestBody):\n"
        "    return {'specialties': [{'name': 'Cardiology'}], "
        "'specialtyIds': ['cardiology'], 'decision': 'answer', "
        "'noMatch': False, 'decisive': True, 'emergency': {'flag': False}}\n",
        encoding="utf-8",
    )
    adapter = AutoAgent(source)
    try:
        result = adapter({"id": "case-1", "input": "chest pain", "context": {}})
    finally:
        adapter.close()

    assert adapter.describe().endswith("POST /route-symptom")
    assert result.error is None
    assert result.text == "Cardiology"
    assert result.final_state["emergency"]["flag"] is False
    assert result.raw["specialtyIds"] == ["cardiology"]


def test_auto_discovers_python_callable(tmp_path):
    source = tmp_path / "agent.py"
    source.write_text(
        "def predict(input):\n"
        "    return {'text': f'handled: {input}', 'final_state': {'ok': True}}\n",
        encoding="utf-8",
    )
    adapter = AutoAgent(source)
    try:
        result = adapter({"id": "case-1", "input": "hello", "context": {}})
    finally:
        adapter.close()

    assert adapter.describe().endswith("callable predict")
    assert result.error is None
    assert result.text == "handled: hello"
    assert result.final_state == {"ok": True}


def test_auto_response_includes_validation_error_details(tmp_path):
    source = tmp_path / "api.py"
    source.write_text(
        "from fastapi import FastAPI\n"
        "from fastapi.responses import JSONResponse\n"
        "app = FastAPI()\n"
        "@app.post('/chat')\n"
        "def chat(body: dict):\n"
        "    return JSONResponse(status_code=422, content={"
        "'detail': 'Request body failed validation', 'errors': ["
        "{'loc': ['query', 'request'], 'msg': 'Field required'}]})\n",
        encoding="utf-8",
    )
    adapter = AutoAgent(source)
    try:
        result = adapter({"id": "case-1", "input": "hello", "context": {}})
    finally:
        adapter.close()

    assert result.error == (
        "HTTP 422: Request body failed validation; query.request: Field required")
