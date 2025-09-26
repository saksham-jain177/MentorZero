from fastapi.testclient import TestClient

from api.main import app, get_llm_client


class _MockClient:
	def health(self):
		return {"host": "http://localhost", "model": "mock", "ready": True}


def test_llm_health_endpoint_mocked():
	app.dependency_overrides[get_llm_client] = lambda: _MockClient()
	with TestClient(app) as client:
		resp = client.get("/llm_health")
		assert resp.status_code == 200
		data = resp.json()
		assert data.get("ready") is True

	app.dependency_overrides.clear()

