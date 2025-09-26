from fastapi.testclient import TestClient

from api.main import app
from agent.config import get_settings


def test_root_ok():
	with TestClient(app) as client:
		resp = client.get("/")
		assert resp.status_code == 200
		assert resp.json().get("service") == "MentorZero API"


def test_settings_loads():
	s = get_settings()
	assert s is not None

