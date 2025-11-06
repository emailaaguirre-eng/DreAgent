"""Test agent endpoints."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient
from agent_service import app

client = TestClient(app)


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_propose_edit_returns_diff():
    response = client.post("/propose_edit", json={
        "path": "test.txt",
        "instruction": "Replace TODO",
        "content": "TODO: implement"
    })
    assert response.status_code == 200
    assert "diff" in response.json()


def test_apply_edit_blocks_on_dry_run():
    response = client.post("/apply_edit", json={
        "path": "test.txt",
        "new_text": "DONE",
        "approved": True,
        "dry_run": True
    })
    assert response.json()["applied"] is False


def test_run_command_blocks_non_whitelisted():
    response = client.post("/run_command", json={
        "cmd": "rm -rf /"
    })
    assert response.status_code == 403
    assert "not whitelisted" in response.json()["detail"].lower()
