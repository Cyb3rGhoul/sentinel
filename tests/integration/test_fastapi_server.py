"""Integration tests for the SENTINEL FastAPI server.

Tests full episode flows via TestClient covering:
- Requirements 1.5, 1.6
"""
import pytest
from fastapi.testclient import TestClient

import sentinel.api.server as server_module
from sentinel.api.server import app

# ---------------------------------------------------------------------------
# Action payloads
# ---------------------------------------------------------------------------

QUERY_LOGS = {
    "agent": "holmes",
    "category": "investigative",
    "name": "QueryLogs",
    "params": {"service": "cart-service", "time_range": [0, 60]},
}
QUERY_METRICS = {
    "agent": "argus",
    "category": "investigative",
    "name": "QueryMetrics",
    "params": {"service": "order-service", "metric_name": "cpu", "time_range": [0, 60]},
}
FORM_HYPOTHESIS = {
    "agent": "holmes",
    "category": "investigative",
    "name": "FormHypothesis",
    "params": {"service": "cart-service", "failure_type": "memory_leak", "confidence": 0.7},
}
RESTART_SERVICE = {
    "agent": "forge",
    "category": "remediation",
    "name": "RestartService",
    "params": {"service": "cart-service"},
}
SCALE_SERVICE = {
    "agent": "forge",
    "category": "remediation",
    "name": "ScaleService",
    "params": {"service": "redis-cache", "replicas": 3},
}
CLOSE_INCIDENT = {
    "agent": "hermes",
    "category": "meta",
    "name": "CloseIncident",
    "params": {"resolution_summary": "Incident resolved"},
}
QUERY_TRACE = {
    "agent": "holmes",
    "category": "investigative",
    "name": "QueryTrace",
    "params": {"trace_id": "trace-001"},
}
CANARY_DEPLOY = {
    "agent": "hermes",
    "category": "deployment",
    "name": "CanaryDeploy",
    "params": {"service": "cart-service", "version": "v2", "traffic_percent": 0.1},
}

# ---------------------------------------------------------------------------
# Observation keys expected in every step response
# ---------------------------------------------------------------------------

OBS_KEYS = {
    "metrics_snapshot",
    "causal_graph_snapshot",
    "active_alerts",
    "recent_logs",
    "active_traces",
    "incident_context",
    "sla_state",
}


def _assert_step_response(data: dict) -> None:
    """Validate the structure of a /step response body."""
    assert "observation" in data
    assert "reward" in data
    assert "terminated" in data
    assert "truncated" in data
    assert "info" in data
    assert isinstance(data["observation"], dict)
    assert OBS_KEYS.issubset(data["observation"].keys())
    assert isinstance(data["reward"], (int, float))
    assert isinstance(data["terminated"], bool)
    assert isinstance(data["truncated"], bool)
    assert isinstance(data["info"], dict)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_server_env():
    """Reset the server's _env singleton before and after each test."""
    server_module._env = None
    yield
    if server_module._env is not None:
        try:
            server_module._env.close()
        except Exception:
            pass
    server_module._env = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_full_episode_investigative_actions():
    """Full episode: investigative actions + render + close.

    Validates: Requirements 1.5, 1.6
    """
    client = TestClient(app, raise_server_exceptions=False)

    # Reset
    resp = client.post("/reset", json={"seed": 42})
    assert resp.status_code == 200
    reset_data = resp.json()
    assert "observation" in reset_data
    assert "info" in reset_data

    # Step 1: QueryLogs
    resp = client.post("/step", json={"action": QUERY_LOGS})
    assert resp.status_code == 200
    _assert_step_response(resp.json())

    # Step 2: QueryMetrics
    resp = client.post("/step", json={"action": QUERY_METRICS})
    assert resp.status_code == 200
    _assert_step_response(resp.json())

    # Step 3: FormHypothesis
    resp = client.post("/step", json={"action": FORM_HYPOTHESIS})
    assert resp.status_code == 200
    _assert_step_response(resp.json())

    # Render
    resp = client.get("/render")
    assert resp.status_code == 200
    assert "render" in resp.json()

    # Close
    resp = client.post("/close")
    assert resp.status_code == 200
    assert resp.json()["status"] == "closed"


def test_full_episode_remediation_and_close():
    """Full episode: remediation actions ending with CloseIncident.

    Validates: Requirements 1.5, 1.6
    """
    client = TestClient(app, raise_server_exceptions=False)

    # Reset
    resp = client.post("/reset", json={"seed": 7})
    assert resp.status_code == 200

    # Step 1: RestartService
    resp = client.post("/step", json={"action": RESTART_SERVICE})
    assert resp.status_code == 200
    _assert_step_response(resp.json())

    # Step 2: ScaleService
    resp = client.post("/step", json={"action": SCALE_SERVICE})
    assert resp.status_code == 200
    _assert_step_response(resp.json())

    # Step 3: CloseIncident — should terminate the episode
    resp = client.post("/step", json={"action": CLOSE_INCIDENT})
    assert resp.status_code == 200
    data = resp.json()
    _assert_step_response(data)
    assert data["terminated"] is True


def test_full_episode_deployment_flow():
    """Full episode: trace query + canary deploy + close, then health check.

    Validates: Requirements 1.5, 1.6
    """
    client = TestClient(app, raise_server_exceptions=False)

    # Reset
    resp = client.post("/reset", json={"seed": 99})
    assert resp.status_code == 200

    # Step 1: QueryTrace
    resp = client.post("/step", json={"action": QUERY_TRACE})
    assert resp.status_code == 200
    _assert_step_response(resp.json())

    # Step 2: CanaryDeploy
    resp = client.post("/step", json={"action": CANARY_DEPLOY})
    assert resp.status_code == 200
    _assert_step_response(resp.json())

    # Step 3: CloseIncident
    resp = client.post("/step", json={"action": CLOSE_INCIDENT})
    assert resp.status_code == 200
    _assert_step_response(resp.json())

    # Health check — env still alive (not closed), step_count >= 3
    resp = client.get("/health")
    assert resp.status_code == 200
    health = resp.json()
    assert health["initialized"] is True
    assert health["step_count"] >= 3
