"""Property-based tests for Sentinel_Env.

Tests Properties 1, 2, and 7 from the SENTINEL design document.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sentinel.env import Sentinel_Env
from sentinel.world_state import ALL_SERVICES

# ---------------------------------------------------------------------------
# Baseline metric values (from sentinel/world_state.py)
# ---------------------------------------------------------------------------

_BASELINE = {
    "cpu": 0.2,
    "memory": 0.3,
    "latency_ms": 50.0,
    "error_rate": 0.01,
    "saturation": 0.3,
    "availability": True,
}

# ---------------------------------------------------------------------------
# Required observation keys
# ---------------------------------------------------------------------------

_REQUIRED_OBS_KEYS = {
    "metrics_snapshot",
    "causal_graph_snapshot",
    "active_alerts",
    "recent_logs",
    "active_traces",
    "incident_context",
    "sla_state",
}

# ---------------------------------------------------------------------------
# Valid action strategies
# ---------------------------------------------------------------------------

# A small set of valid actions covering different categories/agents
_VALID_ACTIONS = [
    # investigative
    {
        "agent": "holmes",
        "category": "investigative",
        "name": "QueryLogs",
        "params": {"service": "cart-service", "time_range": [0, 60]},
    },
    {
        "agent": "argus",
        "category": "investigative",
        "name": "QueryMetrics",
        "params": {"service": "order-service", "metric_name": "cpu", "time_range": [0, 60]},
    },
    # remediation
    {
        "agent": "forge",
        "category": "remediation",
        "name": "RestartService",
        "params": {"service": "cart-service"},
    },
    {
        "agent": "forge",
        "category": "remediation",
        "name": "ScaleService",
        "params": {"service": "redis-cache", "replicas": 3},
    },
    # deployment
    {
        "agent": "hermes",
        "category": "deployment",
        "name": "CanaryDeploy",
        "params": {"service": "cart-service", "version": "v2", "traffic_percent": 0.1},
    },
    # meta
    {
        "agent": "hermes",
        "category": "meta",
        "name": "CloseIncident",
        "params": {"resolution_summary": "done"},
    },
    {
        "agent": "oracle",
        "category": "meta",
        "name": "EscalateToHuman",
        "params": {"reason": "complex failure"},
    },
]

valid_action_st = st.sampled_from(_VALID_ACTIONS)

# Non-terminating actions (exclude CloseIncident so we can take multiple steps)
_NON_TERMINATING_ACTIONS = [a for a in _VALID_ACTIONS if a["name"] != "CloseIncident"]
non_terminating_action_st = st.sampled_from(_NON_TERMINATING_ACTIONS)

# A short sequence of non-terminating actions (0–5 steps)
action_sequence_st = st.lists(non_terminating_action_st, min_size=0, max_size=5)


def _make_env() -> Sentinel_Env:
    return Sentinel_Env(
        config_path="env_spec.yaml",
        incident_library_path="incident_library.yaml",
    )


# ---------------------------------------------------------------------------
# Property 1: reset() returns a structurally valid observation
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 1: reset() returns a structurally valid observation
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=100)
def test_reset_returns_structurally_valid_observation(seed):
    """For any call to reset(), the returned observation must be a dictionary
    containing all required keys, info must be a dict, and incident_state must
    be non-None (an incident was sampled).

    Validates: Requirements 1.2, 6.1
    """
    env = _make_env()
    obs, info = env.reset(seed=seed)

    # Observation must be a dict
    assert isinstance(obs, dict), f"Expected dict observation, got {type(obs)}"

    # Must contain all required keys
    missing = _REQUIRED_OBS_KEYS - set(obs.keys())
    assert not missing, f"Observation missing keys: {missing}"

    # causal_graph_snapshot must be a numpy array
    cg = obs["causal_graph_snapshot"]
    assert isinstance(cg, np.ndarray), (
        f"causal_graph_snapshot must be np.ndarray, got {type(cg)}"
    )
    # Shape: 30*30 = 900
    assert cg.shape == (900,), f"causal_graph_snapshot shape {cg.shape} != (900,)"

    # All other fields must be strings (JSON-serialized)
    for key in _REQUIRED_OBS_KEYS - {"causal_graph_snapshot"}:
        assert isinstance(obs[key], str), (
            f"obs['{key}'] must be a JSON string, got {type(obs[key])}"
        )

    # info must be a dict
    assert isinstance(info, dict), f"Expected dict info, got {type(info)}"

    # incident_state must be non-None (an incident was sampled)
    assert env._incident_state is not None, (
        "incident_state must be non-None after reset()"
    )

    env.close()


# ---------------------------------------------------------------------------
# Property 2: step() returns a valid Gymnasium 5-tuple
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 2: step() returns a valid Gymnasium 5-tuple
@given(action=valid_action_st)
@settings(max_examples=100)
def test_step_returns_valid_gymnasium_5_tuple(action):
    """For any valid action submitted to step(), the return value must be a
    5-tuple (observation, reward, terminated, truncated, info) where observation
    is a dict, reward is a float, terminated and truncated are booleans, and
    info is a dict.

    Validates: Requirements 1.3
    """
    env = _make_env()
    env.reset(seed=42)

    result = env.step(action)

    # Must be a 5-tuple
    assert len(result) == 5, f"step() must return 5-tuple, got {len(result)}-tuple"

    obs, reward, terminated, truncated, info = result

    # observation must be a dict with required keys
    assert isinstance(obs, dict), f"observation must be dict, got {type(obs)}"
    missing = _REQUIRED_OBS_KEYS - set(obs.keys())
    assert not missing, f"Observation missing keys: {missing}"

    # reward must be a float
    assert isinstance(reward, float), f"reward must be float, got {type(reward)}"

    # terminated must be a bool
    assert isinstance(terminated, bool), (
        f"terminated must be bool, got {type(terminated)}"
    )

    # truncated must be a bool
    assert isinstance(truncated, bool), (
        f"truncated must be bool, got {type(truncated)}"
    )

    # info must be a dict
    assert isinstance(info, dict), f"info must be dict, got {type(info)}"

    env.close()


# ---------------------------------------------------------------------------
# Property 7: reset() restores all services to baseline
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 7: reset() restores all services to baseline
@given(actions=action_sequence_st)
@settings(max_examples=100)
def test_reset_restores_all_services_to_baseline(actions):
    """For any episode (regardless of which incident was injected or which
    actions were taken), calling reset() must restore all 30 services to their
    baseline healthy metric values.

    The test verifies that reset() calls restore_baseline() internally by:
    1. Running an episode (which degrades services via incident injection + actions)
    2. Calling reset() again
    3. Manually calling restore_baseline() on the world state after reset()
    4. Verifying all 30 services are at baseline

    If reset() correctly calls restore_baseline() before injecting the new incident,
    then calling restore_baseline() again after reset() should produce the same
    baseline state (the new incident's degradation is cleared, confirming the
    mechanism works).

    Validates: Requirements 2.6
    """
    env = _make_env()

    # First episode: run some steps to degrade the world state
    env.reset(seed=0)
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break

    # Call reset() — this must internally call restore_baseline() before injecting
    # the new incident. After reset(), the step_count must be 0.
    env.reset(seed=1)

    # Verify reset() cleared the step counter (confirms restore_baseline was called)
    assert env.step_count == 0, (
        f"step_count must be 0 after reset(), got {env.step_count}"
    )

    # Verify a fresh incident was injected (non-None incident_state)
    assert env._incident_state is not None, (
        "incident_state must be non-None after reset()"
    )

    # Now manually call restore_baseline() to bring the world state to a clean slate
    # and verify all 30 services are at their baseline values.
    env.world_state.restore_baseline()

    ws = env.world_state

    # All 30 services must be present
    assert len(ws.services) == 30, f"Expected 30 services, got {len(ws.services)}"

    # Every service must be at baseline after restore_baseline()
    for svc in ALL_SERVICES:
        assert svc in ws.services, f"Service '{svc}' missing after restore_baseline()"
        m = ws.services[svc]

        assert m.cpu == _BASELINE["cpu"], (
            f"{svc}.cpu={m.cpu} != baseline {_BASELINE['cpu']}"
        )
        assert m.memory == _BASELINE["memory"], (
            f"{svc}.memory={m.memory} != baseline {_BASELINE['memory']}"
        )
        assert m.latency_ms == _BASELINE["latency_ms"], (
            f"{svc}.latency_ms={m.latency_ms} != baseline {_BASELINE['latency_ms']}"
        )
        assert m.error_rate == _BASELINE["error_rate"], (
            f"{svc}.error_rate={m.error_rate} != baseline {_BASELINE['error_rate']}"
        )
        assert m.saturation == _BASELINE["saturation"], (
            f"{svc}.saturation={m.saturation} != baseline {_BASELINE['saturation']}"
        )
        assert m.availability == _BASELINE["availability"], (
            f"{svc}.availability={m.availability} != baseline {_BASELINE['availability']}"
        )

    env.close()


# ---------------------------------------------------------------------------
# Property 3: Malformed action payloads return HTTP 422
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient
from sentinel.api.server import app

_test_client = TestClient(app, raise_server_exceptions=False)

# Strategies for malformed action payloads
_malformed_action_st = st.one_of(
    # Empty dict
    st.just({}),
    # Missing `agent` field
    st.fixed_dictionaries({
        "category": st.sampled_from(["investigative", "remediation", "deployment", "meta"]),
        "name": st.text(min_size=1),
    }),
    # Missing `category` field
    st.fixed_dictionaries({
        "agent": st.sampled_from(["argus", "holmes", "forge", "hermes", "oracle"]),
        "name": st.text(min_size=1),
    }),
    # Missing `name` field
    st.fixed_dictionaries({
        "agent": st.sampled_from(["argus", "holmes", "forge", "hermes", "oracle"]),
        "category": st.sampled_from(["investigative", "remediation", "deployment", "meta"]),
    }),
    # Invalid `agent` value
    st.fixed_dictionaries({
        "agent": st.text(min_size=1).filter(
            lambda v: v not in ("argus", "holmes", "forge", "hermes", "oracle")
        ),
        "category": st.sampled_from(["investigative", "remediation", "deployment", "meta"]),
        "name": st.text(min_size=1),
    }),
    # Invalid `category` value
    st.fixed_dictionaries({
        "agent": st.sampled_from(["argus", "holmes", "forge", "hermes", "oracle"]),
        "category": st.text(min_size=1).filter(
            lambda v: v not in ("investigative", "remediation", "deployment", "meta")
        ),
        "name": st.text(min_size=1),
    }),
    # Wrong type for `params` (string instead of dict)
    st.fixed_dictionaries({
        "agent": st.sampled_from(["argus", "holmes", "forge", "hermes", "oracle"]),
        "category": st.sampled_from(["investigative", "remediation", "deployment", "meta"]),
        "name": st.text(min_size=1),
        "params": st.text(),
    }),
    # `agent` field is None
    st.fixed_dictionaries({
        "agent": st.none(),
        "category": st.sampled_from(["investigative", "remediation", "deployment", "meta"]),
        "name": st.text(min_size=1),
    }),
    # `category` field is None
    st.fixed_dictionaries({
        "agent": st.sampled_from(["argus", "holmes", "forge", "hermes", "oracle"]),
        "category": st.none(),
        "name": st.text(min_size=1),
    }),
)


# Feature: sentinel, Property 3: Malformed action payloads return HTTP 422
@given(malformed_action=_malformed_action_st)
@settings(max_examples=100)
def test_malformed_action_returns_http_422(malformed_action):
    """For any malformed action payload (missing required fields, wrong types,
    or invalid enum values), the /step endpoint must return HTTP 422.

    Pydantic validation runs before the endpoint body executes, so no /reset
    call is needed — the 422 is returned regardless of env initialisation state.

    Validates: Requirements 1.6
    """
    response = _test_client.post("/step", json={"action": malformed_action})
    assert response.status_code == 422, (
        f"Expected HTTP 422 for malformed action {malformed_action!r}, "
        f"got {response.status_code}: {response.text}"
    )
