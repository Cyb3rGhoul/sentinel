"""Integration tests for the OpenEnv adapter layer."""
from __future__ import annotations

from models import SentinelAction
from server.sentinel_environment import SentinelEnvironment


def test_adapter_reset_step_and_state() -> None:
    adapter = SentinelEnvironment()

    initial_obs = adapter.reset(seed=42)
    initial_state = adapter.state

    assert initial_state.step_count == 0
    assert initial_state.episode_id
    assert initial_state.incident_id is not None

    assert initial_obs.metrics_snapshot
    assert "incident_id" in initial_obs.info
    assert initial_obs.reward == 0.0
    assert initial_obs.done is False

    next_obs = adapter.step(
        SentinelAction(
            agent="holmes",
            category="investigative",
            name="QueryLogs",
            params={"service": "cart-service", "time_range": [0, 60]},
        )
    )

    state = adapter.state
    assert state.step_count == 1
    assert isinstance(next_obs.reward, float)
    assert isinstance(next_obs.done, bool)
    assert isinstance(next_obs.metrics_snapshot, dict)
    assert isinstance(next_obs.active_alerts, list)
    assert isinstance(next_obs.recent_logs, list)
    assert isinstance(next_obs.active_traces, list)
    assert isinstance(next_obs.sla_state, dict)
