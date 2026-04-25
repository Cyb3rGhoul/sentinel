"""Unit tests for HERMES canary deployment logic.

Tests:
- Rollback trigger when error_rate exceeds baseline during observation window
- Successful promotion path when error_rate stays below baseline

Requirements: 11.2, 11.3
"""
from __future__ import annotations

import pytest

from sentinel.agents.hermes import HERMES
from sentinel.world_state import NexaStackWorldState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_world_state(service: str, error_rate: float) -> NexaStackWorldState:
    """Create a world state with a specific error_rate for one service."""
    ws = NexaStackWorldState()
    # Directly set the error_rate on the service metrics
    m = ws.services[service]
    from sentinel.models import ServiceMetrics
    ws.services[service] = ServiceMetrics(
        cpu=m.cpu,
        memory=m.memory,
        latency_ms=m.latency_ms,
        error_rate=error_rate,
        saturation=m.saturation,
        availability=m.availability,
    )
    return ws


# ---------------------------------------------------------------------------
# Rollback tests (Req 11.3)
# ---------------------------------------------------------------------------

class TestHermesCanaryRollback:
    """Tests that HERMES auto-rolls back when error_rate exceeds baseline."""

    def test_rollback_when_error_rate_exceeds_baseline(self):
        """execute_canary returns ('rolled_back', reason) when error_rate rises."""
        hermes = HERMES(observation_window=5)
        service = "cart-service"

        # Start with a low baseline error_rate
        ws = _make_world_state(service, error_rate=0.01)

        # Simulate: after canary starts, error_rate spikes above baseline
        # We need to mutate the world state to simulate the spike
        # The execute_canary method reads error_rate each step from world_state
        # We'll set it above baseline before calling execute_canary
        from sentinel.models import ServiceMetrics
        m = ws.services[service]
        ws.services[service] = ServiceMetrics(
            cpu=m.cpu,
            memory=m.memory,
            latency_ms=m.latency_ms,
            error_rate=0.15,  # above baseline of 0.01
            saturation=m.saturation,
            availability=m.availability,
        )

        outcome, reason = hermes.execute_canary(service, "v2", 10.0, ws)

        assert outcome == "rolled_back"
        assert reason is not None
        assert "error_rate" in reason.lower()

    def test_rollback_records_timeline_entry(self):
        """A rollback must record a post-mortem timeline entry."""
        hermes = HERMES(observation_window=5)
        service = "order-service"

        ws = _make_world_state(service, error_rate=0.20)  # already above threshold

        outcome, reason = hermes.execute_canary(service, "v2", 10.0, ws)

        assert outcome == "rolled_back"
        timeline = hermes.timeline
        assert len(timeline) >= 1
        rollback_entries = [e for e in timeline if e.event_type == "canary_rollback"]
        assert len(rollback_entries) >= 1

    def test_rollback_on_first_step_of_window(self):
        """Rollback should trigger on the very first step if error_rate is already elevated."""
        hermes = HERMES(observation_window=3)
        service = "redis-cache"

        # error_rate is already above baseline (0.01) at canary start
        ws = _make_world_state(service, error_rate=0.50)

        outcome, reason = hermes.execute_canary(service, "v-fix", 5.0, ws)

        assert outcome == "rolled_back"

    def test_rollback_for_unknown_service_returns_rolled_back(self):
        """execute_canary on a non-existent service returns rolled_back."""
        hermes = HERMES(observation_window=5)
        ws = NexaStackWorldState()

        outcome, reason = hermes.execute_canary("nonexistent-service", "v1", 10.0, ws)

        assert outcome == "rolled_back"
        assert reason is not None

    def test_rollback_reason_contains_step_info(self):
        """Rollback reason should mention the step at which it occurred."""
        hermes = HERMES(observation_window=5)
        service = "user-auth"

        ws = _make_world_state(service, error_rate=0.99)

        outcome, reason = hermes.execute_canary(service, "v-bad", 10.0, ws)

        assert outcome == "rolled_back"
        # Reason should mention step info
        assert "step" in reason.lower() or "error_rate" in reason.lower()


# ---------------------------------------------------------------------------
# Promotion tests (Req 11.2)
# ---------------------------------------------------------------------------

class TestHermesCanaryPromotion:
    """Tests that HERMES promotes the canary when error_rate stays below baseline."""

    def test_promotion_when_error_rate_stable(self):
        """execute_canary returns ('promoted', None) when error_rate stays at baseline."""
        hermes = HERMES(observation_window=3)
        service = "cart-service"

        # Baseline error_rate is 0.01 (from _BASELINE); keep it there
        ws = _make_world_state(service, error_rate=0.01)

        outcome, reason = hermes.execute_canary(service, "v2", 10.0, ws)

        assert outcome == "promoted"
        assert reason is None

    def test_promotion_records_timeline_entry(self):
        """A successful promotion must record a post-mortem timeline entry."""
        hermes = HERMES(observation_window=2)
        service = "search-service"

        ws = _make_world_state(service, error_rate=0.005)

        outcome, reason = hermes.execute_canary(service, "v3", 20.0, ws)

        assert outcome == "promoted"
        timeline = hermes.timeline
        promoted_entries = [e for e in timeline if e.event_type == "canary_promoted"]
        assert len(promoted_entries) >= 1

    def test_promotion_with_window_of_one(self):
        """Promotion works with observation_window=1."""
        hermes = HERMES(observation_window=1)
        service = "product-catalog"

        ws = _make_world_state(service, error_rate=0.01)

        outcome, reason = hermes.execute_canary(service, "v-patch", 10.0, ws)

        assert outcome == "promoted"
        assert reason is None

    def test_promotion_when_error_rate_below_baseline(self):
        """Promotion when error_rate is strictly below the pre-canary baseline."""
        hermes = HERMES(observation_window=3)
        service = "notification-service"

        # Set a higher baseline, then the current rate is lower
        ws = _make_world_state(service, error_rate=0.005)

        outcome, reason = hermes.execute_canary(service, "v-improved", 10.0, ws)

        assert outcome == "promoted"
        assert reason is None

    def test_reset_clears_canary_state(self):
        """reset() should clear all canary state."""
        hermes = HERMES(observation_window=3)
        service = "cart-service"
        ws = _make_world_state(service, error_rate=0.01)

        hermes.execute_canary(service, "v2", 10.0, ws)
        hermes.reset()

        assert hermes._canary_active is False
        assert hermes._canary_service is None
        assert hermes.timeline == []
        assert hermes._step == 0


# ---------------------------------------------------------------------------
# act() method tests
# ---------------------------------------------------------------------------

class TestHermesAct:
    """Tests for the HERMES.act() method."""

    def test_act_returns_deployment_action_for_degraded_service(self):
        """act() should return a CanaryDeploy action when a service is degraded."""
        hermes = HERMES(observation_window=5)

        observation = {
            "metrics_snapshot": {
                "cart-service": {"availability": False, "error_rate": 0.5},
            },
            "sla_state": {"cart-service": False},
            "incident_context": {"current_blast_radius": ["cart-service"]},
        }

        action = hermes.act(observation)

        assert action.agent == "hermes"
        assert action.category in ("deployment", "meta")

    def test_act_returns_close_incident_when_all_sla_compliant(self):
        """act() should return CloseIncident when all blast-radius services are SLA-compliant."""
        hermes = HERMES(observation_window=5)

        observation = {
            "metrics_snapshot": {
                "cart-service": {"availability": True, "error_rate": 0.01},
            },
            "sla_state": {"cart-service": True},
            "incident_context": {"current_blast_radius": ["cart-service"]},
        }

        action = hermes.act(observation)

        assert action.agent == "hermes"
        assert action.name == "CloseIncident"

    def test_act_only_emits_deployment_or_meta_actions(self):
        """HERMES must only emit deployment or meta actions (Req 11.6)."""
        hermes = HERMES(observation_window=5)

        observation = {
            "metrics_snapshot": {
                "order-service": {"availability": False, "error_rate": 0.3},
            },
            "sla_state": {"order-service": False},
            "incident_context": {"current_blast_radius": ["order-service"]},
        }

        for _ in range(10):
            action = hermes.act(observation)
            assert action.category in ("deployment", "meta"), (
                f"HERMES emitted forbidden category: {action.category}"
            )
