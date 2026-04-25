"""Unit tests for NexaStackWorldState.

Covers Task 2.6: restore_baseline, apply_degradation clamping, snapshot
serialisability, severity=1.0 availability, and incident_state clearing.
"""
from __future__ import annotations

import json
import math

import pytest

from sentinel.world_state import (
    ALL_SERVICES,
    NexaStackWorldState,
    _BASELINE,
    _THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh() -> NexaStackWorldState:
    return NexaStackWorldState()


# ---------------------------------------------------------------------------
# restore_baseline()
# ---------------------------------------------------------------------------

class TestRestoreBaseline:
    def test_resets_all_30_services_to_baseline_values(self):
        """restore_baseline() must reset every service to the healthy baseline."""
        ws = _fresh()
        # Degrade a handful of services
        for svc in ALL_SERVICES[:5]:
            ws.apply_degradation(svc, 1.0)

        ws.restore_baseline()

        assert len(ws.services) == 30
        for svc, m in ws.services.items():
            assert m.cpu == _BASELINE["cpu"], f"{svc}.cpu not reset"
            assert m.memory == _BASELINE["memory"], f"{svc}.memory not reset"
            assert m.latency_ms == _BASELINE["latency_ms"], f"{svc}.latency_ms not reset"
            assert m.error_rate == _BASELINE["error_rate"], f"{svc}.error_rate not reset"
            assert m.saturation == _BASELINE["saturation"], f"{svc}.saturation not reset"
            assert m.availability is True, f"{svc}.availability not reset to True"

    def test_resets_step_counter_to_zero(self):
        ws = _fresh()
        ws.step = 42
        ws.restore_baseline()
        assert ws.step == 0

    def test_clears_incident_state(self):
        """restore_baseline() must set incident_state to None."""
        from sentinel.models import (
            FailureType,
            HypothesisNode,
            IncidentState,
        )

        ws = _fresh()
        ws.incident_state = IncidentState(
            template_id="E1",
            root_cause_service="cart-service",
            failure_type=FailureType.memory_leak,
            ground_truth_signals=["high_memory"],
            red_herring_signals=["cpu_spike"],
            affected_services={"cart-service": 1.0},
            peak_blast_radius={"cart-service"},
            current_blast_radius={"cart-service"},
            timeline=[],
            attempted_remediations=[],
            active_hypotheses=[],
            resolved=False,
            step_injected=0,
        )

        ws.restore_baseline()
        assert ws.incident_state is None


# ---------------------------------------------------------------------------
# apply_degradation() — severity clamping
# ---------------------------------------------------------------------------

class TestApplyDegradation:
    def test_severity_clamped_to_1_0(self):
        """Passing severity=2.0 must produce the same result as severity=1.0."""
        ws1 = _fresh()
        ws2 = _fresh()

        ws1.apply_degradation("cart-service", 1.0)
        ws2.apply_degradation("cart-service", 2.0)

        m1 = ws1.services["cart-service"]
        m2 = ws2.services["cart-service"]

        assert m1.cpu == m2.cpu
        assert m1.memory == m2.memory
        assert m1.latency_ms == m2.latency_ms
        assert m1.error_rate == m2.error_rate
        assert m1.saturation == m2.saturation
        assert m1.availability == m2.availability

    def test_severity_1_sets_availability_false(self):
        """apply_degradation with severity=1.0 must set availability=False."""
        ws = _fresh()
        ws.apply_degradation("web-gateway", 1.0)
        assert ws.services["web-gateway"].availability is False

    def test_severity_0_does_not_change_availability(self):
        """apply_degradation with severity=0.0 should not degrade metrics."""
        ws = _fresh()
        ws.apply_degradation("web-gateway", 0.0)
        m = ws.services["web-gateway"]
        # At severity=0 metrics stay at baseline — availability should remain True
        assert m.availability is True

    def test_unknown_service_raises(self):
        """apply_degradation on an unknown service must raise CascadeError."""
        from sentinel.exceptions import CascadeError

        ws = _fresh()
        with pytest.raises(CascadeError):
            ws.apply_degradation("nonexistent-service", 0.5)

    def test_metrics_are_finite_after_degradation(self):
        """All metrics must remain finite floats after degradation."""
        ws = _fresh()
        for svc in ALL_SERVICES:
            ws.apply_degradation(svc, 0.75)
            m = ws.services[svc]
            assert math.isfinite(m.cpu)
            assert math.isfinite(m.memory)
            assert math.isfinite(m.latency_ms)
            assert math.isfinite(m.error_rate)
            assert math.isfinite(m.saturation)


# ---------------------------------------------------------------------------
# snapshot()
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_snapshot_is_json_serializable(self):
        """snapshot() must return a dict that json.dumps does not raise on."""
        ws = _fresh()
        snap = ws.snapshot()
        # Should not raise
        json.dumps(snap)

    def test_snapshot_after_degradation_is_json_serializable(self):
        ws = _fresh()
        for svc in ALL_SERVICES[:10]:
            ws.apply_degradation(svc, 0.5)
        snap = ws.snapshot()
        json.dumps(snap)

    def test_snapshot_contains_required_keys(self):
        ws = _fresh()
        snap = ws.snapshot()
        assert "services" in snap
        assert "cdg_edges" in snap
        assert "incident_state" in snap
        assert "step" in snap

    def test_snapshot_has_30_services(self):
        ws = _fresh()
        snap = ws.snapshot()
        assert len(snap["services"]) == 30
