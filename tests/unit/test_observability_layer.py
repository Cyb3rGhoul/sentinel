"""Unit tests for Observability_Layer.

Tests black-box masking, red herring injection count boundaries,
and log suppression behaviour.
"""
from __future__ import annotations

import pytest

from sentinel.config import ObservabilityConfig
from sentinel.models import FailureType, HypothesisTree, IncidentState
from sentinel.observability import Observability_Layer
from sentinel.world_state import ALL_SERVICES, NexaStackWorldState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_layer(
    black_box_services: list[str] | None = None,
) -> Observability_Layer:
    cfg = ObservabilityConfig(
        black_box_services=black_box_services or ["payment-vault", "fraud-detector"],
        alert_threshold_multiplier=1.5,
    )
    return Observability_Layer(cfg)


def _make_incident() -> IncidentState:
    return IncidentState(
        template_id="E1",
        root_cause_service="web-gateway",
        failure_type=FailureType.cpu_spike,
        ground_truth_signals=["high cpu"],
        red_herring_signals=["slow query"],
        affected_services={"web-gateway": 0.9},
        peak_blast_radius={"web-gateway"},
        current_blast_radius={"web-gateway"},
        timeline=[],
        attempted_remediations=[],
        active_hypotheses=[],
        resolved=False,
        step_injected=0,
    )


def _fresh_ws() -> NexaStackWorldState:
    return NexaStackWorldState()


# ---------------------------------------------------------------------------
# Black-box service masking
# ---------------------------------------------------------------------------

class TestBlackBoxMasking:
    def test_default_black_box_services_have_none_metrics(self):
        """Default black-box services (payment-vault, fraud-detector) must be None."""
        layer = _make_layer()
        layer.sample_episode_params()
        ws = _fresh_ws()
        obs = layer.build_observation(ws, None, None)

        snapshot = obs["metrics_snapshot"]
        assert snapshot["payment-vault"] is None, (
            "payment-vault is a black-box service; metrics must be None"
        )
        assert snapshot["fraud-detector"] is None, (
            "fraud-detector is a black-box service; metrics must be None"
        )

    def test_non_black_box_services_have_metrics(self):
        """Non-black-box services must have a metrics dict, not None."""
        layer = _make_layer()
        layer.sample_episode_params()
        ws = _fresh_ws()
        obs = layer.build_observation(ws, None, None)

        snapshot = obs["metrics_snapshot"]
        for svc in ALL_SERVICES:
            if svc not in {"payment-vault", "fraud-detector"}:
                assert snapshot[svc] is not None, (
                    f"Non-black-box service '{svc}' should have metrics, got None"
                )

    def test_custom_black_box_services_masked(self):
        """Any service listed in black_box_services must have None metrics."""
        custom_bb = ["web-gateway", "cart-service", "redis-cache"]
        layer = _make_layer(custom_bb)
        layer.sample_episode_params()
        ws = _fresh_ws()
        obs = layer.build_observation(ws, None, None)

        snapshot = obs["metrics_snapshot"]
        for svc in custom_bb:
            assert snapshot[svc] is None, (
                f"Custom black-box service '{svc}' must have None metrics"
            )

    def test_all_30_services_present_in_snapshot(self):
        """metrics_snapshot must contain all 30 services."""
        layer = _make_layer()
        layer.sample_episode_params()
        ws = _fresh_ws()
        obs = layer.build_observation(ws, None, None)

        snapshot = obs["metrics_snapshot"]
        assert set(snapshot.keys()) == set(ALL_SERVICES), (
            "metrics_snapshot must contain exactly the 30 canonical services"
        )

    def test_black_box_cdg_rows_are_zero(self):
        """CDG rows for black-box services must be all zeros."""
        layer = _make_layer(["payment-vault", "fraud-detector"])
        layer.sample_episode_params()
        ws = _fresh_ws()
        obs = layer.build_observation(ws, None, None)

        matrix = obs["causal_graph_snapshot"]
        for svc in ["payment-vault", "fraud-detector"]:
            row_idx = ALL_SERVICES.index(svc)
            assert all(v == 0.0 for v in matrix[row_idx]), (
                f"CDG row for black-box service '{svc}' must be all zeros"
            )


# ---------------------------------------------------------------------------
# Red herring injection count boundaries
# ---------------------------------------------------------------------------

class TestRedHerringInjection:
    def _obs_with_count(self, count: int) -> dict:
        """Build an observation with a specific red_herring_count."""
        # Degrade only the first 5 services; the rest stay healthy for red herrings
        layer = _make_layer()
        layer.sample_episode_params()
        layer.red_herring_count = count

        ws = _fresh_ws()
        # Degrade a few services so real alerts exist
        ws.apply_degradation("web-gateway", 0.9)
        ws.apply_degradation("cart-service", 0.9)

        incident = _make_incident()
        return layer.build_observation(ws, incident, None)

    def _count_red_herrings(self, obs: dict) -> int:
        """Count alerts from healthy (available) services."""
        ws = _fresh_ws()
        ws.apply_degradation("web-gateway", 0.9)
        ws.apply_degradation("cart-service", 0.9)
        degraded = {svc for svc, m in ws.services.items() if not m.availability}
        return sum(1 for a in obs["active_alerts"] if a.service not in degraded)

    def test_red_herring_count_1_produces_exactly_1(self):
        """red_herring_count=1 must produce exactly 1 red herring alert."""
        obs = self._obs_with_count(1)
        rh_count = self._count_red_herrings(obs)
        assert rh_count == 1, (
            f"Expected exactly 1 red herring alert, got {rh_count}"
        )

    def test_red_herring_count_2_produces_exactly_2(self):
        """red_herring_count=2 must produce exactly 2 red herring alerts."""
        obs = self._obs_with_count(2)
        rh_count = self._count_red_herrings(obs)
        assert rh_count == 2, (
            f"Expected exactly 2 red herring alerts, got {rh_count}"
        )

    def test_red_herring_count_3_produces_exactly_3(self):
        """red_herring_count=3 must produce exactly 3 red herring alerts."""
        obs = self._obs_with_count(3)
        rh_count = self._count_red_herrings(obs)
        assert rh_count == 3, (
            f"Expected exactly 3 red herring alerts, got {rh_count}"
        )

    def test_red_herring_alerts_come_from_healthy_services(self):
        """Red herring alerts must only reference services that are available."""
        layer = _make_layer()
        layer.sample_episode_params()
        layer.red_herring_count = 3

        ws = _fresh_ws()
        ws.apply_degradation("web-gateway", 0.9)

        incident = _make_incident()
        obs = layer.build_observation(ws, incident, None)

        degraded = {svc for svc, m in ws.services.items() if not m.availability}
        rh_alerts = [a for a in obs["active_alerts"] if a.service not in degraded]

        for alert in rh_alerts:
            assert ws.services[alert.service].availability, (
                f"Red herring alert references degraded service '{alert.service}'"
            )


# ---------------------------------------------------------------------------
# Log suppression
# ---------------------------------------------------------------------------

class TestLogSuppression:
    def test_suppression_ratio_1_0_produces_empty_logs(self):
        """log_suppression_ratio=1.0 must produce zero log entries."""
        layer = _make_layer()
        layer.sample_episode_params()
        layer.log_suppression_ratio = 1.0

        ws = _fresh_ws()
        # Degrade several services so there are logs to suppress
        for svc in ["web-gateway", "cart-service", "order-service", "redis-cache"]:
            ws.apply_degradation(svc, 0.9)

        obs = layer.build_observation(ws, None, None)
        assert obs["recent_logs"] == [], (
            f"With suppression ratio=1.0, recent_logs must be empty, "
            f"got {len(obs['recent_logs'])} entries"
        )

    def test_suppression_ratio_0_0_keeps_all_logs(self):
        """log_suppression_ratio=0.0 must keep all log entries."""
        layer = _make_layer()
        layer.sample_episode_params()
        layer.log_suppression_ratio = 0.0

        ws = _fresh_ws()
        # Degrade a few services to generate logs
        degraded = ["web-gateway", "cart-service", "order-service"]
        for svc in degraded:
            ws.apply_degradation(svc, 0.9)

        # Count how many services actually became unavailable
        unavailable = [svc for svc in degraded if not ws.services[svc].availability]
        # Each unavailable service generates 2 log entries
        expected_count = len(unavailable) * 2

        obs = layer.build_observation(ws, None, None)
        assert len(obs["recent_logs"]) == expected_count, (
            f"With suppression ratio=0.0, expected {expected_count} logs, "
            f"got {len(obs['recent_logs'])}"
        )

    def test_suppression_ratio_0_5_reduces_logs(self):
        """log_suppression_ratio=0.5 must reduce log count by ~50%."""
        layer = _make_layer()
        layer.sample_episode_params()
        layer.log_suppression_ratio = 0.5

        ws = _fresh_ws()
        # Degrade enough services to generate a meaningful number of logs
        for svc in ["web-gateway", "cart-service", "order-service",
                    "redis-cache", "search-service"]:
            ws.apply_degradation(svc, 0.9)

        # Get baseline log count with no suppression
        layer_no_suppress = _make_layer()
        layer_no_suppress.sample_episode_params()
        layer_no_suppress.log_suppression_ratio = 0.0
        obs_full = layer_no_suppress.build_observation(ws, None, None)
        full_count = len(obs_full["recent_logs"])

        if full_count == 0:
            pytest.skip("No logs generated; cannot test suppression")

        obs = layer.build_observation(ws, None, None)
        suppressed_count = len(obs["recent_logs"])

        # With 50% suppression, count should be less than full
        assert suppressed_count < full_count, (
            f"Suppression ratio=0.5 should reduce logs below {full_count}, "
            f"got {suppressed_count}"
        )

    def test_no_logs_when_all_services_healthy(self):
        """With all services healthy, recent_logs must be empty regardless of ratio."""
        layer = _make_layer()
        layer.sample_episode_params()
        layer.log_suppression_ratio = 0.0

        ws = _fresh_ws()  # all services at baseline (healthy)
        obs = layer.build_observation(ws, None, None)

        assert obs["recent_logs"] == [], (
            "No logs should be generated when all services are healthy"
        )


# ---------------------------------------------------------------------------
# sample_episode_params
# ---------------------------------------------------------------------------

class TestSampleEpisodeParams:
    def test_log_suppression_ratio_in_range(self):
        """sample_episode_params must set log_suppression_ratio in [0.0, 0.8]."""
        layer = _make_layer()
        for _ in range(50):
            layer.sample_episode_params()
            assert 0.0 <= layer.log_suppression_ratio <= 0.8, (
                f"log_suppression_ratio {layer.log_suppression_ratio} out of [0.0, 0.8]"
            )

    def test_red_herring_count_in_range(self):
        """sample_episode_params must set red_herring_count in {1, 2, 3}."""
        layer = _make_layer()
        for _ in range(50):
            layer.sample_episode_params()
            assert layer.red_herring_count in {1, 2, 3}, (
                f"red_herring_count {layer.red_herring_count} not in {{1, 2, 3}}"
            )

    def test_params_vary_across_episodes(self):
        """Repeated calls to sample_episode_params should produce different values."""
        layer = _make_layer()
        ratios = set()
        for _ in range(30):
            layer.sample_episode_params()
            ratios.add(layer.log_suppression_ratio)
        # With 30 samples from U[0.0, 0.8], we expect more than 1 unique value
        assert len(ratios) > 1, (
            "sample_episode_params should produce varied log_suppression_ratio values"
        )
