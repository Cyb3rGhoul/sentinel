"""Property-based tests for Observability_Layer.

Tests Properties 8, 9, 10, and 15 from the SENTINEL design document.
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from sentinel.config import ObservabilityConfig
from sentinel.models import (
    FailureType,
    HypothesisTree,
    IncidentState,
    TimelineEntry,
)
from sentinel.observability import Observability_Layer
from sentinel.world_state import ALL_SERVICES, NexaStackWorldState

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for a subset of services to treat as black-box
black_box_services_st = st.lists(
    st.sampled_from(ALL_SERVICES),
    min_size=0,
    max_size=5,
    unique=True,
)

# Strategy for a non-empty subset of services to degrade (for active incidents)
degraded_services_st = st.lists(
    st.sampled_from(ALL_SERVICES[:15]),  # only first 15 so remaining 15 stay healthy
    min_size=1,
    max_size=5,
    unique=True,
)

failure_type_st = st.sampled_from(list(FailureType))


def _make_layer(black_box_services: list[str] | None = None) -> Observability_Layer:
    cfg = ObservabilityConfig(
        black_box_services=black_box_services or ["payment-vault", "fraud-detector"],
        alert_threshold_multiplier=1.5,
    )
    return Observability_Layer(cfg)


def _make_incident_state() -> IncidentState:
    return IncidentState(
        template_id="E1",
        root_cause_service="web-gateway",
        failure_type=FailureType.cpu_spike,
        ground_truth_signals=["high cpu on web-gateway"],
        red_herring_signals=["slow query on redis-cache"],
        affected_services={"web-gateway": 0.9},
        peak_blast_radius={"web-gateway"},
        current_blast_radius={"web-gateway"},
        timeline=[],
        attempted_remediations=[],
        active_hypotheses=[],
        resolved=False,
        step_injected=0,
    )


# ---------------------------------------------------------------------------
# Property 8: Log suppression ratio is in [0.0, 0.8] and constant within episode
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 8: Log suppression ratio is in [0.0, 0.8] and constant within an episode
@given(st.integers(min_value=1, max_value=20))
@settings(max_examples=100)
def test_log_suppression_ratio_in_range_and_constant(num_steps: int):
    """After sample_episode_params(), log_suppression_ratio must be in [0.0, 0.8]
    and must not change between steps within the same episode.

    Validates: Requirements 3.1, 3.5
    """
    layer = _make_layer()
    layer.sample_episode_params()

    ratio = layer.log_suppression_ratio

    # Must be in [0.0, 0.8]
    assert 0.0 <= ratio <= 0.8, (
        f"log_suppression_ratio {ratio} is outside [0.0, 0.8]"
    )

    # Must remain constant across all steps in the episode
    ws = NexaStackWorldState()
    incident = _make_incident_state()
    ws.apply_degradation("web-gateway", 0.9)

    for step in range(num_steps):
        layer.build_observation(ws, incident, None)
        assert layer.log_suppression_ratio == ratio, (
            f"log_suppression_ratio changed at step {step}: "
            f"expected {ratio}, got {layer.log_suppression_ratio}"
        )


# ---------------------------------------------------------------------------
# Property 9: Active incidents always have 1–3 red herring alerts
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 9: Active incidents always have 1–3 red herring alerts
@given(
    degraded_services=degraded_services_st,
    red_herring_count=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=100)
def test_active_incidents_have_1_to_3_red_herring_alerts(
    degraded_services: list[str],
    red_herring_count: int,
):
    """When an incident is active, active_alerts must contain between 1 and 3
    red herring alerts (injected from healthy services).

    Validates: Requirements 3.3
    """
    layer = _make_layer()
    layer.sample_episode_params()
    # Override red_herring_count to the hypothesis-generated value
    layer.red_herring_count = red_herring_count

    ws = NexaStackWorldState()

    # Degrade only the first 15 services; the remaining 15 stay healthy
    # so there are always available services for red herring injection
    for svc in degraded_services:
        ws.apply_degradation(svc, 0.9)

    incident = _make_incident_state()
    obs = layer.build_observation(ws, incident, None)

    alerts = obs["active_alerts"]

    # Identify which services are degraded (availability=False)
    degraded_set = {
        svc for svc, m in ws.services.items() if not m.availability
    }

    # Red herring alerts come from healthy (available) services
    rh_alerts = [a for a in alerts if a.service not in degraded_set]

    # Must have between 1 and 3 red herring alerts
    assert 1 <= len(rh_alerts) <= 3, (
        f"Expected 1–3 red herring alerts, got {len(rh_alerts)}. "
        f"degraded={degraded_set}, red_herring_count={red_herring_count}"
    )

    # The actual count must match the configured red_herring_count
    # (capped by available healthy services, but we ensured enough healthy ones)
    available_healthy = sum(
        1 for svc in ALL_SERVICES if ws.services[svc].availability
    )
    expected_count = min(red_herring_count, available_healthy)
    assert len(rh_alerts) == expected_count, (
        f"Expected exactly {expected_count} red herring alerts, got {len(rh_alerts)}"
    )


# ---------------------------------------------------------------------------
# Property 10: CDG snapshot zeros out black-box service rows
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 10: CDG snapshot zeros out black-box service rows
@given(black_box_services=black_box_services_st)
@settings(max_examples=100)
def test_cdg_snapshot_zeros_black_box_rows(black_box_services: list[str]):
    """For any observation, the causal_graph_snapshot 30×30 matrix must have
    all entries in rows corresponding to black-box services set to 0.0.

    Validates: Requirements 3.4
    """
    layer = _make_layer(black_box_services)
    layer.sample_episode_params()

    ws = NexaStackWorldState()
    obs = layer.build_observation(ws, None, None)

    matrix = obs["causal_graph_snapshot"]

    # Verify matrix dimensions
    assert len(matrix) == 30, f"Expected 30 rows, got {len(matrix)}"
    for row in matrix:
        assert len(row) == 30, f"Expected 30 columns, got {len(row)}"

    # Verify black-box rows are all zeros
    for svc in black_box_services:
        row_idx = ALL_SERVICES.index(svc)
        row = matrix[row_idx]
        assert all(v == 0.0 for v in row), (
            f"Black-box service '{svc}' (row {row_idx}) has non-zero entries: "
            f"{[v for v in row if v != 0.0]}"
        )

    # Verify non-black-box rows are NOT all zeros (at least some edges exist)
    non_bb_services = [s for s in ALL_SERVICES if s not in black_box_services]
    # At least one non-black-box service should have outgoing edges
    if non_bb_services:
        non_bb_rows = [matrix[ALL_SERVICES.index(s)] for s in non_bb_services]
        has_any_edge = any(any(v != 0.0 for v in row) for row in non_bb_rows)
        assert has_any_edge, (
            "Expected at least one non-zero entry in non-black-box rows, "
            "but all rows are zero"
        )


# ---------------------------------------------------------------------------
# Property 15: Red herring alerts are not labeled in observations
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 15: Red herring alerts are not labeled in observations
@given(
    degraded_services=degraded_services_st,
    red_herring_count=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=100)
def test_red_herring_alerts_not_labeled(
    degraded_services: list[str],
    red_herring_count: int,
):
    """No alert in active_alerts may contain a field or marker that distinguishes
    it as a red herring. All alerts must appear structurally identical.

    Validates: Requirements 6.4
    """
    layer = _make_layer()
    layer.sample_episode_params()
    layer.red_herring_count = red_herring_count

    ws = NexaStackWorldState()
    for svc in degraded_services:
        ws.apply_degradation(svc, 0.9)

    incident = _make_incident_state()
    obs = layer.build_observation(ws, incident, None)

    alerts = obs["active_alerts"]

    # Forbidden field names that would identify a red herring
    forbidden_fields = {
        "is_red_herring",
        "type",
        "label",
        "kind",
        "source",
        "synthetic",
        "injected",
        "fake",
        "spurious",
        "tag",
        "category",
    }

    for alert in alerts:
        alert_dict = alert.model_dump()

        # No forbidden labeling fields
        for field in forbidden_fields:
            assert field not in alert_dict, (
                f"Alert has forbidden field '{field}' that identifies it as a red herring: "
                f"{alert_dict}"
            )

        # All alerts must have the same set of fields
        expected_fields = {"service", "metric", "value", "threshold", "timestamp", "confidence"}
        actual_fields = set(alert_dict.keys())
        assert actual_fields == expected_fields, (
            f"Alert has unexpected fields. Expected {expected_fields}, got {actual_fields}"
        )

        # All field types must be consistent
        assert isinstance(alert_dict["service"], str), "service must be str"
        assert isinstance(alert_dict["metric"], str), "metric must be str"
        assert isinstance(alert_dict["value"], float), "value must be float"
        assert isinstance(alert_dict["threshold"], float), "threshold must be float"
        assert isinstance(alert_dict["timestamp"], float), "timestamp must be float"
        assert isinstance(alert_dict["confidence"], float), "confidence must be float"
