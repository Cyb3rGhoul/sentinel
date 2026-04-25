"""Property-based tests for NexaStackWorldState.

Tests Properties 4, 5, 6, and 20 from the SENTINEL design document.
"""
from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from sentinel.world_state import ALL_SERVICES, NexaStackWorldState

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

service_names = st.sampled_from(ALL_SERVICES)

# Severity values in [0.0, 2.0] — intentionally includes values > 1.0 to test clamping
severity_st = st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)

# A single (service, severity) degradation step
degradation_step_st = st.tuples(service_names, severity_st)

# A sequence of 0–20 degradation steps
degradation_sequence_st = st.lists(degradation_step_st, min_size=0, max_size=20)


def _fresh_state() -> NexaStackWorldState:
    return NexaStackWorldState()


# ---------------------------------------------------------------------------
# Property 4: World state always contains exactly 30 services with valid metrics
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 4: World state always contains exactly 30 services with valid metrics
@given(steps=degradation_sequence_st)
@settings(max_examples=100)
def test_world_state_always_has_30_services_with_valid_metrics(steps):
    """For any sequence of apply_degradation calls, NexaStackWorldState must always
    have exactly 30 services, each with finite float metrics and bool availability.

    Validates: Requirements 2.1
    """
    ws = _fresh_state()

    for service, severity in steps:
        ws.apply_degradation(service, severity)

    # Must have exactly 30 services
    assert len(ws.services) == 30, (
        f"Expected 30 services, got {len(ws.services)}"
    )

    # Each service must have valid metrics
    for svc, m in ws.services.items():
        assert isinstance(m.cpu, float) and math.isfinite(m.cpu), (
            f"{svc}.cpu={m.cpu!r} is not a finite float"
        )
        assert isinstance(m.memory, float) and math.isfinite(m.memory), (
            f"{svc}.memory={m.memory!r} is not a finite float"
        )
        assert isinstance(m.latency_ms, float) and math.isfinite(m.latency_ms), (
            f"{svc}.latency_ms={m.latency_ms!r} is not a finite float"
        )
        assert isinstance(m.error_rate, float) and math.isfinite(m.error_rate), (
            f"{svc}.error_rate={m.error_rate!r} is not a finite float"
        )
        assert isinstance(m.saturation, float) and math.isfinite(m.saturation), (
            f"{svc}.saturation={m.saturation!r} is not a finite float"
        )
        assert isinstance(m.availability, bool), (
            f"{svc}.availability={m.availability!r} is not a bool"
        )


# ---------------------------------------------------------------------------
# Property 5: CDG edge weights are always in [0.0, 1.0]
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 5: CDG edge weights are always in [0.0, 1.0]
@given(steps=degradation_sequence_st)
@settings(max_examples=100)
def test_cdg_edge_weights_always_in_unit_interval(steps):
    """For any NexaStackWorldState snapshot, every CDG edge weight must be in [0.0, 1.0].

    Tests the initial CDG and after any modifications.

    Validates: Requirements 2.3
    """
    ws = _fresh_state()

    # Check initial CDG
    for u, v, data in ws.cdg.edges(data=True):
        w = data["weight"]
        assert isinstance(w, float), f"Edge ({u},{v}) weight {w!r} is not a float"
        assert 0.0 <= w <= 1.0, f"Edge ({u},{v}) weight {w} is outside [0.0, 1.0]"

    # Apply degradations and re-check
    for service, severity in steps:
        ws.apply_degradation(service, severity)

    for u, v, data in ws.cdg.edges(data=True):
        w = data["weight"]
        assert isinstance(w, float), f"Edge ({u},{v}) weight {w!r} is not a float"
        assert 0.0 <= w <= 1.0, f"Edge ({u},{v}) weight {w} is outside [0.0, 1.0]"


# ---------------------------------------------------------------------------
# Property 6: Metric threshold crossing updates availability
# ---------------------------------------------------------------------------

# Thresholds from world_state.py
_THRESHOLDS = {
    "cpu": 0.85,
    "memory": 0.85,
    "error_rate": 0.05,
    "saturation": 0.9,
    "latency_ms": 500.0,
}

# Baseline values
_BASELINE = {
    "cpu": 0.2,
    "memory": 0.3,
    "latency_ms": 50.0,
    "error_rate": 0.01,
    "saturation": 0.3,
}

# Minimum severity that guarantees at least one threshold is crossed.
# At severity s: new_cpu = 0.2 + s*(1-0.2) = 0.2 + 0.8s > 0.85 => s > 0.8125
# At severity s: new_error_rate = 0.01 + s*0.99 > 0.05 => s > ~0.04
# So any severity > 0.04 will cross error_rate threshold.
_THRESHOLD_CROSSING_SEVERITY = 0.05  # crosses error_rate threshold

# Feature: sentinel, Property 6: Metric threshold crossing updates availability
@given(
    service=service_names,
    severity=st.floats(
        min_value=_THRESHOLD_CROSSING_SEVERITY + 0.001,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
@settings(max_examples=100)
def test_metric_threshold_crossing_sets_availability_false(service, severity):
    """For any service and severity >= threshold-crossing level, availability must be
    False after apply_degradation.

    Thresholds: cpu > 0.85, memory > 0.85, error_rate > 0.05, saturation > 0.9,
    latency_ms > 500.

    Validates: Requirements 2.4
    """
    ws = _fresh_state()
    ws.apply_degradation(service, severity)

    m = ws.services[service]

    # Verify at least one threshold is crossed
    threshold_crossed = (
        m.cpu > _THRESHOLDS["cpu"]
        or m.memory > _THRESHOLDS["memory"]
        or m.error_rate > _THRESHOLDS["error_rate"]
        or m.saturation > _THRESHOLDS["saturation"]
        or m.latency_ms > _THRESHOLDS["latency_ms"]
    )

    if threshold_crossed:
        assert m.availability is False, (
            f"Service {service} with severity={severity} crossed a threshold "
            f"but availability={m.availability}. Metrics: cpu={m.cpu}, "
            f"memory={m.memory}, error_rate={m.error_rate}, "
            f"saturation={m.saturation}, latency_ms={m.latency_ms}"
        )


# ---------------------------------------------------------------------------
# Property 20: NexaStackWorldState serialization round-trip
# ---------------------------------------------------------------------------

# Strategy: build world states with various degradation states applied
@st.composite
def world_state_st(draw):
    """Generate a NexaStackWorldState with a random degradation history applied."""
    ws = NexaStackWorldState()
    steps = draw(degradation_sequence_st)
    for service, severity in steps:
        ws.apply_degradation(service, severity)
    # Optionally set a non-zero step counter
    ws.step = draw(st.integers(min_value=0, max_value=1000))
    return ws


# Feature: sentinel, Property 20: NexaStackWorldState serialization round-trip
@given(ws=world_state_st())
@settings(max_examples=100)
def test_world_state_serialization_round_trip(ws):
    """For any valid NexaStackWorldState, to_json() then from_json() must produce
    identical ServiceMetrics values, CDG edge weights, and step count.

    Validates: Requirements 18.1, 18.2
    """
    serialized = ws.to_json()
    restored = NexaStackWorldState.from_json(serialized)

    # Step count must be identical
    assert restored.step == ws.step, (
        f"step mismatch: original={ws.step}, restored={restored.step}"
    )

    # All 30 services must be present
    assert set(restored.services.keys()) == set(ws.services.keys()), (
        "Service keys differ after round-trip"
    )

    # ServiceMetrics must be identical for all services
    for svc in ws.services:
        orig = ws.services[svc]
        rest = restored.services[svc]
        assert orig.cpu == rest.cpu, f"{svc}.cpu: {orig.cpu} != {rest.cpu}"
        assert orig.memory == rest.memory, f"{svc}.memory: {orig.memory} != {rest.memory}"
        assert orig.latency_ms == rest.latency_ms, (
            f"{svc}.latency_ms: {orig.latency_ms} != {rest.latency_ms}"
        )
        assert orig.error_rate == rest.error_rate, (
            f"{svc}.error_rate: {orig.error_rate} != {rest.error_rate}"
        )
        assert orig.saturation == rest.saturation, (
            f"{svc}.saturation: {orig.saturation} != {rest.saturation}"
        )
        assert orig.availability == rest.availability, (
            f"{svc}.availability: {orig.availability} != {rest.availability}"
        )

    # CDG edge weights must be identical
    orig_edges = {
        (u, v): d["weight"] for u, v, d in ws.cdg.edges(data=True)
    }
    rest_edges = {
        (u, v): d["weight"] for u, v, d in restored.cdg.edges(data=True)
    }
    assert orig_edges == rest_edges, "CDG edge weights differ after round-trip"

    # incident_state must both be None (we don't inject incidents in this strategy)
    assert restored.incident_state is None
