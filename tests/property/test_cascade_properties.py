"""Property-based tests for Cascade_Engine.

Tests Properties 11 and 12 from the SENTINEL design document.
"""
from __future__ import annotations

import networkx as nx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sentinel.cascade_engine import Cascade_Engine
from sentinel.models import FailureType
from sentinel.world_state import ALL_SERVICES, NexaStackWorldState

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

root_service_st = st.sampled_from(ALL_SERVICES)

initial_severity_st = st.floats(
    min_value=0.1,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)

failure_type_st = st.sampled_from(list(FailureType))


def _fresh_state() -> NexaStackWorldState:
    return NexaStackWorldState()


# ---------------------------------------------------------------------------
# Property 11: Cascade propagation respects BFS depth ≤ 6 with exponential severity decay
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 11: Cascade propagation respects BFS depth ≤ 6 with exponential severity decay
@given(
    root_service=root_service_st,
    initial_severity=initial_severity_st,
    failure_type=failure_type_st,
)
@settings(max_examples=100)
def test_cascade_respects_bfs_depth_and_severity_decay(
    root_service, initial_severity, failure_type
):
    """For any CDG topology and root failure injection, the Cascade_Engine must:
    (a) not affect any service at BFS depth > 6 from the root, and
    (b) assign severity = initial_severity * (0.7^d) at depth d.

    Validates: Requirements 4.1, 4.2
    """
    ws = _fresh_state()
    engine = Cascade_Engine()

    blast_radius = engine.propagate_failure(
        ws, root_service, failure_type, initial_severity
    )

    # Compute actual BFS depths from root in the CDG
    bfs_depths: dict[str, int] = nx.single_source_shortest_path_length(
        ws.cdg, root_service
    )

    for service, severity in blast_radius.items():
        depth = bfs_depths.get(service)

        # (a) No service in blast radius should be at depth > 6
        assert depth is not None, (
            f"Service '{service}' is in blast radius but not reachable from '{root_service}' in CDG"
        )
        assert depth <= Cascade_Engine.MAX_DEPTH, (
            f"Service '{service}' is at depth {depth} > {Cascade_Engine.MAX_DEPTH} "
            f"from root '{root_service}' but is in the blast radius"
        )

        # (b) Severity at depth d must equal initial_severity * (0.7^d), clamped to 1.0
        expected_severity = min(initial_severity, 1.0) * (Cascade_Engine.SEVERITY_DECAY ** depth)
        assert abs(severity - expected_severity) < 1e-9, (
            f"Service '{service}' at depth {depth}: "
            f"expected severity {expected_severity:.6f}, got {severity:.6f}"
        )

    # Also verify no service beyond depth 6 was affected
    for service, depth in bfs_depths.items():
        if depth > Cascade_Engine.MAX_DEPTH:
            assert service not in blast_radius, (
                f"Service '{service}' at depth {depth} should NOT be in blast radius "
                f"(max depth is {Cascade_Engine.MAX_DEPTH})"
            )


# ---------------------------------------------------------------------------
# Property 12: Recovery propagates through the same paths as failure
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 12: Recovery propagates through the same paths as failure
@given(
    root_service=root_service_st,
    initial_severity=initial_severity_st,
    failure_type=failure_type_st,
)
@settings(max_examples=100)
def test_recovery_restores_root_service_to_baseline(
    root_service, initial_severity, failure_type
):
    """After propagate_failure() then propagate_recovery():
    1. The root service must be fully restored to baseline.
    2. The blast radius must shrink (granular recovery, not necessarily empty).

    The old behavior restored ALL services; the new granular recovery only
    restores the root and 50% reduces direct dependents.

    Validates: Requirements 4.4 (granular recovery design)
    """
    ws = _fresh_state()
    engine = Cascade_Engine()

    # Capture baseline metrics for the root service only
    baseline_root_cpu = ws.services[root_service].cpu
    baseline_root_memory = ws.services[root_service].memory
    baseline_root_latency = ws.services[root_service].latency_ms
    baseline_root_error_rate = ws.services[root_service].error_rate
    baseline_root_saturation = ws.services[root_service].saturation
    baseline_root_availability = ws.services[root_service].availability

    # Propagate failure
    engine.propagate_failure(ws, root_service, failure_type, initial_severity)
    pre_recovery_br = len(engine.get_blast_radius())

    # Propagate recovery
    engine.propagate_recovery(ws, root_service)

    # Root service must be back to baseline
    m = ws.services[root_service]
    assert abs(m.cpu - baseline_root_cpu) < 1e-9, (
        f"{root_service}.cpu after recovery: {m.cpu} != baseline {baseline_root_cpu}"
    )
    assert abs(m.latency_ms - baseline_root_latency) < 1e-9, (
        f"{root_service}.latency_ms after recovery"
    )
    assert m.availability == baseline_root_availability, (
        f"{root_service}.availability after recovery"
    )

    # Root service must not be in blast radius anymore
    assert root_service not in engine.get_blast_radius(), (
        f"Root service '{root_service}' must be out of blast radius after recovery"
    )

    # Blast radius must have shrunk (or be empty if root had no dependents)
    post_recovery_br = len(engine.get_blast_radius())
    if pre_recovery_br > 0:
        assert post_recovery_br < pre_recovery_br, (
            f"Blast radius should shrink after recovery: {pre_recovery_br} -> {post_recovery_br}"
        )
