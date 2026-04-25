"""Property-based tests for Reward_Function.

Tests Properties 16, 17, 18, and 19 from the SENTINEL design document.
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from sentinel.models import (
    Action,
    FailureType,
    IncidentState,
    RewardBreakdown,
    RewardWeights,
    ServiceMetrics,
    Trajectory,
    TrajectoryStep,
)
from sentinel.reward import Reward_Function
from sentinel.world_state import ALL_SERVICES, NexaStackWorldState

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = RewardWeights(
    r1_root_cause=0.35,
    r2_mttr=0.30,
    r3_recovery_quality=0.25,
    r4_blast_radius=0.10,
)

_SLA_THRESHOLD = 50


def _make_reward_fn() -> Reward_Function:
    return Reward_Function(weights=_DEFAULT_WEIGHTS, sla_breach_threshold=_SLA_THRESHOLD)


def _make_incident_state(
    root_cause_service: str = "redis-cache",
    failure_type: FailureType = FailureType.memory_leak,
    peak_blast_radius: set[str] | None = None,
    current_blast_radius: set[str] | None = None,
) -> IncidentState:
    if peak_blast_radius is None:
        peak_blast_radius = set()
    if current_blast_radius is None:
        current_blast_radius = set()
    return IncidentState(
        template_id="E1",
        root_cause_service=root_cause_service,
        failure_type=failure_type,
        ground_truth_signals=[],
        red_herring_signals=[],
        affected_services={},
        peak_blast_radius=peak_blast_radius,
        current_blast_radius=current_blast_radius,
        timeline=[],
        attempted_remediations=[],
        active_hypotheses=[],
        resolved=False,
        step_injected=0,
    )


def _make_trajectory(
    mttr: int,
    identified_root_cause: str = "",
    identified_failure_type: str = "",
) -> Trajectory:
    action = Action(
        agent="holmes",
        category="meta",
        name="CloseIncident",
        params={"resolution_summary": "done"},
    )
    step = TrajectoryStep(
        observation={},
        action=action,
        reward=0.0,
        terminated=True,
        truncated=False,
        info={
            "identified_root_cause": identified_root_cause,
            "identified_failure_type": identified_failure_type,
        },
    )
    return Trajectory(
        episode_id="test-ep",
        incident_template_id="E1",
        steps=[step],
        final_reward=RewardBreakdown(r1=0.0, r2=0.0, r3=0.0, r4=0.0, penalties=0.0, total=0.0),
        mttr=mttr,
    )


def _baseline_world_state() -> NexaStackWorldState:
    """Return a world state where all services are at baseline (healthy)."""
    ws = NexaStackWorldState()
    ws.restore_baseline()
    return ws


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# R1 ∈ {0.0, 0.5, 1.0}
r1_st = st.sampled_from([0.0, 0.5, 1.0])

# R2 ∈ [0.0, 1.1]
r2_st = st.floats(min_value=0.0, max_value=1.1, allow_nan=False, allow_infinity=False)

# R3 ∈ [0.0, 1.0]
r3_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# R4 ∈ [0.0, 1.0]
r4_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Penalties ∈ {0.0, -0.5}
penalties_st = st.sampled_from([0.0, -0.5])

# MTTR values for late-resolution tests
mttr_above_threshold_st = st.integers(
    min_value=2 * _SLA_THRESHOLD + 1, max_value=2 * _SLA_THRESHOLD + 500
)
mttr_at_or_below_threshold_st = st.integers(
    min_value=0, max_value=2 * _SLA_THRESHOLD
)

# Service names
service_st = st.sampled_from(ALL_SERVICES)

# Blast radius sets (subsets of ALL_SERVICES)
small_br_st = st.sets(st.sampled_from(ALL_SERVICES), min_size=1, max_size=5)


# ---------------------------------------------------------------------------
# Property 17: Episode reward equals weighted sum of components plus penalties
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 17: Episode reward equals the weighted sum of components plus penalties
@given(
    r1=r1_st,
    r2=r2_st,
    r3=r3_st,
    r4=r4_st,
    penalties=penalties_st,
)
@settings(max_examples=100)
def test_episode_reward_equals_weighted_sum(r1, r2, r3, r4, penalties):
    """For any completed episode, total = 0.35*R1 + 0.30*R2 + 0.25*R3 + 0.10*R4 + penalties.

    Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5
    """
    expected_total = (
        0.35 * r1
        + 0.30 * r2
        + 0.25 * r3
        + 0.10 * r4
        + penalties
    )

    breakdown = RewardBreakdown(
        r1=r1,
        r2=r2,
        r3=r3,
        r4=r4,
        penalties=penalties,
        total=expected_total,
    )

    # Verify the formula holds
    computed = (
        _DEFAULT_WEIGHTS.r1_root_cause * breakdown.r1
        + _DEFAULT_WEIGHTS.r2_mttr * breakdown.r2
        + _DEFAULT_WEIGHTS.r3_recovery_quality * breakdown.r3
        + _DEFAULT_WEIGHTS.r4_blast_radius * breakdown.r4
        + breakdown.penalties
    )

    assert abs(computed - breakdown.total) < 1e-9, (
        f"total={breakdown.total:.6f} != weighted_sum={computed:.6f} "
        f"(r1={r1}, r2={r2}, r3={r3}, r4={r4}, penalties={penalties})"
    )

    # Also verify compute_episode_reward produces the same formula
    rf = _make_reward_fn()
    ws = _baseline_world_state()

    # Build an incident state that will produce the desired R1
    # For R1=1.0: correct service + type; R1=0.5: correct service only; R1=0.0: wrong service
    if r1 == 1.0:
        root_cause_service = "redis-cache"
        failure_type = FailureType.memory_leak
        identified_root_cause = "redis-cache"
        identified_failure_type = "memory_leak"
    elif r1 == 0.5:
        root_cause_service = "redis-cache"
        failure_type = FailureType.memory_leak
        identified_root_cause = "redis-cache"
        identified_failure_type = "cpu_spike"  # wrong type
    else:  # r1 == 0.0
        root_cause_service = "redis-cache"
        failure_type = FailureType.memory_leak
        identified_root_cause = "postgres-primary"  # wrong service
        identified_failure_type = "memory_leak"

    # Build incident state with peak/current blast radius to produce desired R4
    # R4 = 1 - current/peak; to get R4 in [0,1] we use peak=1, current=round((1-r4)*1)
    # Use peak=10 services for finer granularity
    peak_size = 10
    current_size = round((1.0 - r4) * peak_size)
    current_size = max(0, min(peak_size, current_size))
    peak_br = set(ALL_SERVICES[:peak_size])
    current_br = set(ALL_SERVICES[:current_size])

    incident_state = _make_incident_state(
        root_cause_service=root_cause_service,
        failure_type=failure_type,
        peak_blast_radius=peak_br,
        current_blast_radius=current_br,
    )

    # Build MTTR to produce desired R2 and penalty
    # We can't perfectly control R2 from mttr alone, so we just verify the formula
    # holds for the actual computed breakdown
    mttr = 0  # fast resolution, no late penalty
    trajectory = _make_trajectory(
        mttr=mttr,
        identified_root_cause=identified_root_cause,
        identified_failure_type=identified_failure_type,
    )

    result = rf.compute_episode_reward(trajectory, ws, incident_state)

    # The key invariant: total == weighted sum of components + penalties
    expected = (
        _DEFAULT_WEIGHTS.r1_root_cause * result.r1
        + _DEFAULT_WEIGHTS.r2_mttr * result.r2
        + _DEFAULT_WEIGHTS.r3_recovery_quality * result.r3
        + _DEFAULT_WEIGHTS.r4_blast_radius * result.r4
        + result.penalties
    )
    assert abs(result.total - expected) < 1e-9, (
        f"total={result.total:.6f} != weighted_sum={expected:.6f}"
    )

    # R1 must be in {0.0, 0.5, 1.0}
    assert result.r1 in (0.0, 0.5, 1.0), f"R1={result.r1} not in {{0.0, 0.5, 1.0}}"

    # R2 must be in [0.0, 1.1]
    assert 0.0 <= result.r2 <= 1.1, f"R2={result.r2} not in [0.0, 1.1]"

    # R3 must be in [0.0, 1.0]
    assert 0.0 <= result.r3 <= 1.0, f"R3={result.r3} not in [0.0, 1.0]"

    # R4 must be in [0.0, 1.0]
    assert 0.0 <= result.r4 <= 1.0, f"R4={result.r4} not in [0.0, 1.0]"


# ---------------------------------------------------------------------------
# Property 18: Blast radius expansion triggers a -1.0 penalty
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 18: Blast radius expansion triggers a -1.0 penalty
@given(
    pre_br=small_br_st,
    extra_services=st.sets(st.sampled_from(ALL_SERVICES), min_size=1, max_size=3),
)
@settings(max_examples=100)
def test_blast_radius_expansion_triggers_penalty(pre_br, extra_services):
    """For any remediation action that causes current_blast_radius to expand,
    step reward must include -1.0.

    Validates: Requirements 13.6
    """
    rf = _make_reward_fn()

    # Post-action blast radius is strictly larger than pre-action
    post_br = pre_br | extra_services  # union always >= pre_br

    # Ensure post_br is strictly larger (extra_services may overlap with pre_br)
    if len(post_br) <= len(pre_br):
        # All extra_services were already in pre_br; add a guaranteed new one
        new_service = next(s for s in ALL_SERVICES if s not in pre_br)
        post_br = pre_br | {new_service}

    # Pre-action incident state (smaller blast radius)
    pre_incident_state = _make_incident_state(
        current_blast_radius=set(pre_br),
    )

    # Post-action world state (larger blast radius)
    ws = _baseline_world_state()
    ws.incident_state = _make_incident_state(
        current_blast_radius=set(post_br),
    )

    # Pick a service NOT in any blast radius, so no +0.05 affected-service shaping applies
    target_service = next(
        (s for s in ALL_SERVICES if s not in pre_br and s not in post_br),
        "audit-log",  # fallback — always last in list, rarely in small blast radii
    )

    action = Action(
        agent="forge",
        category="remediation",
        name="ScaleService",
        params={"service": target_service, "replicas": 3},
    )

    step_reward = rf.compute_step_reward(action, ws, pre_incident_state)

    assert step_reward <= -1.0, (
        f"Expected step_reward <= -1.0 when blast radius expanded "
        f"from {len(pre_br)} to {len(post_br)} services, got {step_reward}"
    )


# ---------------------------------------------------------------------------
# Property 19: Late resolution triggers a -0.5 penalty
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 19: Late resolution triggers a -0.5 penalty
@given(
    mttr=mttr_above_threshold_st,
)
@settings(max_examples=100)
def test_late_resolution_triggers_penalty(mttr):
    """For any episode where mttr > 2 * sla_breach_threshold,
    episode reward must include -0.5 penalty.

    Validates: Requirements 13.7
    """
    rf = _make_reward_fn()
    ws = _baseline_world_state()
    incident_state = _make_incident_state()
    trajectory = _make_trajectory(mttr=mttr)

    result = rf.compute_episode_reward(trajectory, ws, incident_state)

    assert result.penalties == -0.5, (
        f"Expected penalties=-0.5 for mttr={mttr} > 2*{_SLA_THRESHOLD}={2*_SLA_THRESHOLD}, "
        f"got penalties={result.penalties}"
    )


@given(
    mttr=mttr_at_or_below_threshold_st,
)
@settings(max_examples=100)
def test_no_late_penalty_when_resolved_on_time(mttr):
    """For any episode where mttr <= 2 * sla_breach_threshold,
    no late resolution penalty is applied.

    Validates: Requirements 13.7
    """
    rf = _make_reward_fn()
    ws = _baseline_world_state()
    incident_state = _make_incident_state()
    trajectory = _make_trajectory(mttr=mttr)

    result = rf.compute_episode_reward(trajectory, ws, incident_state)

    assert result.penalties == 0.0, (
        f"Expected penalties=0.0 for mttr={mttr} <= 2*{_SLA_THRESHOLD}={2*_SLA_THRESHOLD}, "
        f"got penalties={result.penalties}"
    )


# ---------------------------------------------------------------------------
# Property 16: RestartService on a healthy service incurs a blast_radius penalty
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 16: RestartService on a healthy service incurs a blast_radius penalty
@given(
    target_service=service_st,
)
@settings(max_examples=100)
def test_restart_healthy_service_incurs_penalty(target_service):
    """For any healthy service (availability=True), RestartService must result
    in a negative step reward.

    Validates: Requirements 7.6
    """
    rf = _make_reward_fn()

    # World state where target service is healthy (baseline)
    ws = _baseline_world_state()
    # Ensure the target service is healthy
    assert ws.services[target_service].availability is True, (
        f"Service '{target_service}' should be healthy at baseline"
    )

    # Pre-action incident state — same blast radius as post-action (no expansion)
    pre_incident_state = _make_incident_state(
        current_blast_radius=set(),
    )
    # Post-action world state has same blast radius (no expansion penalty)
    ws.incident_state = _make_incident_state(
        current_blast_radius=set(),
    )

    action = Action(
        agent="forge",
        category="remediation",
        name="RestartService",
        params={"service": target_service},
    )

    step_reward = rf.compute_step_reward(action, ws, pre_incident_state)

    assert step_reward < 0.0, (
        f"Expected negative step_reward for RestartService on healthy '{target_service}', "
        f"got {step_reward}"
    )
