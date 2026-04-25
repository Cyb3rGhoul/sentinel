"""Unit tests for Reward_Function.

Tests exact R1/R2/R3/R4 values and penalty application.
"""
from __future__ import annotations

import pytest

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
# Fixtures / helpers
# ---------------------------------------------------------------------------

SLA_THRESHOLD = 50

WEIGHTS = RewardWeights(
    r1_root_cause=0.35,
    r2_mttr=0.30,
    r3_recovery_quality=0.25,
    r4_blast_radius=0.10,
)


@pytest.fixture
def rf() -> Reward_Function:
    return Reward_Function(weights=WEIGHTS, sla_breach_threshold=SLA_THRESHOLD)


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
    ws = NexaStackWorldState()
    ws.restore_baseline()
    return ws


# ---------------------------------------------------------------------------
# R1 — Root cause accuracy
# ---------------------------------------------------------------------------

class TestR1RootCauseAccuracy:
    def test_r1_correct_service_and_type_returns_1_0(self, rf):
        """Correct service + correct failure type → R1 = 1.0"""
        incident = _make_incident_state(
            root_cause_service="redis-cache",
            failure_type=FailureType.memory_leak,
        )
        r1 = rf._r1_root_cause_accuracy(incident, "redis-cache", "memory_leak")
        assert r1 == 1.0

    def test_r1_correct_service_wrong_type_returns_0_5(self, rf):
        """Correct service + wrong failure type → R1 = 0.5"""
        incident = _make_incident_state(
            root_cause_service="redis-cache",
            failure_type=FailureType.memory_leak,
        )
        r1 = rf._r1_root_cause_accuracy(incident, "redis-cache", "cpu_spike")
        assert r1 == 0.5

    def test_r1_wrong_service_returns_0_0(self, rf):
        """Wrong service → R1 = 0.0 regardless of failure type"""
        incident = _make_incident_state(
            root_cause_service="redis-cache",
            failure_type=FailureType.memory_leak,
        )
        r1 = rf._r1_root_cause_accuracy(incident, "postgres-primary", "memory_leak")
        assert r1 == 0.0

    def test_r1_wrong_service_wrong_type_returns_0_0(self, rf):
        """Wrong service + wrong type → R1 = 0.0"""
        incident = _make_incident_state(
            root_cause_service="redis-cache",
            failure_type=FailureType.memory_leak,
        )
        r1 = rf._r1_root_cause_accuracy(incident, "kafka-broker", "cpu_spike")
        assert r1 == 0.0

    def test_r1_empty_identified_returns_0_0(self, rf):
        """Empty identified root cause → R1 = 0.0"""
        incident = _make_incident_state(root_cause_service="redis-cache")
        r1 = rf._r1_root_cause_accuracy(incident, "", "")
        assert r1 == 0.0


# ---------------------------------------------------------------------------
# R2 — MTTR score
# ---------------------------------------------------------------------------

class TestR2Mttr:
    def test_r2_mttr_zero_returns_1_1(self, rf):
        """mttr=0 → R2 = 1.0 / (1.0 + 0/50) + 0.1 = 1.1"""
        r2 = rf._r2_mttr(0)
        assert abs(r2 - 1.1) < 1e-9, f"Expected 1.1, got {r2}"

    def test_r2_mttr_at_sla_threshold_returns_0_5(self, rf):
        """mttr=sla_threshold → R2 = 1.0 / (1.0 + 1.0) = 0.5 (no bonus)"""
        r2 = rf._r2_mttr(SLA_THRESHOLD)
        # At exactly sla_threshold, mttr is NOT < threshold, so no bonus
        expected = 1.0 / (1.0 + SLA_THRESHOLD / SLA_THRESHOLD)  # = 0.5
        assert abs(r2 - expected) < 1e-9, f"Expected {expected}, got {r2}"

    def test_r2_mttr_double_threshold_returns_one_third(self, rf):
        """mttr=2*sla_threshold → R2 = 1.0 / (1.0 + 2.0) = 1/3"""
        mttr = 2 * SLA_THRESHOLD
        r2 = rf._r2_mttr(mttr)
        expected = 1.0 / (1.0 + mttr / SLA_THRESHOLD)  # = 1/3
        assert abs(r2 - expected) < 1e-9, f"Expected {expected:.6f}, got {r2:.6f}"

    def test_r2_pre_sla_bonus_applied_when_mttr_below_threshold(self, rf):
        """mttr < sla_threshold → bonus of 0.1 is added"""
        mttr = SLA_THRESHOLD - 1
        r2 = rf._r2_mttr(mttr)
        base = 1.0 / (1.0 + mttr / SLA_THRESHOLD)
        expected = base + 0.1
        assert abs(r2 - expected) < 1e-9, f"Expected {expected:.6f}, got {r2:.6f}"

    def test_r2_no_bonus_at_threshold(self, rf):
        """mttr == sla_threshold → no bonus"""
        r2_at = rf._r2_mttr(SLA_THRESHOLD)
        r2_below = rf._r2_mttr(SLA_THRESHOLD - 1)
        # Below threshold gets bonus, at threshold does not
        assert r2_below > r2_at

    def test_r2_clamped_to_1_1(self, rf):
        """R2 is clamped to max 1.1"""
        r2 = rf._r2_mttr(0)
        assert r2 <= 1.1

    def test_r2_clamped_to_0_0(self, rf):
        """R2 is clamped to min 0.0"""
        r2 = rf._r2_mttr(10_000_000)
        assert r2 >= 0.0


# ---------------------------------------------------------------------------
# R3 — Recovery quality
# ---------------------------------------------------------------------------

class TestR3RecoveryQuality:
    def test_r3_all_services_at_baseline_returns_1_0(self, rf):
        """All services at baseline → R3 = 1.0"""
        ws = _baseline_world_state()
        r3 = rf._r3_recovery_quality(ws)
        assert abs(r3 - 1.0) < 1e-9, f"Expected 1.0, got {r3}"

    def test_r3_all_services_degraded_returns_0_0(self, rf):
        """All services severely degraded → R3 = 0.0"""
        ws = _baseline_world_state()
        # Degrade all services far beyond baseline
        for svc in ALL_SERVICES:
            ws.services[svc] = ServiceMetrics(
                cpu=1.0,
                memory=1.0,
                latency_ms=1000.0,
                error_rate=1.0,
                saturation=1.0,
                availability=False,
            )
        r3 = rf._r3_recovery_quality(ws)
        assert abs(r3 - 0.0) < 1e-9, f"Expected 0.0, got {r3}"

    def test_r3_half_services_recovered_returns_0_5(self, rf):
        """Half services at baseline, half degraded → R3 = 0.5"""
        ws = _baseline_world_state()
        half = len(ALL_SERVICES) // 2
        for svc in ALL_SERVICES[:half]:
            ws.services[svc] = ServiceMetrics(
                cpu=1.0,
                memory=1.0,
                latency_ms=1000.0,
                error_rate=1.0,
                saturation=1.0,
                availability=False,
            )
        r3 = rf._r3_recovery_quality(ws)
        assert abs(r3 - 0.5) < 1e-9, f"Expected 0.5, got {r3}"

    def test_r3_empty_services_returns_1_0(self, rf):
        """No services → R3 = 1.0 (vacuously true)"""
        ws = _baseline_world_state()
        ws.services = {}
        r3 = rf._r3_recovery_quality(ws)
        assert abs(r3 - 1.0) < 1e-9, f"Expected 1.0, got {r3}"


# ---------------------------------------------------------------------------
# R4 — Blast radius minimization
# ---------------------------------------------------------------------------

class TestR4BlastRadius:
    def test_r4_current_equals_peak_returns_0_0(self, rf):
        """current_br == peak_br → R4 = 0.0"""
        services = {"svc-a", "svc-b", "svc-c"}
        incident = _make_incident_state(
            peak_blast_radius=services,
            current_blast_radius=services,
        )
        r4 = rf._r4_blast_radius(incident)
        assert abs(r4 - 0.0) < 1e-9, f"Expected 0.0, got {r4}"

    def test_r4_current_zero_returns_1_0(self, rf):
        """current_br = 0, peak_br > 0 → R4 = 1.0"""
        incident = _make_incident_state(
            peak_blast_radius={"svc-a", "svc-b"},
            current_blast_radius=set(),
        )
        r4 = rf._r4_blast_radius(incident)
        assert abs(r4 - 1.0) < 1e-9, f"Expected 1.0, got {r4}"

    def test_r4_peak_zero_returns_1_0(self, rf):
        """peak_br = 0 → R4 = 1.0 (no blast occurred)"""
        incident = _make_incident_state(
            peak_blast_radius=set(),
            current_blast_radius=set(),
        )
        r4 = rf._r4_blast_radius(incident)
        assert abs(r4 - 1.0) < 1e-9, f"Expected 1.0, got {r4}"

    def test_r4_partial_reduction(self, rf):
        """current_br = half of peak_br → R4 = 0.5"""
        peak = {"a", "b", "c", "d"}
        current = {"a", "b"}
        incident = _make_incident_state(
            peak_blast_radius=peak,
            current_blast_radius=current,
        )
        r4 = rf._r4_blast_radius(incident)
        expected = 1.0 - (2 / 4)  # = 0.5
        assert abs(r4 - expected) < 1e-9, f"Expected {expected}, got {r4}"


# ---------------------------------------------------------------------------
# Penalty application
# ---------------------------------------------------------------------------

class TestPenalties:
    def test_blast_radius_expansion_penalty(self, rf):
        """Blast radius expansion → step reward includes -1.0 base penalty."""
        pre_br = {"svc-a"}
        post_br = {"svc-a", "svc-b"}

        # Use a non-root-cause, non-blast-radius service as the action target
        # so no shaping bonuses interfere
        pre_incident = _make_incident_state(
            root_cause_service="redis-cache",
            current_blast_radius=pre_br,
        )
        ws = _baseline_world_state()
        ws.incident_state = _make_incident_state(current_blast_radius=post_br)

        action = Action(
            agent="forge",
            category="remediation",
            name="ScaleService",
            params={"service": "api-gateway", "replicas": 3},  # not the root cause
        )

        step_reward = rf.compute_step_reward(action, ws, pre_incident)
        # -1.0 (blast expansion) + 0.05 (remediating affected blast service) = -0.95
        # svc-a is in pre_br and api-gateway is not in pre_br, so no 0.05 bonus
        # api-gateway is not redis-cache, not in blast → 0.0 remediation shaping
        assert step_reward == -1.0, f"Expected -1.0, got {step_reward}"

    def test_no_blast_radius_penalty_when_br_unchanged(self, rf):
        """No blast radius change → no -1.0 penalty"""
        br = {"svc-a", "svc-b"}

        pre_incident = _make_incident_state(
            root_cause_service="redis-cache",
            current_blast_radius=br,
        )
        ws = _baseline_world_state()
        ws.incident_state = _make_incident_state(current_blast_radius=br)

        action = Action(
            agent="forge",
            category="remediation",
            name="ScaleService",
            params={"service": "api-gateway", "replicas": 3},  # not root cause, not in blast
        )

        step_reward = rf.compute_step_reward(action, ws, pre_incident)
        assert step_reward == 0.0, f"Expected 0.0, got {step_reward}"

    def test_restart_healthy_service_penalty(self, rf):
        """RestartService on healthy non-root service → step reward = -1.0"""
        ws = _baseline_world_state()
        # Ensure target is healthy and is NOT the root cause service
        assert ws.services["api-gateway"].availability is True

        pre_incident = _make_incident_state(
            root_cause_service="redis-cache",  # root is redis-cache, NOT api-gateway
            current_blast_radius=set(),
        )
        ws.incident_state = _make_incident_state(
            root_cause_service="redis-cache",
            current_blast_radius=set(),
        )

        action = Action(
            agent="forge",
            category="remediation",
            name="RestartService",
            params={"service": "api-gateway"},  # healthy, non-root service
        )

        step_reward = rf.compute_step_reward(action, ws, pre_incident)
        assert step_reward == -1.0, f"Expected -1.0, got {step_reward}"

    def test_restart_unhealthy_service_no_penalty(self, rf):
        """RestartService on unhealthy root cause service → +0.25 remediation shaping"""
        ws = _baseline_world_state()
        # Degrade the target service
        ws.services["redis-cache"] = ServiceMetrics(
            cpu=1.0, memory=1.0, latency_ms=1000.0,
            error_rate=1.0, saturation=1.0, availability=False,
        )

        pre_incident = _make_incident_state(
            root_cause_service="redis-cache",
            current_blast_radius={"redis-cache"},
        )
        ws.incident_state = _make_incident_state(
            root_cause_service="redis-cache",
            current_blast_radius={"redis-cache"},
        )

        action = Action(
            agent="forge",
            category="remediation",
            name="RestartService",
            params={"service": "redis-cache"},
        )

        step_reward = rf.compute_step_reward(action, ws, pre_incident)
        # No healthy-restart penalty (service is degraded); +0.25 root-cause remediation
        assert step_reward == 0.25, f"Expected 0.25, got {step_reward}"

    def test_late_resolution_penalty_in_episode_reward(self, rf):
        """mttr > 2 * sla_threshold → episode penalties = -0.5"""
        ws = _baseline_world_state()
        incident = _make_incident_state()
        trajectory = _make_trajectory(mttr=2 * SLA_THRESHOLD + 1)

        result = rf.compute_episode_reward(trajectory, ws, incident)
        assert result.penalties == -0.5, f"Expected -0.5, got {result.penalties}"

    def test_no_late_penalty_when_resolved_on_time(self, rf):
        """mttr <= 2 * sla_threshold → episode penalties = 0.0"""
        ws = _baseline_world_state()
        incident = _make_incident_state()
        trajectory = _make_trajectory(mttr=2 * SLA_THRESHOLD)

        result = rf.compute_episode_reward(trajectory, ws, incident)
        assert result.penalties == 0.0, f"Expected 0.0, got {result.penalties}"

    def test_both_penalties_accumulate(self, rf):
        """Blast radius expansion + healthy restart on non-root service → step reward = -2.0"""
        pre_br = {"svc-a"}
        post_br = {"svc-a", "svc-b"}

        pre_incident = _make_incident_state(
            root_cause_service="redis-cache",
            current_blast_radius=pre_br,
        )
        ws = _baseline_world_state()
        ws.incident_state = _make_incident_state(current_blast_radius=post_br)

        # RestartService on a healthy non-root service (api-gateway) AND blast radius expanded
        action = Action(
            agent="forge",
            category="remediation",
            name="RestartService",
            params={"service": "api-gateway"},  # not the root cause, and healthy
        )

        step_reward = rf.compute_step_reward(action, ws, pre_incident)
        assert step_reward == -2.0, f"Expected -2.0, got {step_reward}"


# ---------------------------------------------------------------------------
# compute_episode_reward — integration
# ---------------------------------------------------------------------------

class TestComputeEpisodeReward:
    def test_total_equals_weighted_sum(self, rf):
        """total == 0.35*R1 + 0.30*R2 + 0.25*R3 + 0.10*R4 + penalties"""
        ws = _baseline_world_state()
        incident = _make_incident_state(
            root_cause_service="redis-cache",
            failure_type=FailureType.memory_leak,
            peak_blast_radius={"a", "b", "c", "d"},
            current_blast_radius={"a", "b"},
        )
        trajectory = _make_trajectory(
            mttr=10,
            identified_root_cause="redis-cache",
            identified_failure_type="memory_leak",
        )

        result = rf.compute_episode_reward(trajectory, ws, incident)

        expected_total = (
            0.35 * result.r1
            + 0.30 * result.r2
            + 0.25 * result.r3
            + 0.10 * result.r4
            + result.penalties
        )
        assert abs(result.total - expected_total) < 1e-9

    def test_perfect_episode_reward(self, rf):
        """Perfect identification + fast MTTR + all recovered + no blast → high reward"""
        ws = _baseline_world_state()
        incident = _make_incident_state(
            root_cause_service="redis-cache",
            failure_type=FailureType.memory_leak,
            peak_blast_radius=set(),
            current_blast_radius=set(),
        )
        trajectory = _make_trajectory(
            mttr=0,
            identified_root_cause="redis-cache",
            identified_failure_type="memory_leak",
        )

        result = rf.compute_episode_reward(trajectory, ws, incident)

        # R1=1.0, R2=1.1, R3=1.0, R4=1.0, penalties=0.0
        assert result.r1 == 1.0
        assert abs(result.r2 - 1.1) < 1e-9
        assert abs(result.r3 - 1.0) < 1e-9
        assert abs(result.r4 - 1.0) < 1e-9
        assert result.penalties == 0.0

        expected_total = 0.35 * 1.0 + 0.30 * 1.1 + 0.25 * 1.0 + 0.10 * 1.0
        assert abs(result.total - expected_total) < 1e-9
