"""Property-based tests for SENTINEL serialization round-trips.

# Feature: sentinel, Property 21: Trajectory serialization round-trip
"""
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sentinel.models import Action, RewardBreakdown, Trajectory, TrajectoryStep, FailureType


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

action_st = st.builds(
    Action,
    agent=st.sampled_from(["argus", "holmes", "forge", "hermes", "oracle"]),
    category=st.sampled_from(["investigative", "remediation", "deployment", "meta"]),
    name=st.text(min_size=1, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz-"),
    params=st.fixed_dictionaries({
        "service": st.sampled_from(["cart-service", "redis-cache", "web-gateway"]),
    }),
)

step_st = st.builds(
    TrajectoryStep,
    observation=st.fixed_dictionaries({"key": st.text(min_size=0, max_size=20)}),
    action=action_st,
    reward=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    terminated=st.booleans(),
    truncated=st.booleans(),
    info=st.fixed_dictionaries({"step": st.integers(min_value=0, max_value=100)}),
)

reward_breakdown_st = st.builds(
    RewardBreakdown,
    r1=st.sampled_from([0.0, 0.5, 1.0]),
    r2=st.floats(min_value=0.0, max_value=1.1, allow_nan=False, allow_infinity=False),
    r3=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    r4=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    penalties=st.sampled_from([0.0, -0.5]),
    total=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)

trajectory_st = st.builds(
    Trajectory,
    episode_id=st.text(min_size=1, max_size=36, alphabet="abcdef0123456789-"),
    incident_template_id=st.sampled_from(["E1", "E2", "M1", "M2", "H1", "H2"]),
    steps=st.lists(step_st, min_size=1, max_size=10),
    final_reward=reward_breakdown_st,
    mttr=st.integers(min_value=0, max_value=200),
)


# ---------------------------------------------------------------------------
# Property 21: Trajectory serialization round-trip
# Validates: Requirements 18.3, 18.4
# ---------------------------------------------------------------------------

@given(original=trajectory_st)
@settings(max_examples=100)
def test_trajectory_serialization_round_trip(original: Trajectory) -> None:
    """For any valid Trajectory, to_json() followed by from_json() must produce
    a Trajectory with identical observation dicts, action parameters, and reward
    values for every step.

    # Feature: sentinel, Property 21: Trajectory serialization round-trip
    **Validates: Requirements 18.3, 18.4**
    """
    serialized = original.to_json()
    restored = Trajectory.from_json(serialized)

    # Top-level identity
    assert restored.episode_id == original.episode_id
    assert restored.incident_template_id == original.incident_template_id
    assert restored.mttr == original.mttr

    # Step-level identity
    assert len(restored.steps) == len(original.steps)
    for orig_step, rest_step in zip(original.steps, restored.steps):
        # Observation dicts must be equal
        assert rest_step.observation == orig_step.observation, (
            f"Observation mismatch: {rest_step.observation!r} != {orig_step.observation!r}"
        )
        # Action parameters must be equal
        assert rest_step.action.params == orig_step.action.params, (
            f"Action params mismatch: {rest_step.action.params!r} != {orig_step.action.params!r}"
        )
        assert rest_step.action.agent == orig_step.action.agent
        assert rest_step.action.category == orig_step.action.category
        assert rest_step.action.name == orig_step.action.name
        # Reward values must be close (float round-trip)
        assert math.isclose(rest_step.reward, orig_step.reward, rel_tol=1e-9, abs_tol=1e-12), (
            f"Reward mismatch: {rest_step.reward} != {orig_step.reward}"
        )
        assert rest_step.terminated == orig_step.terminated
        assert rest_step.truncated == orig_step.truncated
        assert rest_step.info == orig_step.info

    # Final reward identity
    assert restored.final_reward.r1 == original.final_reward.r1
    assert math.isclose(restored.final_reward.r2, original.final_reward.r2, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(restored.final_reward.r3, original.final_reward.r3, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(restored.final_reward.r4, original.final_reward.r4, rel_tol=1e-9, abs_tol=1e-12)
    assert restored.final_reward.penalties == original.final_reward.penalties
    assert math.isclose(restored.final_reward.total, original.final_reward.total, rel_tol=1e-9, abs_tol=1e-12)
