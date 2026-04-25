"""Property-based tests for Incident_Generator.

Tests Properties 13, 14, and 22 from the SENTINEL design document.
"""
from __future__ import annotations

import io
import math
from typing import Any

import yaml
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sentinel.incident_generator import Incident_Generator
from sentinel.models import FailureType, IncidentTemplate

# ---------------------------------------------------------------------------
# Required fields per design doc (Property 13)
# ---------------------------------------------------------------------------

REQUIRED_TEMPLATE_FIELDS = {
    "root_cause_service",
    "failure_type",
    "ground_truth_signals",
    "red_herring_signals",
    "cascade_risk",
    "missing_log_ratio",
    "expected_steps_to_resolve",
}

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_VALID_SERVICES = [
    "web-gateway", "mobile-api", "cdn-edge",
    "cart-service", "order-service", "product-catalog", "search-service",
    "recommendation-engine", "user-auth", "notification-service",
    "pricing-engine", "inventory-service", "review-service",
    "wishlist-service", "session-manager",
    "postgres-primary", "postgres-replica", "redis-cache", "elasticsearch",
    "kafka-broker", "object-storage", "analytics-db", "audit-log",
    "service-mesh", "load-balancer", "api-gateway", "config-service",
    "secret-manager", "payment-vault", "fraud-detector",
]

_VALID_DIFFICULTIES = ["easy", "medium", "hard"]
_VALID_CASCADE_RISKS = ["low", "medium", "high"]
_VALID_FAILURE_TYPES = list(FailureType)

signal_st = st.text(min_size=5, max_size=80, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs"), whitelist_characters="-_()"))

oracle_template_st = st.builds(
    IncidentTemplate,
    id=st.text(min_size=2, max_size=8, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    name=st.text(min_size=5, max_size=60, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Zs"))),
    difficulty=st.sampled_from(_VALID_DIFFICULTIES),
    root_cause_service=st.sampled_from(_VALID_SERVICES),
    failure_type=st.sampled_from(_VALID_FAILURE_TYPES),
    ground_truth_signals=st.lists(signal_st, min_size=1, max_size=5),
    red_herring_signals=st.lists(signal_st, min_size=1, max_size=4),
    cascade_risk=st.sampled_from(_VALID_CASCADE_RISKS),
    missing_log_ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    expected_steps_to_resolve=st.builds(
        lambda lo, hi: (lo, lo + hi),
        lo=st.integers(min_value=1, max_value=50),
        hi=st.integers(min_value=1, max_value=100),
    ),
)


def _difficulty_distribution_st():
    """Generate a valid difficulty distribution that sums to 1.0.

    Draws two values from [0.05, 0.90] that leave at least 0.05 for the third.
    """
    @st.composite
    def _build(draw):
        # Draw a in [0.05, 0.90]
        a = draw(st.floats(min_value=0.05, max_value=0.90, allow_nan=False, allow_infinity=False))
        # Remaining budget after a and the minimum 0.05 reserved for c
        remaining = 1.0 - a - 0.05
        if remaining < 0.05:
            # a is too large; clamp and set b=c=0.05, adjust a
            a = 0.90
            remaining = 0.05
        b = draw(st.floats(min_value=0.05, max_value=remaining, allow_nan=False, allow_infinity=False))
        c = round(1.0 - a - b, 10)
        # Clamp floating-point drift
        c = max(c, 0.0)
        return {"easy": round(a, 6), "medium": round(b, 6), "hard": round(c, 6)}
    return _build()


# ---------------------------------------------------------------------------
# Property 13: Every incident template contains all required fields
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 13: Every incident template contains all required fields
@settings(max_examples=100)
@given(oracle_template=oracle_template_st)
def test_every_template_contains_required_fields(oracle_template: IncidentTemplate):
    """For any incident template in the Incident_Library (including ORACLE-generated ones),
    the template must contain all required fields.

    Validates: Requirements 5.2, 5.5
    """
    gen = Incident_Generator()

    # Check all templates loaded from incident_library.yaml
    for template in gen.templates:
        template_dict = vars(template)
        for field in REQUIRED_TEMPLATE_FIELDS:
            assert field in template_dict, (
                f"Template '{template.id}' is missing required field '{field}'"
            )
            assert template_dict[field] is not None, (
                f"Template '{template.id}' has None for required field '{field}'"
            )

    # Add an ORACLE-generated template and verify it also has all required fields
    gen.add_template(oracle_template)
    added = gen.templates[-1]
    added_dict = vars(added)
    for field in REQUIRED_TEMPLATE_FIELDS:
        assert field in added_dict, (
            f"ORACLE template '{added.id}' is missing required field '{field}'"
        )
        assert added_dict[field] is not None, (
            f"ORACLE template '{added.id}' has None for required field '{field}'"
        )


# ---------------------------------------------------------------------------
# Property 14: Incident sampling matches the configured difficulty distribution
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 14: Incident sampling matches the configured difficulty distribution
@settings(max_examples=100)
@given(dist=_difficulty_distribution_st())
def test_sampling_matches_difficulty_distribution(dist: dict[str, float]):
    """For any difficulty distribution config, sampling 1000+ incidents must produce
    a distribution within ±5% of configured probabilities.

    Validates: Requirements 5.4
    """
    gen = Incident_Generator()
    n_samples = 1000
    counts: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}

    for _ in range(n_samples):
        template = gen.sample(dist)
        counts[template.difficulty] += 1

    for difficulty, expected_prob in dist.items():
        empirical_prob = counts[difficulty] / n_samples
        tolerance = 0.05
        assert abs(empirical_prob - expected_prob) <= tolerance, (
            f"Difficulty '{difficulty}': expected ~{expected_prob:.3f}, "
            f"got {empirical_prob:.3f} (diff={abs(empirical_prob - expected_prob):.3f} > {tolerance})"
        )


# ---------------------------------------------------------------------------
# Property 22: Incident template YAML round-trip
# ---------------------------------------------------------------------------

# Feature: sentinel, Property 22: Incident template YAML round-trip
def test_incident_template_yaml_round_trip():
    """For any incident template parsed from incident_library.yaml, serializing it
    back to YAML and re-parsing must produce an equivalent template.

    Validates: Requirements 18.5
    """
    gen = Incident_Generator()

    for original in gen.templates:
        # Serialize IncidentTemplate to a dict suitable for YAML
        template_dict: dict[str, Any] = {
            "id": original.id,
            "name": original.name,
            "difficulty": original.difficulty,
            "root_cause_service": original.root_cause_service,
            "failure_type": original.failure_type.value,
            "ground_truth_signals": list(original.ground_truth_signals),
            "red_herring_signals": list(original.red_herring_signals),
            "cascade_risk": original.cascade_risk,
            "missing_log_ratio": original.missing_log_ratio,
            "expected_steps_to_resolve": list(original.expected_steps_to_resolve),
        }

        # Dump to YAML string and re-parse
        yaml_str = yaml.dump({"incidents": [template_dict]}, default_flow_style=False)
        reparsed_raw = yaml.safe_load(yaml_str)
        assert "incidents" in reparsed_raw
        assert len(reparsed_raw["incidents"]) == 1

        # Re-parse using the same _parse_template logic via a fresh generator
        # We do this by writing a minimal YAML and loading it
        from sentinel.incident_generator import _parse_template
        reparsed_template = _parse_template(reparsed_raw["incidents"][0])

        # All fields must be identical
        assert reparsed_template.id == original.id
        assert reparsed_template.name == original.name
        assert reparsed_template.difficulty == original.difficulty
        assert reparsed_template.root_cause_service == original.root_cause_service
        assert reparsed_template.failure_type == original.failure_type
        assert reparsed_template.ground_truth_signals == original.ground_truth_signals
        assert reparsed_template.red_herring_signals == original.red_herring_signals
        assert reparsed_template.cascade_risk == original.cascade_risk
        assert math.isclose(reparsed_template.missing_log_ratio, original.missing_log_ratio, rel_tol=1e-9)
        assert reparsed_template.expected_steps_to_resolve == original.expected_steps_to_resolve
