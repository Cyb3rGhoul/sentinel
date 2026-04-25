"""Unit tests for ORACLE template retirement logic.

Tests:
- Templates are retired when oracle_template_count exceeds 50
- Below-median utility templates are removed first

Requirements: 12.7
"""
from __future__ import annotations

import pytest

from sentinel.agents.oracle import ORACLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_oracle_with_n_templates(n: int, base_utility: float = 0.5) -> ORACLE:
    """Create an ORACLE instance with n tracked templates at uniform utility."""
    oracle = ORACLE()
    for i in range(n):
        tid = f"ORACLE-{i:04d}"
        oracle.oracle_template_utility[tid] = base_utility
    oracle.oracle_template_count = n
    return oracle


def _make_oracle_with_mixed_utilities(utilities: list[float]) -> ORACLE:
    """Create an ORACLE instance with templates at the given utility scores."""
    oracle = ORACLE()
    for i, util in enumerate(utilities):
        tid = f"ORACLE-{i:04d}"
        oracle.oracle_template_utility[tid] = util
    oracle.oracle_template_count = len(utilities)
    return oracle


# ---------------------------------------------------------------------------
# Retirement trigger tests (Req 12.7)
# ---------------------------------------------------------------------------

class TestOracleRetirementTrigger:
    """Tests that retirement is triggered when count exceeds 50."""

    def test_no_retirement_when_count_is_50(self):
        """No templates should be retired when count == 50."""
        oracle = _make_oracle_with_n_templates(50)
        retired = oracle.retire_below_median_templates()
        assert retired == []
        assert oracle.oracle_template_count == 50

    def test_no_retirement_when_count_is_below_50(self):
        """No templates should be retired when count < 50."""
        oracle = _make_oracle_with_n_templates(30)
        retired = oracle.retire_below_median_templates()
        assert retired == []
        assert oracle.oracle_template_count == 30

    def test_retirement_triggered_when_count_exceeds_50(self):
        """Retirement should occur when count > 50."""
        # 51 templates: utilities range from 0.1 to 0.51 so median ~0.31
        # and the lower half are clearly below median
        utilities = [i * 0.01 for i in range(1, 52)]  # 0.01, 0.02, ..., 0.51
        oracle = _make_oracle_with_mixed_utilities(utilities)
        assert oracle.oracle_template_count == 51

        retired = oracle.retire_below_median_templates()

        assert len(retired) > 0
        assert oracle.oracle_template_count < 51

    def test_retirement_reduces_count(self):
        """After retirement, oracle_template_count must decrease."""
        utilities = [0.1, 0.2, 0.3, 0.8, 0.9] * 11  # 55 templates
        oracle = _make_oracle_with_mixed_utilities(utilities)
        initial_count = oracle.oracle_template_count

        retired = oracle.retire_below_median_templates()

        assert oracle.oracle_template_count == initial_count - len(retired)

    def test_retirement_removes_templates_from_utility_dict(self):
        """Retired template IDs must be removed from oracle_template_utility."""
        utilities = [0.1] * 30 + [0.9] * 25  # 55 templates; 30 below median
        oracle = _make_oracle_with_mixed_utilities(utilities)

        retired = oracle.retire_below_median_templates()

        for tid in retired:
            assert tid not in oracle.oracle_template_utility


# ---------------------------------------------------------------------------
# Below-median selection tests (Req 12.7)
# ---------------------------------------------------------------------------

class TestOracleRetirementBelowMedian:
    """Tests that below-median utility templates are removed first."""

    def test_only_below_median_templates_are_retired(self):
        """Only templates with utility < median should be retired."""
        # 51 templates: 25 with utility 0.2 (below median), 26 with utility 0.8
        # median = 0.8 (26th value), so 0.2 templates are strictly below median
        utilities = [0.2] * 25 + [0.8] * 26
        oracle = _make_oracle_with_mixed_utilities(utilities)

        retired = oracle.retire_below_median_templates()

        # All retired templates should have had utility 0.2 (indices 0-24)
        for tid in retired:
            idx = int(tid.split("-")[1])
            assert idx < 25, f"Template {tid} at index {idx} was above-median but retired"

    def test_above_median_templates_are_preserved(self):
        """Templates with utility >= median must not be retired."""
        # 51 templates: 25 with utility 0.1, 26 with utility 0.9
        # median = 0.9 (26th value from top), so 0.1 templates are below median
        utilities = [0.1] * 25 + [0.9] * 26
        oracle = _make_oracle_with_mixed_utilities(utilities)

        oracle.retire_below_median_templates()

        # All remaining templates should have utility 0.9
        for tid, util in oracle.oracle_template_utility.items():
            assert util >= 0.9, f"Template {tid} with utility {util} should have been retired"

    def test_retirement_with_all_equal_utilities(self):
        """When all utilities are equal, no templates are below median."""
        oracle = _make_oracle_with_n_templates(55, base_utility=0.5)

        retired = oracle.retire_below_median_templates()

        # All utilities equal median — nothing should be retired
        assert retired == []

    def test_retirement_with_single_low_outlier(self):
        """A single low-utility template among 50+ should be retired."""
        # 51 templates: 1 very low, 50 high
        utilities = [0.01] + [0.9] * 50
        oracle = _make_oracle_with_mixed_utilities(utilities)

        retired = oracle.retire_below_median_templates()

        assert len(retired) >= 1
        # The low-utility template (index 0) should be among the retired
        assert "ORACLE-0000" in retired

    def test_retirement_preserves_high_utility_templates(self):
        """High-utility templates must survive retirement."""
        utilities = [0.1, 0.2, 0.3, 0.4, 0.5] * 11  # 55 templates
        oracle = _make_oracle_with_mixed_utilities(utilities)

        oracle.retire_below_median_templates()

        # All remaining templates should have utility >= median
        import statistics
        remaining_utilities = list(oracle.oracle_template_utility.values())
        if remaining_utilities:
            med = statistics.median(
                [0.1, 0.2, 0.3, 0.4, 0.5] * 11  # original utilities
            )
            for util in remaining_utilities:
                assert util >= med or util == 0.5  # at-median may be kept

    def test_set_template_utility_updates_score(self):
        """set_template_utility() should update the utility score for a template."""
        oracle = ORACLE()
        oracle.oracle_template_utility["ORACLE-TEST"] = 0.3
        oracle.oracle_template_count = 1

        oracle.set_template_utility("ORACLE-TEST", 0.9)

        assert oracle.oracle_template_utility["ORACLE-TEST"] == 0.9

    def test_retirement_count_boundary_exactly_51(self):
        """Retirement should trigger at exactly 51 templates."""
        # 25 low-utility (0.1) + 26 high-utility (0.9) = 51 total
        # median = 0.9, so the 25 low-utility ones are below median
        utilities = [0.1] * 25 + [0.9] * 26
        oracle = _make_oracle_with_mixed_utilities(utilities)
        assert oracle.oracle_template_count == 51

        retired = oracle.retire_below_median_templates()

        assert len(retired) > 0

    def test_retirement_does_not_retire_at_exactly_50(self):
        """Retirement must NOT trigger at exactly 50 templates."""
        utilities = [0.1] * 25 + [0.9] * 25  # 50 total
        oracle = _make_oracle_with_mixed_utilities(utilities)
        assert oracle.oracle_template_count == 50

        retired = oracle.retire_below_median_templates()

        assert retired == []
        assert oracle.oracle_template_count == 50
