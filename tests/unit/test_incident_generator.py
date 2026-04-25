"""Unit tests for Incident_Generator.

Covers schema validation, error handling, add_template(), and sample().
"""
from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from sentinel.exceptions import IncidentLibraryError
from sentinel.incident_generator import Incident_Generator
from sentinel.models import FailureType, IncidentTemplate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_template(**overrides) -> IncidentTemplate:
    """Return a minimal valid IncidentTemplate, with optional field overrides."""
    defaults = dict(
        id="T1",
        name="Test Incident",
        difficulty="easy",
        root_cause_service="cart-service",
        failure_type=FailureType.memory_leak,
        ground_truth_signals=["signal-a", "signal-b"],
        red_herring_signals=["red-a"],
        cascade_risk="low",
        missing_log_ratio=0.1,
        expected_steps_to_resolve=(5, 15),
    )
    defaults.update(overrides)
    return IncidentTemplate(**defaults)


def _write_yaml(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh)


def _minimal_library_yaml() -> dict:
    return {
        "incidents": [
            {
                "id": "E1",
                "name": "Test Incident",
                "difficulty": "easy",
                "root_cause_service": "cart-service",
                "failure_type": "memory_leak",
                "ground_truth_signals": ["signal-a"],
                "red_herring_signals": ["red-a"],
                "cascade_risk": "low",
                "missing_log_ratio": 0.1,
                "expected_steps_to_resolve": [5, 15],
            }
        ]
    }


# ---------------------------------------------------------------------------
# Schema validation — rejects templates with missing fields
# ---------------------------------------------------------------------------

def test_validate_template_rejects_missing_root_cause_service():
    """validate_template() must raise ValueError when root_cause_service is invalid."""
    gen = Incident_Generator()
    # Use an unrecognised service name to trigger the validation error
    template = _make_valid_template(root_cause_service="nonexistent-service")
    with pytest.raises(ValueError, match="root_cause_service"):
        gen.validate_template(template)


def test_validate_template_rejects_empty_id():
    gen = Incident_Generator()
    template = _make_valid_template(id="")
    with pytest.raises(ValueError, match="id"):
        gen.validate_template(template)


def test_validate_template_rejects_invalid_difficulty():
    gen = Incident_Generator()
    template = _make_valid_template(difficulty="extreme")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="difficulty"):
        gen.validate_template(template)


def test_validate_template_rejects_empty_ground_truth_signals():
    gen = Incident_Generator()
    template = _make_valid_template(ground_truth_signals=[])
    with pytest.raises(ValueError, match="ground_truth_signals"):
        gen.validate_template(template)


def test_validate_template_rejects_empty_red_herring_signals():
    gen = Incident_Generator()
    template = _make_valid_template(red_herring_signals=[])
    with pytest.raises(ValueError, match="red_herring_signals"):
        gen.validate_template(template)


def test_validate_template_rejects_missing_log_ratio_out_of_range():
    gen = Incident_Generator()
    template = _make_valid_template(missing_log_ratio=1.5)
    with pytest.raises(ValueError, match="missing_log_ratio"):
        gen.validate_template(template)


def test_validate_template_rejects_invalid_steps_to_resolve():
    gen = Incident_Generator()
    # min > max is invalid
    template = _make_valid_template(expected_steps_to_resolve=(20, 5))
    with pytest.raises(ValueError, match="expected_steps_to_resolve"):
        gen.validate_template(template)


# ---------------------------------------------------------------------------
# IncidentLibraryError on missing YAML file
# ---------------------------------------------------------------------------

def test_raises_incident_library_error_on_missing_file():
    """Incident_Generator must raise IncidentLibraryError when the YAML file is absent."""
    with pytest.raises(IncidentLibraryError, match="not found"):
        Incident_Generator(library_path="/nonexistent/path/incident_library.yaml")


# ---------------------------------------------------------------------------
# IncidentLibraryError on malformed YAML
# ---------------------------------------------------------------------------

def test_raises_incident_library_error_on_malformed_yaml():
    """Incident_Generator must raise IncidentLibraryError when the YAML is malformed."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
        fh.write(": invalid: yaml: [\n")
        tmp_path = fh.name

    try:
        with pytest.raises(IncidentLibraryError, match="[Mm]alformed"):
            Incident_Generator(library_path=tmp_path)
    finally:
        os.unlink(tmp_path)


def test_raises_incident_library_error_on_missing_incidents_key():
    """Incident_Generator must raise IncidentLibraryError when 'incidents' key is absent."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
        yaml.dump({"not_incidents": []}, fh)
        tmp_path = fh.name

    try:
        with pytest.raises(IncidentLibraryError):
            Incident_Generator(library_path=tmp_path)
    finally:
        os.unlink(tmp_path)


def test_raises_incident_library_error_on_template_missing_required_field():
    """Incident_Generator must raise IncidentLibraryError when a template is missing a required field."""
    bad_data = {
        "incidents": [
            {
                "id": "X1",
                "name": "Bad Template",
                # missing root_cause_service and other required fields
                "difficulty": "easy",
            }
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
        yaml.dump(bad_data, fh)
        tmp_path = fh.name

    try:
        with pytest.raises(IncidentLibraryError):
            Incident_Generator(library_path=tmp_path)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# add_template() adds a valid template
# ---------------------------------------------------------------------------

def test_add_template_increases_template_count():
    """add_template() must append a valid template to the library."""
    gen = Incident_Generator()
    initial_count = len(gen.templates)
    new_template = _make_valid_template(id="ORACLE-1")
    gen.add_template(new_template)
    assert len(gen.templates) == initial_count + 1


def test_add_template_stores_correct_template():
    """add_template() must store the exact template that was added."""
    gen = Incident_Generator()
    new_template = _make_valid_template(id="ORACLE-2", name="Oracle Generated")
    gen.add_template(new_template)
    assert gen.templates[-1].id == "ORACLE-2"
    assert gen.templates[-1].name == "Oracle Generated"


def test_add_template_rejects_invalid_template():
    """add_template() must raise ValueError for an invalid template."""
    gen = Incident_Generator()
    invalid_template = _make_valid_template(root_cause_service="not-a-real-service")
    with pytest.raises(ValueError):
        gen.add_template(invalid_template)


# ---------------------------------------------------------------------------
# sample() returns a template of the correct difficulty
# ---------------------------------------------------------------------------

def test_sample_returns_easy_template_when_only_easy():
    """sample() must return an easy template when distribution is 100% easy."""
    gen = Incident_Generator()
    dist = {"easy": 1.0, "medium": 0.0, "hard": 0.0}
    for _ in range(20):
        template = gen.sample(dist)
        assert template.difficulty == "easy"


def test_sample_returns_hard_template_when_only_hard():
    """sample() must return a hard template when distribution is 100% hard."""
    gen = Incident_Generator()
    dist = {"easy": 0.0, "medium": 0.0, "hard": 1.0}
    for _ in range(20):
        template = gen.sample(dist)
        assert template.difficulty == "hard"


def test_sample_raises_when_no_templates_for_difficulty():
    """sample() must raise ValueError when no templates exist for the sampled difficulty."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
        yaml.dump(_minimal_library_yaml(), fh)
        tmp_path = fh.name

    try:
        gen = Incident_Generator(library_path=tmp_path)
        # Library only has easy; force hard
        dist = {"easy": 0.0, "medium": 0.0, "hard": 1.0}
        with pytest.raises(ValueError, match="hard"):
            gen.sample(dist)
    finally:
        os.unlink(tmp_path)


def test_sample_returns_incident_template_instance():
    """sample() must always return an IncidentTemplate object."""
    gen = Incident_Generator()
    dist = {"easy": 0.3, "medium": 0.4, "hard": 0.3}
    result = gen.sample(dist)
    assert isinstance(result, IncidentTemplate)


def test_sample_returns_template_from_library():
    """sample() must return a template that exists in the loaded library."""
    gen = Incident_Generator()
    dist = {"easy": 0.3, "medium": 0.4, "hard": 0.3}
    template_ids = {t.id for t in gen.templates}
    for _ in range(30):
        result = gen.sample(dist)
        assert result.id in template_ids
