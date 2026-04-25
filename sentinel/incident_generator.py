"""Incident_Generator: loads and samples from incident_library.yaml."""
from __future__ import annotations

import random
from typing import Any

import yaml

from sentinel.exceptions import IncidentLibraryError
from sentinel.models import FailureType, IncidentTemplate

# All valid NexaStack service names
_VALID_SERVICES: frozenset[str] = frozenset({
    # Frontend
    "web-gateway", "mobile-api", "cdn-edge",
    # Application
    "cart-service", "order-service", "product-catalog", "search-service",
    "recommendation-engine", "user-auth", "notification-service",
    "pricing-engine", "inventory-service", "review-service",
    "wishlist-service", "session-manager",
    # Data
    "postgres-primary", "postgres-replica", "redis-cache", "elasticsearch",
    "kafka-broker", "object-storage", "analytics-db", "audit-log",
    # Infrastructure
    "service-mesh", "load-balancer", "api-gateway", "config-service",
    "secret-manager", "payment-vault", "fraud-detector",
})

_VALID_DIFFICULTIES: frozenset[str] = frozenset({"easy", "medium", "hard"})
_VALID_CASCADE_RISKS: frozenset[str] = frozenset({"low", "medium", "high"})


def _parse_template(raw: dict[str, Any]) -> IncidentTemplate:
    """Convert a raw YAML dict into an IncidentTemplate, raising ValueError on bad data."""
    required_fields = {
        "id", "name", "difficulty", "root_cause_service", "failure_type",
        "ground_truth_signals", "red_herring_signals", "cascade_risk",
        "missing_log_ratio", "expected_steps_to_resolve",
    }
    missing = required_fields - raw.keys()
    if missing:
        raise ValueError(f"Template missing required fields: {missing}")

    steps = raw["expected_steps_to_resolve"]
    if not (isinstance(steps, (list, tuple)) and len(steps) == 2):
        raise ValueError(
            f"expected_steps_to_resolve must be a list of 2 ints, got {steps!r}"
        )

    return IncidentTemplate(
        id=str(raw["id"]),
        name=str(raw["name"]),
        difficulty=raw["difficulty"],
        root_cause_service=raw["root_cause_service"],
        failure_type=FailureType(raw["failure_type"]),
        ground_truth_signals=list(raw["ground_truth_signals"]),
        red_herring_signals=list(raw["red_herring_signals"]),
        cascade_risk=raw["cascade_risk"],
        missing_log_ratio=float(raw["missing_log_ratio"]),
        expected_steps_to_resolve=(int(steps[0]), int(steps[1])),
    )


class Incident_Generator:
    """Loads incident templates from a YAML library and provides sampling."""

    def __init__(self, library_path: str = "incident_library.yaml") -> None:
        self._library_path = library_path
        self._templates: list[IncidentTemplate] = []
        self._load_library()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_library(self) -> None:
        """Parse incident_library.yaml; raise IncidentLibraryError on failure."""
        try:
            with open(self._library_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except FileNotFoundError:
            raise IncidentLibraryError(
                f"Incident library not found: {self._library_path!r}"
            )
        except yaml.YAMLError as exc:
            raise IncidentLibraryError(
                f"Malformed YAML in incident library: {exc}"
            ) from exc

        if not isinstance(raw, dict) or "incidents" not in raw:
            raise IncidentLibraryError(
                "Incident library must be a YAML mapping with an 'incidents' key."
            )

        incidents_raw = raw["incidents"]
        if not isinstance(incidents_raw, list) or len(incidents_raw) == 0:
            raise IncidentLibraryError(
                "'incidents' must be a non-empty list of template mappings."
            )

        templates: list[IncidentTemplate] = []
        for i, entry in enumerate(incidents_raw):
            try:
                templates.append(_parse_template(entry))
            except (ValueError, KeyError) as exc:
                raise IncidentLibraryError(
                    f"Invalid template at index {i}: {exc}"
                ) from exc

        self._templates = templates

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def templates(self) -> list[IncidentTemplate]:
        """Return all loaded templates."""
        return list(self._templates)

    def sample(self, difficulty_distribution: dict[str, float]) -> IncidentTemplate:
        """Weighted-sample a template by difficulty distribution.

        Args:
            difficulty_distribution: mapping of difficulty -> probability weight,
                e.g. {"easy": 0.3, "medium": 0.4, "hard": 0.3}.

        Returns:
            A randomly selected IncidentTemplate matching the sampled difficulty.

        Raises:
            ValueError: if no templates exist for the sampled difficulty.
        """
        difficulties = list(difficulty_distribution.keys())
        weights = [difficulty_distribution[d] for d in difficulties]

        sampled_difficulty: str = random.choices(difficulties, weights=weights, k=1)[0]

        candidates = [t for t in self._templates if t.difficulty == sampled_difficulty]
        if not candidates:
            raise ValueError(
                f"No templates available for difficulty {sampled_difficulty!r}. "
                f"Available difficulties: "
                f"{sorted({t.difficulty for t in self._templates})}"
            )

        return random.choice(candidates)

    def validate_template(self, template: IncidentTemplate) -> bool:
        """Validate a template against the IncidentTemplate schema.

        Returns:
            True if valid.

        Raises:
            ValueError: with a descriptive message if any field is invalid.
        """
        if not isinstance(template.id, str) or not template.id:
            raise ValueError("Template 'id' must be a non-empty string.")

        if not isinstance(template.name, str) or not template.name:
            raise ValueError("Template 'name' must be a non-empty string.")

        if template.difficulty not in _VALID_DIFFICULTIES:
            raise ValueError(
                f"Template 'difficulty' must be one of {sorted(_VALID_DIFFICULTIES)}, "
                f"got {template.difficulty!r}."
            )

        if template.root_cause_service not in _VALID_SERVICES:
            raise ValueError(
                f"Template 'root_cause_service' {template.root_cause_service!r} "
                f"is not a recognised NexaStack service."
            )

        if not isinstance(template.failure_type, FailureType):
            raise ValueError(
                f"Template 'failure_type' must be a FailureType enum member, "
                f"got {template.failure_type!r}."
            )

        if not isinstance(template.ground_truth_signals, list) or not template.ground_truth_signals:
            raise ValueError("Template 'ground_truth_signals' must be a non-empty list.")

        if not isinstance(template.red_herring_signals, list) or not template.red_herring_signals:
            raise ValueError("Template 'red_herring_signals' must be a non-empty list.")

        if template.cascade_risk not in _VALID_CASCADE_RISKS:
            raise ValueError(
                f"Template 'cascade_risk' must be one of {sorted(_VALID_CASCADE_RISKS)}, "
                f"got {template.cascade_risk!r}."
            )

        if not (0.0 <= template.missing_log_ratio <= 1.0):
            raise ValueError(
                f"Template 'missing_log_ratio' must be in [0.0, 1.0], "
                f"got {template.missing_log_ratio}."
            )

        steps = template.expected_steps_to_resolve
        if not (
            isinstance(steps, (tuple, list))
            and len(steps) == 2
            and isinstance(steps[0], int)
            and isinstance(steps[1], int)
            and steps[0] <= steps[1]
        ):
            raise ValueError(
                f"Template 'expected_steps_to_resolve' must be a (min, max) pair of ints "
                f"with min <= max, got {steps!r}."
            )

        return True

    def add_template(self, template: IncidentTemplate) -> None:
        """Validate and add an ORACLE-generated template to the library.

        Args:
            template: The IncidentTemplate to add.

        Raises:
            ValueError: if the template fails schema validation.
        """
        self.validate_template(template)
        self._templates.append(template)
