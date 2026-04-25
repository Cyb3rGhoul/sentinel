"""Configuration loader for SENTINEL.

Loads and validates env_spec.yaml using Pydantic v2 models.
Falls back to defaults when the file is absent.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-config models
# ---------------------------------------------------------------------------

class NexaStackConfig(BaseModel):
    topology: str = "default"


class DifficultyDistribution(BaseModel):
    easy: float = 0.3
    medium: float = 0.4
    hard: float = 0.3

    @model_validator(mode="after")
    def _sum_to_one(self) -> "DifficultyDistribution":
        total = self.easy + self.medium + self.hard
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"difficulty_distribution must sum to 1.0, got {total}"
            )
        return self


class IncidentConfig(BaseModel):
    difficulty_distribution: DifficultyDistribution = Field(
        default_factory=DifficultyDistribution
    )


class RewardWeightsConfig(BaseModel):
    r1_root_cause: float = 0.35
    r2_mttr: float = 0.30
    r3_recovery_quality: float = 0.25
    r4_blast_radius: float = 0.10

    @model_validator(mode="after")
    def _sum_to_one(self) -> "RewardWeightsConfig":
        total = (
            self.r1_root_cause
            + self.r2_mttr
            + self.r3_recovery_quality
            + self.r4_blast_radius
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"reward weights must sum to 1.0, got {total}")
        return self


class RewardConfig(BaseModel):
    weights: RewardWeightsConfig = Field(default_factory=RewardWeightsConfig)


class ObservabilityConfig(BaseModel):
    black_box_services: list[str] = Field(
        default_factory=lambda: ["payment-vault", "fraud-detector"]
    )
    alert_threshold_multiplier: float = 1.5


class TrainingConfig(BaseModel):
    max_steps_per_episode: int = 200
    sla_breach_threshold_steps: int = 50
    placeholder_action: dict[str, Any] = Field(
        default_factory=lambda: {
            "agent": "holmes",
            "category": "investigative",
            "name": "QueryLogs",
            "params": {"service": "cart-service", "time_range": [0, 60]},
        }
    )


class DemoConfig(BaseModel):
    seed: int = 42


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class SentinelConfig(BaseModel):
    nexastack: NexaStackConfig = Field(default_factory=NexaStackConfig)
    incident: IncidentConfig = Field(default_factory=IncidentConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    demo: DemoConfig = Field(default_factory=DemoConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str = "env_spec.yaml") -> SentinelConfig:
    """Load and validate config from *path*.

    Falls back to default values with a warning when the file is absent.
    """
    if not os.path.exists(path):
        logger.warning(
            "env_spec.yaml not found at '%s'; using default configuration.", path
        )
        return SentinelConfig()

    with open(path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    return SentinelConfig.model_validate(raw)
