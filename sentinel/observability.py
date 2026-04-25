"""Observability Layer for SENTINEL.

Sits between NexaStackWorldState and agent observations, enforcing:
- Black-box service masking (metrics hidden, CDG rows zeroed)
- Log suppression (random fraction of log entries dropped)
- Red herring alert injection (structurally identical to real alerts)
"""
from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING

from sentinel.config import ObservabilityConfig
from sentinel.models import (
    Alert,
    HypothesisTree,
    IncidentContext,
    IncidentState,
    LogEntry,
    Trace,
)
from sentinel.world_state import ALL_SERVICES, NexaStackWorldState

# Index lookup for the 30×30 matrix
_SERVICE_INDEX: dict[str, int] = {svc: i for i, svc in enumerate(ALL_SERVICES)}

# Plausible red-herring metric names and messages
_RH_METRICS = ["cpu", "memory", "latency_ms", "error_rate", "saturation"]
_RH_LOG_LEVELS = ["WARN", "ERROR"]
_RH_MESSAGES = [
    "Connection timeout to upstream",
    "High GC pause detected",
    "Thread pool near capacity",
    "Slow query detected",
    "Retry limit approaching",
]


class Observability_Layer:
    """Filters and transforms world state into a partial observation."""

    def __init__(self, config: ObservabilityConfig) -> None:
        self.black_box_services: set[str] = set(config.black_box_services)
        self.alert_threshold_multiplier: float = config.alert_threshold_multiplier
        # Sampled once per episode via sample_episode_params()
        self.log_suppression_ratio: float = 0.0
        self.red_herring_count: int = 1

    # ------------------------------------------------------------------
    # Episode parameter sampling
    # ------------------------------------------------------------------

    def sample_episode_params(self) -> None:
        """Sample per-episode observability parameters.

        Called at reset() before the first observation is built.
        - log_suppression_ratio ~ Uniform[0.0, 0.8]
        - red_herring_count ~ Uniform{1, 2, 3}
        """
        self.log_suppression_ratio = random.uniform(0.0, 0.8)
        self.red_herring_count = random.choice([1, 2, 3])

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def build_observation(
        self,
        world_state: NexaStackWorldState,
        incident_state: IncidentState | None,
        hypothesis_tree: HypothesisTree | None,
    ) -> dict:
        """Build a full Observation dict from the current world state.

        Returns:
            dict with keys: metrics_snapshot, causal_graph_snapshot,
            active_alerts, recent_logs, active_traces, incident_context,
            sla_state.
        """
        return {
            "metrics_snapshot": self._build_metrics_snapshot(world_state),
            "causal_graph_snapshot": self._build_causal_graph_snapshot(world_state),
            "active_alerts": self._build_active_alerts(world_state, incident_state),
            "recent_logs": self._build_recent_logs(world_state),
            "active_traces": self._build_active_traces(world_state),
            "incident_context": self._build_incident_context(incident_state),
            "sla_state": self._build_sla_state(world_state),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_metrics_snapshot(
        self, world_state: NexaStackWorldState
    ) -> dict:
        """Return metrics for all services; None for black-box services."""
        snapshot: dict = {}
        for svc, metrics in world_state.services.items():
            if svc in self.black_box_services:
                snapshot[svc] = None
            else:
                snapshot[svc] = {
                    "cpu": metrics.cpu,
                    "memory": metrics.memory,
                    "latency_ms": metrics.latency_ms,
                    "error_rate": metrics.error_rate,
                    "saturation": metrics.saturation,
                    "availability": metrics.availability,
                }
        return snapshot

    def _build_causal_graph_snapshot(
        self, world_state: NexaStackWorldState
    ) -> list[list[float]]:
        """Build a 30×30 adjacency matrix; zero out rows for black-box services."""
        n = len(ALL_SERVICES)
        matrix: list[list[float]] = [[0.0] * n for _ in range(n)]

        for src, dst, data in world_state.cdg.edges(data=True):
            if src not in _SERVICE_INDEX or dst not in _SERVICE_INDEX:
                continue
            i = _SERVICE_INDEX[src]
            j = _SERVICE_INDEX[dst]
            matrix[i][j] = float(data.get("weight", 0.0))

        # Zero out rows for black-box services
        for svc in self.black_box_services:
            if svc in _SERVICE_INDEX:
                row = _SERVICE_INDEX[svc]
                matrix[row] = [0.0] * n

        return matrix

    def _build_active_alerts(
        self,
        world_state: NexaStackWorldState,
        incident_state: IncidentState | None,
    ) -> list[Alert]:
        """Generate real alerts for degraded services, then inject red herrings."""
        now = time.time()
        alerts: list[Alert] = []

        # Real alerts: services with availability=False
        for svc, metrics in world_state.services.items():
            if not metrics.availability:
                # Pick the most-breached metric as the alert signal
                metric, value, threshold = self._worst_metric(metrics)
                alerts.append(
                    Alert(
                        service=svc,
                        metric=metric,
                        value=value,
                        threshold=threshold,
                        timestamp=now,
                        confidence=0.9,
                    )
                )

        # Red herring alerts — structurally identical, no labeling
        available_services = [
            svc for svc in ALL_SERVICES
            if world_state.services[svc].availability
        ]
        rh_services = random.sample(
            available_services,
            min(self.red_herring_count, len(available_services)),
        )
        for svc in rh_services:
            metric = random.choice(_RH_METRICS)
            # Slightly elevated but below threshold to look plausible
            value = random.uniform(0.5, 0.75)
            threshold = 0.85
            alerts.append(
                Alert(
                    service=svc,
                    metric=metric,
                    value=value,
                    threshold=threshold,
                    timestamp=now,
                    confidence=random.uniform(0.4, 0.75),
                )
            )

        return alerts

    def _build_recent_logs(
        self, world_state: NexaStackWorldState
    ) -> list[LogEntry]:
        """Generate log entries for affected services, then apply suppression."""
        now = time.time()
        logs: list[LogEntry] = []

        for svc, metrics in world_state.services.items():
            if not metrics.availability:
                logs.append(
                    LogEntry(
                        service=svc,
                        timestamp=now,
                        level="ERROR",
                        message=f"Service {svc} is degraded: availability=False",
                    )
                )
                logs.append(
                    LogEntry(
                        service=svc,
                        timestamp=now - 1.0,
                        level="WARN",
                        message=f"High error rate detected on {svc}",
                    )
                )

        # Apply log suppression: randomly drop log_suppression_ratio fraction
        if self.log_suppression_ratio > 0.0 and logs:
            keep_count = max(
                0,
                int(len(logs) * (1.0 - self.log_suppression_ratio)),
            )
            logs = random.sample(logs, keep_count)

        return logs

    def _build_active_traces(
        self, world_state: NexaStackWorldState
    ) -> list[Trace]:
        """Generate traces for degraded services."""
        now = time.time()
        traces: list[Trace] = []

        for svc, metrics in world_state.services.items():
            if not metrics.availability:
                traces.append(
                    Trace(
                        trace_id=f"trace-{svc}-{int(now)}",
                        service=svc,
                        operation="request",
                        duration_ms=metrics.latency_ms,
                        status="error",
                    )
                )

        return traces

    def _build_incident_context(
        self, incident_state: IncidentState | None
    ) -> dict:
        """Build incident context dict from incident_state, or empty context."""
        if incident_state is None:
            return IncidentContext(
                incident_id="",
                timeline=[],
                active_hypotheses=[],
                attempted_remediations=[],
                current_blast_radius=[],
            ).model_dump()

        return IncidentContext(
            incident_id=incident_state.template_id,
            timeline=[
                {
                    "step": e.step,
                    "event_type": e.event_type,
                    "description": e.description,
                    "agent": e.agent,
                }
                for e in incident_state.timeline
            ],
            active_hypotheses=[
                {
                    "service": h.service,
                    "failure_type": h.failure_type.value,
                    "confidence": h.confidence,
                }
                for h in incident_state.active_hypotheses
            ],
            attempted_remediations=[
                a.model_dump() for a in incident_state.attempted_remediations
            ],
            current_blast_radius=list(incident_state.current_blast_radius),
        ).model_dump()

    def _build_sla_state(
        self, world_state: NexaStackWorldState
    ) -> dict[str, bool]:
        """Return SLA compliance per service: True if available, False otherwise."""
        return {
            svc: metrics.availability
            for svc, metrics in world_state.services.items()
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _worst_metric(
        metrics,
    ) -> tuple[str, float, float]:
        """Return (metric_name, value, threshold) for the most-breached metric."""
        candidates = [
            ("cpu", metrics.cpu, 0.85),
            ("memory", metrics.memory, 0.85),
            ("error_rate", metrics.error_rate, 0.05),
            ("saturation", metrics.saturation, 0.9),
            ("latency_ms", metrics.latency_ms, 500.0),
        ]
        # Pick the metric with the highest ratio of value/threshold
        worst = max(candidates, key=lambda t: t[1] / t[2])
        return worst
