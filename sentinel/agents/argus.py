"""ARGUS — Observer Agent for SENTINEL.

Polls NexaStackWorldState each step, emits alerts with confidence scores,
and enforces investigative + meta actions only. Respects black-box constraints.
"""
from __future__ import annotations

from sentinel.agents.base import BaseAgent
from sentinel.models import Action
from sentinel.world_state import NexaStackWorldState, _THRESHOLDS

# Metric names that ARGUS monitors
_MONITORED_METRICS = ("cpu", "memory", "latency_ms", "error_rate", "saturation")


class ARGUS(BaseAgent):
    """Observer agent that monitors all NexaStack services and surfaces anomalies.

    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """

    def __init__(
        self,
        black_box_services: set[str] | None = None,
        alert_threshold_multiplier: float = 1.0,
    ) -> None:
        self.black_box_services: set[str] = black_box_services or set()
        self.alert_threshold_multiplier = alert_threshold_multiplier
        # Track which service/metric pair triggered the last alert
        self._last_alert_service: str | None = None
        self._last_alert_metric: str | None = None

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: dict) -> Action:
        """Scan metrics_snapshot for anomalies; return a QueryMetrics investigative action.

        Finds the service+metric with the highest deviation above threshold and
        emits a QueryMetrics action for it. Falls back to a meta NoOp-style
        EscalateToHuman if no anomaly is found (should be rare).

        Returns an Action with category="investigative" or "meta".
        """
        metrics_snapshot: dict = observation.get("metrics_snapshot", {})

        best_service: str | None = None
        best_metric: str | None = None
        best_confidence: float = 0.0

        for service, metrics in metrics_snapshot.items():
            # Respect black-box constraints — skip services with no metrics
            if metrics is None:
                continue
            if service in self.black_box_services:
                continue

            for metric_name in _MONITORED_METRICS:
                value = metrics.get(metric_name) if isinstance(metrics, dict) else getattr(metrics, metric_name, None)
                if value is None:
                    continue

                threshold = _THRESHOLDS.get(metric_name)
                if threshold is None:
                    continue

                effective_threshold = threshold * self.alert_threshold_multiplier
                if value > effective_threshold:
                    # Confidence = metric_value / threshold, clamped to [0.0, 1.0]
                    confidence = min(value / effective_threshold, 1.0)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_service = service
                        best_metric = metric_name

        if best_service is not None:
            self._last_alert_service = best_service
            self._last_alert_metric = best_metric
            return Action(
                agent="argus",
                category="investigative",
                name="QueryMetrics",
                params={
                    "service": best_service,
                    "metric_name": best_metric,
                    "time_range": [0, 60],
                    "confidence": best_confidence,
                },
            )

        # No anomaly detected — emit a meta action indicating all-clear
        return Action(
            agent="argus",
            category="meta",
            name="EscalateToHuman",
            params={"reason": "No anomalies detected in current observation window."},
        )

    def reset(self) -> None:
        """Reset ARGUS state for a new episode."""
        self._last_alert_service = None
        self._last_alert_metric = None

    # ------------------------------------------------------------------
    # Helper: build anomaly report for inclusion in observation
    # ------------------------------------------------------------------

    def build_anomaly_report(self, world_state: NexaStackWorldState) -> list[dict]:
        """Scan world state and return a list of alert dicts with confidence scores.

        This is called by the Observability_Layer to populate active_alerts.
        Black-box services are skipped (Req 8.3).
        """
        alerts: list[dict] = []
        import time

        now = time.time()
        for service, metrics in world_state.services.items():
            if service in self.black_box_services:
                continue

            for metric_name in _MONITORED_METRICS:
                value = getattr(metrics, metric_name, None)
                if value is None:
                    continue

                threshold = _THRESHOLDS.get(metric_name)
                if threshold is None:
                    continue

                effective_threshold = threshold * self.alert_threshold_multiplier
                if value > effective_threshold:
                    confidence = min(value / effective_threshold, 1.0)
                    alerts.append({
                        "service": service,
                        "metric": metric_name,
                        "value": value,
                        "threshold": effective_threshold,
                        "timestamp": now,
                        "confidence": confidence,
                    })

        return alerts
