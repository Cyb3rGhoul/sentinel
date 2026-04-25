"""HERMES — Deployment Agent for SENTINEL.

Executes canary deployments; monitors error_rate during observation window.
Auto-rollback when error_rate exceeds pre-canary baseline.
Validates SLA compliance after successful deployment.
Records post-mortem timeline entries.
Emits deployment + CloseIncident actions only.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
"""
from __future__ import annotations

import time

from sentinel.agents.base import BaseAgent
from sentinel.models import Action, TimelineEntry


class HERMES(BaseAgent):
    """Deployment agent with canary strategy and automated rollback.

    observation_window: number of steps to monitor before promoting canary.
    """

    def __init__(self, observation_window: int = 5) -> None:
        self.observation_window: int = observation_window
        self._timeline: list[TimelineEntry] = []
        self._step: int = 0
        # Canary state
        self._canary_service: str | None = None
        self._canary_version: str | None = None
        self._canary_traffic_percent: float = 0.0
        self._pre_canary_error_rate: float = 0.0
        self._canary_step_start: int = 0
        self._canary_active: bool = False

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: dict) -> Action:
        """Emit deployment or CloseIncident actions.

        If a canary is active, monitor it and decide to promote or rollback.
        Otherwise, check if the incident is resolved and close it.
        """
        self._step += 1

        if self._canary_active:
            return self._monitor_canary(observation)

        # Check if incident is resolved — all services SLA-compliant
        sla_state: dict[str, bool] = observation.get("sla_state", {})
        incident_context = observation.get("incident_context", {})
        current_blast_radius: list[str] = incident_context.get("current_blast_radius", [])

        all_compliant = all(sla_state.get(svc, True) for svc in current_blast_radius)
        if all_compliant and current_blast_radius:
            self._record_timeline_entry(
                event_type="incident_resolved",
                description="All affected services returned to SLA-compliant state.",
                agent="hermes",
            )
            return Action(
                agent="hermes",
                category="meta",
                name="CloseIncident",
                params={"resolution_summary": "All services SLA-compliant after deployment."},
            )

        # Default: initiate a canary deploy for the first degraded service
        metrics_snapshot = observation.get("metrics_snapshot", {})
        for svc, metrics in metrics_snapshot.items():
            if metrics is None:
                continue
            avail = metrics.get("availability") if isinstance(metrics, dict) else getattr(metrics, "availability", True)
            if avail is False:
                error_rate = metrics.get("error_rate") if isinstance(metrics, dict) else getattr(metrics, "error_rate", 0.0)
                self._start_canary(svc, "v-rollback", traffic_percent=10.0, baseline_error_rate=error_rate)
                self._record_timeline_entry(
                    event_type="canary_started",
                    description=f"Canary deploy started for {svc} at 10% traffic.",
                    agent="hermes",
                )
                return Action(
                    agent="hermes",
                    category="deployment",
                    name="CanaryDeploy",
                    params={
                        "service": svc,
                        "version": "v-rollback",
                        "traffic_percent": 10.0,
                    },
                )

        # Nothing to do
        return Action(
            agent="hermes",
            category="meta",
            name="CloseIncident",
            params={"resolution_summary": "No degraded services detected; incident closed."},
        )

    def reset(self) -> None:
        """Reset HERMES state for a new episode."""
        self._timeline = []
        self._step = 0
        self._canary_service = None
        self._canary_version = None
        self._canary_traffic_percent = 0.0
        self._pre_canary_error_rate = 0.0
        self._canary_step_start = 0
        self._canary_active = False

    # ------------------------------------------------------------------
    # Canary execution logic (Req 11.1–11.5)
    # ------------------------------------------------------------------

    def execute_canary(
        self,
        service: str,
        version: str,
        traffic_percent: float,
        world_state,  # NexaStackWorldState
    ) -> tuple[str, str | None]:
        """Execute a canary deployment and monitor for observation_window steps.

        Monitors error_rate for `observation_window` steps. Auto-rolls back
        when error_rate exceeds pre-canary baseline.

        Returns:
            ("promoted", None) on success.
            ("rolled_back", reason) on failure.
        """
        from sentinel.world_state import NexaStackWorldState

        if service not in world_state.services:
            return ("rolled_back", f"Service '{service}' not found in world state.")

        # Record pre-canary baseline error_rate
        pre_canary_error_rate = world_state.services[service].error_rate

        # Simulate observation window
        for step in range(self.observation_window):
            current_error_rate = world_state.services[service].error_rate

            if current_error_rate > pre_canary_error_rate:
                # Auto-rollback (Req 11.3)
                reason = (
                    f"error_rate {current_error_rate:.4f} exceeded pre-canary baseline "
                    f"{pre_canary_error_rate:.4f} at step {step + 1}/{self.observation_window}"
                )
                self._record_timeline_entry(
                    event_type="canary_rollback",
                    description=f"Canary for {service}@{version} rolled back: {reason}",
                    agent="hermes",
                )
                return ("rolled_back", reason)

        # Observation window passed — validate SLA compliance (Req 11.4)
        sla_violation = self._check_sla_violation(service, world_state)
        if sla_violation:
            reason = f"SLA violation after canary: {sla_violation}"
            self._record_timeline_entry(
                event_type="canary_rollback",
                description=f"Canary for {service}@{version} rolled back: {reason}",
                agent="hermes",
            )
            return ("rolled_back", reason)

        # Promote (Req 11.4)
        self._record_timeline_entry(
            event_type="canary_promoted",
            description=f"Canary for {service}@{version} promoted to full deployment.",
            agent="hermes",
        )
        return ("promoted", None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _start_canary(
        self,
        service: str,
        version: str,
        traffic_percent: float,
        baseline_error_rate: float,
    ) -> None:
        self._canary_service = service
        self._canary_version = version
        self._canary_traffic_percent = traffic_percent
        self._pre_canary_error_rate = baseline_error_rate
        self._canary_step_start = self._step
        self._canary_active = True

    def _monitor_canary(self, observation: dict) -> Action:
        """Check canary health; promote or rollback based on error_rate."""
        assert self._canary_service is not None

        metrics_snapshot = observation.get("metrics_snapshot", {})
        service_metrics = metrics_snapshot.get(self._canary_service)

        if service_metrics is not None:
            current_error_rate = (
                service_metrics.get("error_rate")
                if isinstance(service_metrics, dict)
                else getattr(service_metrics, "error_rate", 0.0)
            )
            if current_error_rate is not None and current_error_rate > self._pre_canary_error_rate:
                # Rollback (Req 11.3)
                reason = (
                    f"error_rate {current_error_rate:.4f} exceeded baseline "
                    f"{self._pre_canary_error_rate:.4f}"
                )
                self._record_timeline_entry(
                    event_type="canary_rollback",
                    description=f"Canary for {self._canary_service} rolled back: {reason}",
                    agent="hermes",
                )
                self._canary_active = False
                return Action(
                    agent="hermes",
                    category="deployment",
                    name="Rollback",
                    params={"service": self._canary_service},
                )

        # Check if observation window has elapsed
        steps_elapsed = self._step - self._canary_step_start
        if steps_elapsed >= self.observation_window:
            # Promote
            self._record_timeline_entry(
                event_type="canary_promoted",
                description=f"Canary for {self._canary_service} promoted after {steps_elapsed} steps.",
                agent="hermes",
            )
            self._canary_active = False
            return Action(
                agent="hermes",
                category="deployment",
                name="FullDeploy",
                params={
                    "service": self._canary_service,
                    "version": self._canary_version or "latest",
                },
            )

        # Still monitoring — emit another CanaryDeploy to signal continuation
        return Action(
            agent="hermes",
            category="deployment",
            name="CanaryDeploy",
            params={
                "service": self._canary_service,
                "version": self._canary_version or "latest",
                "traffic_percent": self._canary_traffic_percent,
            },
        )

    def _check_sla_violation(self, service: str, world_state) -> str | None:
        """Return a violation description if service is not SLA-compliant, else None."""
        from sentinel.world_state import _THRESHOLDS

        metrics = world_state.services.get(service)
        if metrics is None:
            return None

        if metrics.error_rate > _THRESHOLDS.get("error_rate", 0.05):
            return f"error_rate={metrics.error_rate:.4f} > threshold={_THRESHOLDS['error_rate']}"
        if metrics.latency_ms > _THRESHOLDS.get("latency_ms", 500.0):
            return f"latency_ms={metrics.latency_ms:.1f} > threshold={_THRESHOLDS['latency_ms']}"
        return None

    def _record_timeline_entry(self, event_type: str, description: str, agent: str) -> None:
        """Append a post-mortem timeline entry (Req 11.5)."""
        self._timeline.append(
            TimelineEntry(
                step=self._step,
                event_type=event_type,
                description=description,
                agent=agent,
            )
        )

    @property
    def timeline(self) -> list[TimelineEntry]:
        """Return the post-mortem timeline recorded by HERMES."""
        return list(self._timeline)
