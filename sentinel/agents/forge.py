"""FORGE — Solutions Architect Agent for SENTINEL.

Receives HypothesisTree + world state snapshot; produces ranked remediation plans.
Simulates CDG effect to estimate blast radius before submitting action.
Escalates to HERMES with warning flag when blast radius would expand.
Emits remediation actions only.

Requirements: 10.1, 10.2, 10.3, 10.5, 10.6
"""
from __future__ import annotations

from sentinel.agents.base import BaseAgent
from sentinel.models import Action, HypothesisTree


class FORGE(BaseAgent):
    """Solutions Architect agent.

    When model is None, uses heuristic: RestartService on the primary
    hypothesis service.
    """

    def __init__(self, model=None) -> None:
        self.model = model
        self._current_blast_radius: set[str] = set()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: dict) -> Action:
        """Produce a remediation action based on the current hypothesis tree.

        Returns an Action with category="remediation" or "meta" (escalation).
        """
        if self.model is not None:
            return self._model_act(observation)
        return self._heuristic_act(observation)

    def reset(self) -> None:
        """Reset FORGE state for a new episode."""
        self._current_blast_radius = set()

    # ------------------------------------------------------------------
    # Blast radius estimation
    # ------------------------------------------------------------------

    def estimate_blast_radius(
        self,
        service: str,
        world_state_snapshot: dict,
    ) -> set[str]:
        """Simulate the CDG effect of restarting *service*.

        Uses the CDG edges in the snapshot to find immediate successors
        (services that depend on *service*) as a proxy for blast radius.

        Returns the set of services that would be affected.
        """
        affected: set[str] = {service}
        cdg_edges: list[dict] = world_state_snapshot.get("cdg_edges", [])
        # Direct successors of the target service
        for edge in cdg_edges:
            if edge.get("src") == service:
                affected.add(edge["dst"])
        return affected

    # ------------------------------------------------------------------
    # Private: heuristic act (no model)
    # ------------------------------------------------------------------

    def _heuristic_act(self, observation: dict) -> Action:
        """Heuristic: RestartService on the primary hypothesis service."""
        incident_context = observation.get("incident_context", {})
        active_hypotheses = incident_context.get("active_hypotheses", [])
        world_state_snapshot = observation.get("metrics_snapshot", {})

        # Determine target service from hypotheses
        target_service: str | None = None
        if active_hypotheses:
            # Pick highest-confidence hypothesis
            best = max(
                active_hypotheses,
                key=lambda h: h.get("confidence", 0.0) if isinstance(h, dict) else getattr(h, "confidence", 0.0),
            )
            target_service = best.get("service") if isinstance(best, dict) else getattr(best, "service", None)

        if target_service is None:
            # Fall back to the first degraded service in metrics_snapshot
            metrics_snapshot = observation.get("metrics_snapshot", {})
            for svc, metrics in metrics_snapshot.items():
                if metrics is None:
                    continue
                avail = metrics.get("availability") if isinstance(metrics, dict) else getattr(metrics, "availability", True)
                if avail is False:
                    target_service = svc
                    break

        if target_service is None:
            # Nothing to remediate — escalate
            return Action(
                agent="forge",
                category="meta",
                name="EscalateToHuman",
                params={"reason": "No degraded service identified for remediation."},
            )

        # Estimate blast radius of the proposed action
        full_snapshot = observation.get("world_state_snapshot", {})
        estimated_br = self.estimate_blast_radius(target_service, full_snapshot)

        # Current blast radius from incident context
        current_br_list: list[str] = incident_context.get("current_blast_radius", [])
        current_br: set[str] = set(current_br_list)

        # Check if action would expand blast radius (Req 10.5)
        new_services = estimated_br - current_br
        if new_services and current_br:
            # Escalate to HERMES with warning flag
            return Action(
                agent="forge",
                category="meta",
                name="EscalateToHuman",
                params={
                    "reason": (
                        f"Blast radius would expand by {len(new_services)} service(s): "
                        f"{sorted(new_services)}. Escalating to HERMES."
                    ),
                    "warning": True,
                    "proposed_action": "RestartService",
                    "target_service": target_service,
                },
            )

        # Safe to proceed — emit RestartService
        return Action(
            agent="forge",
            category="remediation",
            name="RestartService",
            params={"service": target_service},
        )

    # ------------------------------------------------------------------
    # Private: model-based act
    # ------------------------------------------------------------------

    def _model_act(self, observation: dict) -> Action:
        """Call the fine-tuned GRPOTrainer model to produce an action."""
        result = self.model(observation)
        if isinstance(result, Action):
            return result
        return Action(**result)
