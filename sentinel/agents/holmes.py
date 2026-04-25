"""HOLMES — Root Cause Detective Agent for SENTINEL.

Maintains a HypothesisTree, updates confidences from observations,
and emits FormHypothesis + investigative actions only.
Wired to GRPOTrainer interface (model=None uses heuristic).

Requirements: 9.1, 9.2, 9.3, 9.5, 9.6
"""
from __future__ import annotations

from collections import Counter

from sentinel.agents.base import BaseAgent
from sentinel.models import Action, FailureType, HypothesisNode, HypothesisTree


class HOLMES(BaseAgent):
    """Root Cause Detective agent.

    When model is None, uses a heuristic: picks the service with the most
    alerts as the primary hypothesis candidate.
    """

    def __init__(self, model=None) -> None:
        self.model = model
        self.hypothesis_tree: HypothesisTree = HypothesisTree()
        self._step: int = 0

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, observation: dict) -> Action:
        """Emit FormHypothesis or QueryLogs/QueryMetrics based on current evidence.

        Returns an Action with category="investigative".
        """
        self._step += 1

        # Update existing hypothesis confidences with new evidence
        self.hypothesis_tree.update_confidences(observation)

        if self.model is not None:
            return self._model_act(observation)

        return self._heuristic_act(observation)

    def reset(self) -> None:
        """Reset HOLMES state for a new episode."""
        self.hypothesis_tree = HypothesisTree()
        self._step = 0

    # ------------------------------------------------------------------
    # HypothesisTree helpers (exposed for external use)
    # ------------------------------------------------------------------

    def update_confidences(self, observation: dict) -> None:
        """Delegate to HypothesisTree.update_confidences()."""
        self.hypothesis_tree.update_confidences(observation)

    def get_primary_candidate(self, threshold: float = 0.85) -> HypothesisNode | None:
        """Return the highest-confidence hypothesis above threshold, or None."""
        return self.hypothesis_tree.get_primary_candidate(threshold)

    # ------------------------------------------------------------------
    # Private: heuristic act (no model)
    # ------------------------------------------------------------------

    def _heuristic_act(self, observation: dict) -> Action:
        """Heuristic: pick the service with the most alerts as the hypothesis."""
        alerts = observation.get("active_alerts", [])

        # Count alerts per service
        service_counts: Counter = Counter()
        for alert in alerts:
            svc = alert.service if hasattr(alert, "service") else alert.get("service")
            if svc:
                service_counts[svc] += 1

        # Check if we already have a strong primary candidate
        primary = self.hypothesis_tree.get_primary_candidate(threshold=0.85)
        if primary is not None:
            # Emit a QueryLogs action to gather more evidence
            return Action(
                agent="holmes",
                category="investigative",
                name="QueryLogs",
                params={
                    "service": primary.service,
                    "time_range": [0, 300],
                },
            )

        if not service_counts:
            # No alerts — query metrics on a broad basis
            return Action(
                agent="holmes",
                category="investigative",
                name="QueryMetrics",
                params={
                    "service": "web-gateway",
                    "metric_name": "error_rate",
                    "time_range": [0, 300],
                },
            )

        # Pick the most-alerted service
        top_service, _ = service_counts.most_common(1)[0]

        # Derive a confidence from alert count (normalised, capped at 0.9)
        total_alerts = sum(service_counts.values())
        confidence = min(service_counts[top_service] / max(total_alerts, 1), 0.9)

        # Add hypothesis node to tree
        node = HypothesisNode(
            service=top_service,
            failure_type=FailureType.cpu_spike,  # default; model would refine this
            confidence=confidence,
        )
        if self.hypothesis_tree.root is None:
            self.hypothesis_tree.root = node
        else:
            self.hypothesis_tree.root.children.append(node)

        return Action(
            agent="holmes",
            category="investigative",
            name="FormHypothesis",
            params={
                "service": top_service,
                "failure_type": FailureType.cpu_spike.value,
                "confidence": confidence,
            },
        )

    # ------------------------------------------------------------------
    # Private: model-based act
    # ------------------------------------------------------------------

    def _model_act(self, observation: dict) -> Action:
        """Call the fine-tuned GRPOTrainer model to produce an action."""
        # The model is expected to return an Action-compatible dict
        result = self.model(observation)
        if isinstance(result, Action):
            return result
        # Assume dict
        return Action(**result)
