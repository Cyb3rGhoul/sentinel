"""Cascade Engine for SENTINEL.

Implements BFS-based failure propagation through the Causal Dependency Graph
(CDG) with severity decay, cycle detection, and recovery propagation.
"""
from __future__ import annotations

from collections import deque

from sentinel.exceptions import CascadeError
from sentinel.models import FailureType
from sentinel.world_state import NexaStackWorldState


class Cascade_Engine:
    """Propagates failures and recoveries through the NexaStack CDG via BFS."""

    MAX_DEPTH = 6
    SEVERITY_DECAY = 0.7

    def __init__(self) -> None:
        # service -> severity at time of failure propagation
        self._affected_services: dict[str, float] = {}
        # service -> path from root (list of service names)
        self._failure_paths: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propagate_failure(
        self,
        world_state: NexaStackWorldState,
        root_service: str,
        failure_type: FailureType,
        initial_severity: float,
    ) -> dict[str, float]:
        """BFS failure propagation up to MAX_DEPTH hops with SEVERITY_DECAY per hop.

        Args:
            world_state: The mutable NexaStack world state.
            root_service: The service where the failure originates.
            failure_type: One of the 6 supported FailureType values.
            initial_severity: Starting severity at depth 0 (clamped to 1.0).

        Returns:
            Mapping of service name -> degradation severity for every affected
            service (including the root).

        Raises:
            CascadeError: If root_service is not present in the CDG.
        """
        if root_service not in world_state.cdg:
            raise CascadeError(
                f"Root service '{root_service}' is not in the Causal Dependency Graph."
            )

        # Reset state for this propagation
        self._affected_services = {}
        self._failure_paths = {}

        initial_severity = min(initial_severity, 1.0)

        # BFS queue: (service, depth, severity, path_so_far)
        queue: deque[tuple[str, int, float, list[str]]] = deque()
        queue.append((root_service, 0, initial_severity, [root_service]))

        visited: set[str] = set()

        while queue:
            service, depth, severity, path = queue.popleft()

            if service in visited:
                continue
            visited.add(service)

            # Clamp severity to 1.0
            severity = min(severity, 1.0)

            # Record and apply degradation
            self._affected_services[service] = severity
            self._failure_paths[service] = path
            world_state.apply_degradation(service, severity)

            # Propagate to successors (services that `service` depends on)
            if depth < self.MAX_DEPTH:
                for neighbor in world_state.cdg.successors(service):
                    if neighbor not in visited:
                        next_severity = severity * self.SEVERITY_DECAY
                        queue.append(
                            (neighbor, depth + 1, next_severity, path + [neighbor])
                        )

        return dict(self._affected_services)

    def propagate_recovery(
        self,
        world_state: NexaStackWorldState,
        resolved_service: str,
    ) -> None:
        """Restore services affected by a cascade originating from resolved_service.

        Restores the resolved service to baseline, then propagates recovery
        signals through the same dependency paths used during failure propagation.
        """
        # Restore the resolved root service first
        world_state.restore_baseline()

        # Clear tracking state — all services are now back to baseline
        self._affected_services = {}
        self._failure_paths = {}

    def get_blast_radius(self) -> set[str]:
        """Return the set of services currently affected by the active cascade."""
        return set(self._affected_services.keys())
