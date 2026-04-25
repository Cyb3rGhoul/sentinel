"""NexaStack world state for SENTINEL.

Implements the 30-service topology across 4 layers with a Causal Dependency
Graph (CDG) and full serialization support.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from sentinel.exceptions import CascadeError
from sentinel.models import FailureType, IncidentState, ServiceMetrics

# ---------------------------------------------------------------------------
# Service topology
# ---------------------------------------------------------------------------

FRONTEND_SERVICES: list[str] = [
    "web-gateway",
    "mobile-api",
    "cdn-edge",
]

APPLICATION_SERVICES: list[str] = [
    "cart-service",
    "order-service",
    "product-catalog",
    "search-service",
    "recommendation-engine",
    "user-auth",
    "notification-service",
    "pricing-engine",
    "inventory-service",
    "review-service",
    "wishlist-service",
    "session-manager",
]

DATA_SERVICES: list[str] = [
    "postgres-primary",
    "postgres-replica",
    "redis-cache",
    "elasticsearch",
    "kafka-broker",
    "object-storage",
    "analytics-db",
    "audit-log",
]

INFRASTRUCTURE_SERVICES: list[str] = [
    "service-mesh",
    "load-balancer",
    "api-gateway",
    "config-service",
    "secret-manager",
    "payment-vault",
    "fraud-detector",
]

ALL_SERVICES: list[str] = (
    FRONTEND_SERVICES
    + APPLICATION_SERVICES
    + DATA_SERVICES
    + INFRASTRUCTURE_SERVICES
)

assert len(ALL_SERVICES) == 30, f"Expected 30 services, got {len(ALL_SERVICES)}"

# ---------------------------------------------------------------------------
# Baseline healthy metrics
# ---------------------------------------------------------------------------

_BASELINE: dict[str, Any] = dict(
    cpu=0.2,
    memory=0.3,
    latency_ms=50.0,
    error_rate=0.01,
    saturation=0.3,
    availability=True,
)

# ---------------------------------------------------------------------------
# Degradation thresholds — crossing any makes availability=False
# ---------------------------------------------------------------------------

_THRESHOLDS: dict[str, float] = {
    "cpu": 0.85,
    "memory": 0.85,
    "error_rate": 0.05,
    "saturation": 0.9,
    "latency_ms": 500.0,
}

# ---------------------------------------------------------------------------
# CDG edge definitions  (source depends-on target, i.e. source → target)
# ---------------------------------------------------------------------------

_CDG_EDGES: list[tuple[str, str, float]] = [
    # ── Frontend → Application ──────────────────────────────────────────────
    ("web-gateway",   "cart-service",          0.8),
    ("web-gateway",   "order-service",         0.8),
    ("web-gateway",   "product-catalog",       0.7),
    ("web-gateway",   "search-service",        0.7),
    ("web-gateway",   "user-auth",             0.9),
    ("web-gateway",   "pricing-engine",        0.6),
    ("web-gateway",   "recommendation-engine", 0.5),
    ("mobile-api",    "cart-service",          0.7),
    ("mobile-api",    "order-service",         0.7),
    ("mobile-api",    "user-auth",             0.9),
    ("mobile-api",    "notification-service",  0.6),
    ("mobile-api",    "session-manager",       0.8),
    ("cdn-edge",      "product-catalog",       0.6),
    ("cdn-edge",      "search-service",        0.5),
    ("cdn-edge",      "recommendation-engine", 0.4),
    # ── Application → Data ──────────────────────────────────────────────────
    ("cart-service",          "redis-cache",       0.9),
    ("cart-service",          "postgres-primary",  0.7),
    ("order-service",         "postgres-primary",  0.8),
    ("order-service",         "kafka-broker",      0.7),
    ("order-service",         "audit-log",         0.5),
    ("product-catalog",       "postgres-primary",  0.6),
    ("product-catalog",       "elasticsearch",     0.8),
    ("product-catalog",       "object-storage",    0.5),
    ("search-service",        "elasticsearch",     0.9),
    ("search-service",        "redis-cache",       0.6),
    ("recommendation-engine", "redis-cache",       0.7),
    ("recommendation-engine", "analytics-db",      0.6),
    ("recommendation-engine", "postgres-replica",  0.5),
    ("user-auth",             "postgres-primary",  0.8),
    ("user-auth",             "redis-cache",       0.7),
    ("user-auth",             "secret-manager",    0.9),
    ("notification-service",  "kafka-broker",      0.8),
    ("notification-service",  "audit-log",         0.4),
    ("pricing-engine",        "redis-cache",       0.7),
    ("pricing-engine",        "postgres-replica",  0.5),
    ("inventory-service",     "postgres-primary",  0.7),
    ("inventory-service",     "kafka-broker",      0.6),
    ("review-service",        "postgres-primary",  0.6),
    ("review-service",        "object-storage",    0.4),
    ("wishlist-service",      "redis-cache",       0.8),
    ("wishlist-service",      "postgres-primary",  0.5),
    ("session-manager",       "redis-cache",       0.9),
    ("session-manager",       "postgres-replica",  0.4),
    # ── Data → Infrastructure ───────────────────────────────────────────────
    ("postgres-primary",  "service-mesh",   0.7),
    ("postgres-primary",  "load-balancer",  0.6),
    ("postgres-replica",  "service-mesh",   0.6),
    ("postgres-replica",  "load-balancer",  0.5),
    ("redis-cache",       "service-mesh",   0.7),
    ("redis-cache",       "load-balancer",  0.5),
    ("elasticsearch",     "service-mesh",   0.7),
    ("elasticsearch",     "load-balancer",  0.6),
    ("kafka-broker",      "service-mesh",   0.8),
    ("kafka-broker",      "load-balancer",  0.6),
    ("object-storage",    "service-mesh",   0.5),
    ("object-storage",    "api-gateway",    0.4),
    ("analytics-db",      "service-mesh",   0.5),
    ("analytics-db",      "load-balancer",  0.4),
    ("audit-log",         "service-mesh",   0.4),
    ("audit-log",         "api-gateway",    0.3),
    # ── Infrastructure internal ─────────────────────────────────────────────
    ("service-mesh",   "api-gateway",    0.8),
    ("service-mesh",   "config-service", 0.7),
    ("load-balancer",  "api-gateway",    0.7),
    ("load-balancer",  "service-mesh",   0.6),
    ("api-gateway",    "config-service", 0.6),
    ("api-gateway",    "secret-manager", 0.7),
    ("config-service", "secret-manager", 0.5),
    ("payment-vault",  "service-mesh",   0.8),
    ("payment-vault",  "secret-manager", 0.9),
    ("fraud-detector", "service-mesh",   0.7),
    ("fraud-detector", "analytics-db",   0.6),
]


def _build_cdg() -> nx.DiGraph:
    """Build the Causal Dependency Graph from the hard-coded edge list."""
    g = nx.DiGraph()
    g.add_nodes_from(ALL_SERVICES)
    for src, dst, weight in _CDG_EDGES:
        g.add_edge(src, dst, weight=weight)
    return g


def _baseline_metrics() -> ServiceMetrics:
    return ServiceMetrics(**_BASELINE)


def _baseline_services() -> dict[str, ServiceMetrics]:
    return {svc: _baseline_metrics() for svc in ALL_SERVICES}


# ---------------------------------------------------------------------------
# NexaStackWorldState
# ---------------------------------------------------------------------------

@dataclass
class NexaStackWorldState:
    """Mutable world state for the NexaStack simulation."""

    services: dict[str, ServiceMetrics] = field(
        default_factory=_baseline_services
    )
    cdg: nx.DiGraph = field(default_factory=_build_cdg)
    incident_state: IncidentState | None = None
    step: int = 0

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def restore_baseline(self) -> None:
        """Reset all 30 services to their healthy baseline metric values."""
        for svc in ALL_SERVICES:
            self.services[svc] = _baseline_metrics()
        self.step = 0
        self.incident_state = None

    # ------------------------------------------------------------------
    # Degradation
    # ------------------------------------------------------------------

    def apply_degradation(self, service: str, severity: float) -> None:
        """Degrade *service* metrics proportionally to *severity* (clamped to 1.0).

        Availability is set to False when any metric crosses its threshold.
        """
        if service not in self.services:
            raise CascadeError(f"Service '{service}' not found in world state.")

        severity = min(severity, 1.0)
        m = self.services[service]

        new_cpu = min(m.cpu + severity * (1.0 - _BASELINE["cpu"]), 1.0)
        new_memory = min(m.memory + severity * (1.0 - _BASELINE["memory"]), 1.0)
        new_latency = m.latency_ms + severity * 950.0          # up to 1000 ms at sev=1
        new_error_rate = min(m.error_rate + severity * 0.99, 1.0)
        new_saturation = min(m.saturation + severity * (1.0 - _BASELINE["saturation"]), 1.0)

        availability = not (
            new_cpu > _THRESHOLDS["cpu"]
            or new_memory > _THRESHOLDS["memory"]
            or new_error_rate > _THRESHOLDS["error_rate"]
            or new_saturation > _THRESHOLDS["saturation"]
            or new_latency > _THRESHOLDS["latency_ms"]
        )

        self.services[service] = ServiceMetrics(
            cpu=new_cpu,
            memory=new_memory,
            latency_ms=new_latency,
            error_rate=new_error_rate,
            saturation=new_saturation,
            availability=availability,
        )

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a JSON-serializable dict of the current world state."""
        services_dict: dict[str, dict] = {}
        for svc, m in self.services.items():
            services_dict[svc] = {
                "cpu": m.cpu,
                "memory": m.memory,
                "latency_ms": m.latency_ms,
                "error_rate": m.error_rate,
                "saturation": m.saturation,
                "availability": m.availability,
            }

        # Serialize CDG edges
        edges = [
            {"src": u, "dst": v, "weight": d["weight"]}
            for u, v, d in self.cdg.edges(data=True)
        ]

        # Serialize incident_state
        incident: dict | None = None
        if self.incident_state is not None:
            inc = self.incident_state
            incident = {
                "template_id": inc.template_id,
                "root_cause_service": inc.root_cause_service,
                "failure_type": inc.failure_type.value,
                "ground_truth_signals": inc.ground_truth_signals,
                "red_herring_signals": inc.red_herring_signals,
                "affected_services": inc.affected_services,
                "peak_blast_radius": list(inc.peak_blast_radius),
                "current_blast_radius": list(inc.current_blast_radius),
                "timeline": [
                    {
                        "step": e.step,
                        "event_type": e.event_type,
                        "description": e.description,
                        "agent": e.agent,
                    }
                    for e in inc.timeline
                ],
                "attempted_remediations": [
                    a.model_dump() for a in inc.attempted_remediations
                ],
                "active_hypotheses": [
                    {
                        "service": h.service,
                        "failure_type": h.failure_type.value,
                        "confidence": h.confidence,
                    }
                    for h in inc.active_hypotheses
                ],
                "resolved": inc.resolved,
                "step_injected": inc.step_injected,
            }

        return {
            "services": services_dict,
            "cdg_edges": edges,
            "incident_state": incident,
            "step": self.step,
        }

    def to_json(self) -> str:
        """Serialize world state to a JSON string."""
        return json.dumps(self.snapshot())

    @classmethod
    def from_json(cls, data: str) -> "NexaStackWorldState":
        """Deserialize a world state from a JSON string produced by `to_json()`."""
        from sentinel.models import (
            Action,
            HypothesisNode,
            IncidentState,
            TimelineEntry,
        )

        raw: dict = json.loads(data)

        # Rebuild services
        services: dict[str, ServiceMetrics] = {}
        for svc, m in raw["services"].items():
            services[svc] = ServiceMetrics(
                cpu=m["cpu"],
                memory=m["memory"],
                latency_ms=m["latency_ms"],
                error_rate=m["error_rate"],
                saturation=m["saturation"],
                availability=m["availability"],
            )

        # Rebuild CDG
        cdg = nx.DiGraph()
        cdg.add_nodes_from(ALL_SERVICES)
        for edge in raw.get("cdg_edges", []):
            cdg.add_edge(edge["src"], edge["dst"], weight=edge["weight"])

        # Rebuild incident state
        incident_state: IncidentState | None = None
        if raw.get("incident_state") is not None:
            inc = raw["incident_state"]
            from sentinel.models import TimelineEntry as TE

            timeline = [
                TE(
                    step=e["step"],
                    event_type=e["event_type"],
                    description=e["description"],
                    agent=e["agent"],
                )
                for e in inc.get("timeline", [])
            ]
            remediations = [
                Action(**a) for a in inc.get("attempted_remediations", [])
            ]
            hypotheses = [
                HypothesisNode(
                    service=h["service"],
                    failure_type=FailureType(h["failure_type"]),
                    confidence=h["confidence"],
                )
                for h in inc.get("active_hypotheses", [])
            ]
            incident_state = IncidentState(
                template_id=inc["template_id"],
                root_cause_service=inc["root_cause_service"],
                failure_type=FailureType(inc["failure_type"]),
                ground_truth_signals=inc["ground_truth_signals"],
                red_herring_signals=inc["red_herring_signals"],
                affected_services=inc["affected_services"],
                peak_blast_radius=set(inc["peak_blast_radius"]),
                current_blast_radius=set(inc["current_blast_radius"]),
                timeline=timeline,
                attempted_remediations=remediations,
                active_hypotheses=hypotheses,
                resolved=inc["resolved"],
                step_injected=inc["step_injected"],
            )

        return cls(
            services=services,
            cdg=cdg,
            incident_state=incident_state,
            step=raw.get("step", 0),
        )
