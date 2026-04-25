"""Core data models for SENTINEL."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FailureType(str, Enum):
    memory_leak = "memory_leak"
    connection_pool_exhaustion = "connection_pool_exhaustion"
    cpu_spike = "cpu_spike"
    bad_deployment = "bad_deployment"
    cache_miss_storm = "cache_miss_storm"
    network_partition = "network_partition"


# ---------------------------------------------------------------------------
# Service / World State models
# ---------------------------------------------------------------------------

@dataclass
class ServiceMetrics:
    cpu: float          # 0.0–1.0
    memory: float       # 0.0–1.0
    latency_ms: float
    error_rate: float   # 0.0–1.0
    saturation: float   # 0.0–1.0
    availability: bool


# ---------------------------------------------------------------------------
# Hypothesis Tree
# ---------------------------------------------------------------------------

@dataclass
class HypothesisNode:
    service: str
    failure_type: FailureType
    confidence: float   # Bayesian score 0.0–1.0
    children: list["HypothesisNode"] = field(default_factory=list)


class HypothesisTree:
    def __init__(self, root: HypothesisNode | None = None) -> None:
        self.root = root

    def update_confidences(self, observation: dict) -> None:
        """Update confidence scores of all nodes using new evidence."""
        if self.root is None:
            return
        self._update_node(self.root, observation)

    def _update_node(self, node: HypothesisNode, observation: dict) -> None:
        alerts = observation.get("active_alerts", [])
        alert_services = {
            (a.service if hasattr(a, "service") else a.get("service")) for a in alerts
        }
        if node.service in alert_services:
            node.confidence = min(1.0, node.confidence * 1.1)
        for child in node.children:
            self._update_node(child, observation)

    def get_primary_candidate(self, threshold: float = 0.85) -> HypothesisNode | None:
        """Return the highest-confidence node above threshold, or None."""
        if self.root is None:
            return None
        return self._find_best(self.root, threshold)

    def _find_best(self, node: HypothesisNode, threshold: float) -> HypothesisNode | None:
        best = node if node.confidence >= threshold else None
        for child in node.children:
            candidate = self._find_best(child, threshold)
            if candidate is not None:
                if best is None or candidate.confidence > best.confidence:
                    best = candidate
        return best


# ---------------------------------------------------------------------------
# Timeline / Incident models
# ---------------------------------------------------------------------------

@dataclass
class TimelineEntry:
    step: int
    event_type: str
    description: str
    agent: str


@dataclass
class IncidentState:
    template_id: str
    root_cause_service: str
    failure_type: FailureType
    ground_truth_signals: list[str]
    red_herring_signals: list[str]
    affected_services: dict[str, float]       # service -> severity
    peak_blast_radius: set[str]
    current_blast_radius: set[str]
    timeline: list[TimelineEntry]
    attempted_remediations: list["Action"]
    active_hypotheses: list[HypothesisNode]
    resolved: bool
    step_injected: int


@dataclass
class IncidentTemplate:
    id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    root_cause_service: str
    failure_type: FailureType
    ground_truth_signals: list[str]
    red_herring_signals: list[str]
    cascade_risk: Literal["low", "medium", "high"]
    missing_log_ratio: float
    expected_steps_to_resolve: tuple[int, int]


# ---------------------------------------------------------------------------
# Action models (Pydantic v2)
# ---------------------------------------------------------------------------

class Action(BaseModel):
    agent: Literal["argus", "holmes", "forge", "hermes", "oracle"]
    category: Literal["investigative", "remediation", "deployment", "meta"]
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


# Investigative sub-actions
class QueryLogs(BaseModel):
    service: str
    time_range: tuple[int, int]


class QueryTrace(BaseModel):
    trace_id: str


class QueryMetrics(BaseModel):
    service: str
    metric_name: str
    time_range: tuple[int, int]


class FormHypothesis(BaseModel):
    service: str
    failure_type: str
    confidence: float


# Remediation sub-actions
class RestartService(BaseModel):
    service: str


class ScaleService(BaseModel):
    service: str
    replicas: int


class ModifyConfig(BaseModel):
    service: str
    key: str
    value: str


class RollbackDeployment(BaseModel):
    service: str
    version: str


class DrainTraffic(BaseModel):
    service: str


class ModifyRateLimit(BaseModel):
    service: str
    limit_rps: int


# Deployment sub-actions
class CanaryDeploy(BaseModel):
    service: str
    version: str
    traffic_percent: float


class FullDeploy(BaseModel):
    service: str
    version: str


class Rollback(BaseModel):
    service: str


# Meta sub-actions
class GenerateNewScenario(BaseModel):
    difficulty: str
    target_gap: str


class EscalateToHuman(BaseModel):
    reason: str


class CloseIncident(BaseModel):
    resolution_summary: str


# ---------------------------------------------------------------------------
# Observability models (Pydantic v2)
# ---------------------------------------------------------------------------

class Alert(BaseModel):
    service: str
    metric: str
    value: float
    threshold: float
    timestamp: float
    confidence: float = Field(ge=0.0, le=1.0)


class LogEntry(BaseModel):
    service: str
    timestamp: float
    level: str
    message: str


class Trace(BaseModel):
    trace_id: str
    service: str
    operation: str
    duration_ms: float
    status: str


class IncidentContext(BaseModel):
    incident_id: str
    timeline: list[dict]
    active_hypotheses: list[dict]
    attempted_remediations: list[dict]
    current_blast_radius: list[str]


# ---------------------------------------------------------------------------
# Observation TypedDict
# ---------------------------------------------------------------------------

class Observation(TypedDict):
    metrics_snapshot: dict[str, Any]          # ServiceMetrics | None per service
    causal_graph_snapshot: list[list[float]]  # 30×30 adjacency matrix
    active_alerts: list[Alert]
    recent_logs: list[LogEntry]
    active_traces: list[Trace]
    incident_context: dict
    sla_state: dict[str, bool]


# ---------------------------------------------------------------------------
# Reward models
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    r1_root_cause: float = 0.35
    r2_mttr: float = 0.30
    r3_recovery_quality: float = 0.25
    r4_blast_radius: float = 0.10


@dataclass
class RewardBreakdown:
    r1: float   # root cause accuracy
    r2: float   # MTTR score
    r3: float   # recovery quality
    r4: float   # blast radius minimization
    penalties: float
    total: float


# ---------------------------------------------------------------------------
# Multi-Agent Communication Protocol
# ---------------------------------------------------------------------------

@dataclass
class AgentMessage:
    """Inter-agent communication message."""
    sender: str          # e.g. "holmes"
    receiver: str        # e.g. "forge"
    message_type: str    # "hypothesis_confirmed", "remediation_needed", "status_update"
    payload: dict        # message-specific data
    step: int


class MessageBus:
    """Shared communication channel for multi-agent coordination.

    Enables the investigate→diagnose→remediate→verify workflow:
    1. ARGUS monitors → alerts → sends to HOLMES
    2. HOLMES investigates → forms hypothesis → sends to FORGE when confident
    3. FORGE remediates → sends result to ORACLE
    4. ORACLE evaluates → decides CloseIncident or escalate
    """

    def __init__(self) -> None:
        self._messages: list[AgentMessage] = []

    def send(self, msg: AgentMessage) -> None:
        """Send a message to a specific agent."""
        self._messages.append(msg)

    def receive(self, agent_id: str, since_step: int = 0) -> list[AgentMessage]:
        """Retrieve messages for a specific agent since a given step."""
        return [
            m for m in self._messages
            if m.receiver == agent_id and m.step >= since_step
        ]

    def broadcast(self, sender: str, message_type: str, payload: dict, step: int) -> None:
        """Broadcast a message to all agents."""
        for agent_id in ("argus", "holmes", "forge", "hermes", "oracle"):
            if agent_id != sender:
                self._messages.append(
                    AgentMessage(
                        sender=sender,
                        receiver=agent_id,
                        message_type=message_type,
                        payload=payload,
                        step=step,
                    )
                )

    def clear(self) -> None:
        """Clear all messages (called on episode reset)."""
        self._messages.clear()

    @property
    def messages(self) -> list[AgentMessage]:
        return list(self._messages)


# ---------------------------------------------------------------------------
# Trajectory models
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryStep:
    observation: dict
    action: Action
    reward: float
    terminated: bool
    truncated: bool
    info: dict


@dataclass
class Trajectory:
    episode_id: str
    incident_template_id: str
    steps: list[TrajectoryStep]
    final_reward: RewardBreakdown
    mttr: int

    def to_json(self) -> str:
        """Serialize trajectory to JSON string."""
        def _serialize_step(s: TrajectoryStep) -> dict:
            return {
                "observation": s.observation,
                "action": s.action.model_dump(),
                "reward": s.reward,
                "terminated": s.terminated,
                "truncated": s.truncated,
                "info": s.info,
            }

        data = {
            "episode_id": self.episode_id,
            "incident_template_id": self.incident_template_id,
            "steps": [_serialize_step(s) for s in self.steps],
            "final_reward": {
                "r1": self.final_reward.r1,
                "r2": self.final_reward.r2,
                "r3": self.final_reward.r3,
                "r4": self.final_reward.r4,
                "penalties": self.final_reward.penalties,
                "total": self.final_reward.total,
            },
            "mttr": self.mttr,
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: str) -> "Trajectory":
        """Deserialize trajectory from JSON string."""
        raw = json.loads(data)
        steps = [
            TrajectoryStep(
                observation=s["observation"],
                action=Action(**s["action"]),
                reward=s["reward"],
                terminated=s["terminated"],
                truncated=s["truncated"],
                info=s["info"],
            )
            for s in raw["steps"]
        ]
        final_reward = RewardBreakdown(**raw["final_reward"])
        return cls(
            episode_id=raw["episode_id"],
            incident_template_id=raw["incident_template_id"],
            steps=steps,
            final_reward=final_reward,
            mttr=raw["mttr"],
        )
