"""Sentinel_Env — top-level Gymnasium environment for SENTINEL.

Wires together NexaStackWorldState, Observability_Layer, Incident_Generator,
Reward_Function, and all agents into a single gym.Env-compatible interface.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import gymnasium
import gymnasium.spaces

from sentinel.cascade_engine import Cascade_Engine
from sentinel.config import load_config
from sentinel.exceptions import IncidentLibraryError
from sentinel.incident_generator import Incident_Generator
from sentinel.models import (
    Action,
    IncidentState,
    RewardWeights,
)
from sentinel.observability import Observability_Layer
from sentinel.reward import Reward_Function
from sentinel.world_state import ALL_SERVICES, NexaStackWorldState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent role constraints
# ---------------------------------------------------------------------------

_AGENT_ALLOWED_CATEGORIES: dict[str, list[str]] = {
    "argus":  ["investigative", "meta"],
    "holmes": ["investigative"],
    "forge":  ["remediation"],
    "hermes": ["deployment", "meta"],
    "oracle": ["meta"],
}


class Sentinel_Env(gymnasium.Env):
    """Multi-agent incident response RL environment for NexaStack."""

    metadata = {"render_modes": ["human", "json"]}

    def __init__(
        self,
        config_path: str = "env_spec.yaml",
        incident_library_path: str = "incident_library.yaml",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode

        # Load config (falls back to defaults with warning if absent)
        self._config = load_config(config_path)

        # Incident generator — raises IncidentLibraryError if library is missing/malformed
        self.incident_generator = Incident_Generator(incident_library_path)

        # World state and supporting components
        self.world_state = NexaStackWorldState()
        self._cascade_engine = Cascade_Engine()

        obs_cfg = self._config.observability
        self.observability_layer = Observability_Layer(obs_cfg)

        rw_cfg = self._config.reward.weights
        weights = RewardWeights(
            r1_root_cause=rw_cfg.r1_root_cause,
            r2_mttr=rw_cfg.r2_mttr,
            r3_recovery_quality=rw_cfg.r3_recovery_quality,
            r4_blast_radius=rw_cfg.r4_blast_radius,
        )
        self.reward_function = Reward_Function(
            weights=weights,
            sla_breach_threshold=self._config.training.sla_breach_threshold_steps,
        )

        self._max_steps: int = self._config.training.max_steps_per_episode
        self._difficulty_distribution: dict[str, float] = {
            "easy":   self._config.incident.difficulty_distribution.easy,
            "medium": self._config.incident.difficulty_distribution.medium,
            "hard":   self._config.incident.difficulty_distribution.hard,
        }

        # Episode state
        self.current_episode: dict | None = None
        self._incident_state: IncidentState | None = None
        self.step_count: int = 0
        self._needs_reset: bool = True
        self._episode_id: str = ""

        # Define spaces
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    # ------------------------------------------------------------------
    # Space definitions
    # ------------------------------------------------------------------

    def _build_observation_space(self) -> gymnasium.spaces.Dict:
        import numpy as np

        n = len(ALL_SERVICES)
        return gymnasium.spaces.Dict({
            # Flat JSON string for the metrics snapshot dict
            "metrics_snapshot": gymnasium.spaces.Text(min_length=0, max_length=65536),
            # 30×30 adjacency matrix flattened
            "causal_graph_snapshot": gymnasium.spaces.Box(
                low=0.0, high=1.0, shape=(n * n,), dtype=np.float32
            ),
            # JSON-encoded list of alerts
            "active_alerts": gymnasium.spaces.Text(min_length=0, max_length=65536),
            # JSON-encoded list of log entries
            "recent_logs": gymnasium.spaces.Text(min_length=0, max_length=65536),
            # JSON-encoded list of traces
            "active_traces": gymnasium.spaces.Text(min_length=0, max_length=65536),
            # JSON-encoded incident context dict
            "incident_context": gymnasium.spaces.Text(min_length=0, max_length=65536),
            # JSON-encoded sla_state dict
            "sla_state": gymnasium.spaces.Text(min_length=0, max_length=65536),
        })

    def _build_action_space(self) -> gymnasium.spaces.Dict:
        return gymnasium.spaces.Dict({
            "agent":    gymnasium.spaces.Text(min_length=1, max_length=64),
            "category": gymnasium.spaces.Text(min_length=1, max_length=64),
            "name":     gymnasium.spaces.Text(min_length=1, max_length=128),
            # JSON-encoded params dict
            "params":   gymnasium.spaces.Text(min_length=0, max_length=65536),
        })

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """Reset the environment and inject a new incident.

        Returns:
            (observation, info) where info contains the incident_id.
        """
        super().reset(seed=seed)

        # 1. Restore world state to baseline
        self.world_state.restore_baseline()
        self.step_count = 0
        self._needs_reset = False
        self._episode_id = str(uuid.uuid4())

        # 2. Sample a new incident template
        template = self.incident_generator.sample(self._difficulty_distribution)

        # 3. Create IncidentState from template
        incident_state = IncidentState(
            template_id=template.id,
            root_cause_service=template.root_cause_service,
            failure_type=template.failure_type,
            ground_truth_signals=list(template.ground_truth_signals),
            red_herring_signals=list(template.red_herring_signals),
            affected_services={},
            peak_blast_radius=set(),
            current_blast_radius=set(),
            timeline=[],
            attempted_remediations=[],
            active_hypotheses=[],
            resolved=False,
            step_injected=0,
        )

        # 4. Inject failure via Cascade_Engine
        affected = self._cascade_engine.propagate_failure(
            world_state=self.world_state,
            root_service=template.root_cause_service,
            failure_type=template.failure_type,
            initial_severity=1.0,
        )
        blast_radius = set(affected.keys())
        incident_state.affected_services = affected
        incident_state.peak_blast_radius = set(blast_radius)
        incident_state.current_blast_radius = set(blast_radius)

        # Attach incident state to world state
        self.world_state.incident_state = incident_state
        self._incident_state = incident_state

        # 5. Sample episode params for Observability_Layer
        self.observability_layer.sample_episode_params()

        # 6. Build and return initial observation
        obs = self._build_obs()
        info = {"incident_id": template.id}
        return obs, info

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        """Apply an action and advance the environment by one step.

        Args:
            action: dict with keys agent, category, name, params.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # 1. Raise ResetNeeded if not reset yet
        if self._needs_reset:
            raise gymnasium.error.ResetNeeded(
                "Call reset() before step()."
            )

        # 2. Parse action dict into Action model
        if isinstance(action, dict):
            # params may be a JSON string (from action_space) or a dict
            params = action.get("params", {})
            if isinstance(params, str):
                try:
                    params = json.loads(params) if params else {}
                except json.JSONDecodeError:
                    params = {}
            try:
                parsed_action = Action(
                    agent=action["agent"],
                    category=action["category"],
                    name=action["name"],
                    params=params,
                )
            except Exception:
                obs = self._build_obs()
                return obs, -0.1, False, False, {"error": "invalid_action"}
        else:
            parsed_action = action

        # 3. Validate action
        # Role violation check
        allowed_categories = _AGENT_ALLOWED_CATEGORIES.get(parsed_action.agent, [])
        if parsed_action.category not in allowed_categories:
            obs = self._build_obs()
            return obs, -0.1, False, False, {"error": "role_violation"}

        # Unknown service check
        service_param = parsed_action.params.get("service", None)
        if service_param is not None and service_param not in self.world_state.services:
            obs = self._build_obs()
            return obs, -0.1, False, False, {"error": "unknown_service"}

        # 4. Apply action effects to world state
        self._apply_action(parsed_action)

        # 5. Compute step reward
        incident_state = self._incident_state
        reward = self.reward_function.compute_step_reward(
            action=parsed_action,
            world_state=self.world_state,
            incident_state=incident_state,
        )

        # 6. Increment step count
        self.step_count += 1

        # 7. Check termination
        terminated = (
            parsed_action.name == "CloseIncident"
            or self.step_count >= self._max_steps
        )
        truncated = False

        if terminated:
            self._needs_reset = True
            if incident_state is not None:
                incident_state.resolved = parsed_action.name == "CloseIncident"

        # 8. Build observation
        obs = self._build_obs()
        info: dict[str, Any] = {
            "step_count": self.step_count,
            "episode_id": self._episode_id,
        }
        if terminated:
            info["terminated_reason"] = (
                "CloseIncident" if parsed_action.name == "CloseIncident" else "max_steps"
            )

        return obs, float(reward), terminated, truncated, info

    def render(self) -> str | None:
        """Render the current environment state.

        Returns a human-readable string or JSON string depending on render_mode.
        """
        if self.render_mode is None:
            return None

        snapshot = self.world_state.snapshot()

        if self.render_mode == "json":
            return json.dumps(snapshot, indent=2)

        # human mode — compact text summary
        lines = [f"=== SENTINEL Episode {self._episode_id} | Step {self.step_count} ==="]
        if self._incident_state is not None:
            inc = self._incident_state
            lines.append(
                f"Incident: {inc.template_id} | Root: {inc.root_cause_service} "
                f"({inc.failure_type.value})"
            )
            lines.append(
                f"Blast radius: {len(inc.current_blast_radius)} services "
                f"(peak: {len(inc.peak_blast_radius)})"
            )
        degraded = [
            svc for svc, m in self.world_state.services.items() if not m.availability
        ]
        lines.append(f"Degraded services ({len(degraded)}): {', '.join(degraded) or 'none'}")
        return "\n".join(lines)

    def close(self) -> None:
        """Clean up environment resources."""
        self._needs_reset = True
        self._incident_state = None
        self.world_state.incident_state = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> dict:
        """Build a gymnasium-compatible observation dict."""
        import numpy as np

        raw = self.observability_layer.build_observation(
            world_state=self.world_state,
            incident_state=self._incident_state,
            hypothesis_tree=None,
        )

        # Flatten causal_graph_snapshot (list[list[float]]) to 1-D np.float32 array
        matrix = raw["causal_graph_snapshot"]
        flat_matrix = np.array(
            [cell for row in matrix for cell in row], dtype=np.float32
        )

        # Serialize complex fields to JSON strings for the Text spaces
        def _serialize(obj: Any) -> str:
            if hasattr(obj, "model_dump"):
                return json.dumps(obj.model_dump())
            if isinstance(obj, list):
                items = []
                for item in obj:
                    if hasattr(item, "model_dump"):
                        items.append(item.model_dump())
                    else:
                        items.append(item)
                return json.dumps(items)
            return json.dumps(obj)

        return {
            "metrics_snapshot": _serialize(raw["metrics_snapshot"]),
            "causal_graph_snapshot": flat_matrix,
            "active_alerts": _serialize(raw["active_alerts"]),
            "recent_logs": _serialize(raw["recent_logs"]),
            "active_traces": _serialize(raw["active_traces"]),
            "incident_context": _serialize(raw["incident_context"]),
            "sla_state": _serialize(raw["sla_state"]),
        }

    def _apply_action(self, action: Action) -> None:
        """Apply the action's effects to the world state."""
        name = action.name
        params = action.params
        incident_state = self._incident_state

        if name == "RestartService":
            service = params.get("service", "")
            if service and service in self.world_state.services:
                # Propagate recovery for the restarted service
                self._cascade_engine.propagate_recovery(
                    world_state=self.world_state,
                    resolved_service=service,
                )
                # Update blast radius
                if incident_state is not None:
                    new_br = self._cascade_engine.get_blast_radius()
                    incident_state.current_blast_radius = new_br

        elif name == "CloseIncident":
            # Record the close action in the timeline
            if incident_state is not None:
                from sentinel.models import TimelineEntry
                incident_state.timeline.append(
                    TimelineEntry(
                        step=self.step_count,
                        event_type="close_incident",
                        description=params.get("resolution_summary", "Incident closed"),
                        agent=action.agent,
                    )
                )

        # Record attempted remediations
        if incident_state is not None and action.category == "remediation":
            incident_state.attempted_remediations.append(action)
