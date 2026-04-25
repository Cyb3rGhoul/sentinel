"""Prompt builder for SENTINEL LLM agent.

Converts a raw Gymnasium observation dict into a compact decision prompt with:
  [SYSTEM] agent role + allowed action schema
  [USER] observation snapshot + operating constraints
  [ASSISTANT PREFILL] opening JSON fence to force structured output
"""
from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Action schema (shown to model every step)
# ---------------------------------------------------------------------------

_ACTION_SCHEMA = """\
Output a single JSON action with this exact schema:
{
  "agent":    "<holmes|forge|argus|hermes|oracle>",
  "category": "<investigative|remediation|deployment|meta>",
  "name":     "<one of the valid action names below>",
  "params":   { ... }
}

INVESTIGATIVE actions (agent=holmes or argus):
  QueryLogs        params: {service, time_range:[start,end]}
  QueryMetrics     params: {service, metric_name, time_range:[start,end]}
  QueryTrace       params: {trace_id}
  FormHypothesis   params: {service, failure_type, confidence:0-1}

REMEDIATION actions (agent=forge):
  RestartService     params: {service}
  ScaleService       params: {service, replicas}
  RollbackDeployment params: {service, version}
  DrainTraffic       params: {service}
  ModifyRateLimit    params: {service, limit_rps}
  ModifyConfig       params: {service, key, value}

DEPLOYMENT actions (agent=hermes):
  CanaryDeploy       params: {service, version, traffic_percent}
  FullDeploy         params: {service, version}
  Rollback           params: {service}

META actions (agent=oracle or hermes):
  CloseIncident      params: {resolution_summary}
  EscalateToHuman    params: {reason}
  GenerateNewScenario params: {difficulty, target_gap}

FAILURE TYPES: cpu_spike | memory_leak | bad_deployment |
               connection_pool_exhaustion | cache_miss_storm | network_partition
"""

# ---------------------------------------------------------------------------
# System prompt per agent role
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS: dict[str, str] = {
    "holmes": """\
You are HOLMES, the Root Cause Analysis detective for NexaStack — a 30-service \
microservice platform. Your mission: investigate cascading failures, reason about \
causal dependencies, and form a high-confidence hypothesis about the root cause \
before handing off to FORGE for remediation.

NexaStack layers (4 tiers, 30 services):
  Frontend:       web-gateway, mobile-api, cdn-edge
  Application:    cart-service, order-service, product-catalog, search-service,
                  recommendation-engine, user-auth, notification-service,
                  pricing-engine, inventory-service, review-service,
                  wishlist-service, session-manager
  Data:           postgres-primary, postgres-replica, redis-cache, elasticsearch,
                  kafka-broker, object-storage, analytics-db, audit-log
  Infrastructure: service-mesh, load-balancer, api-gateway, config-service,
                  secret-manager, payment-vault, fraud-detector

Decision policy:
  1. Investigate the most suspicious affected service first
  2. Use QueryTrace only after logs/metrics leave ambiguity
  3. FormHypothesis only when the observation supports a specific service/failure pair
  4. Stay within the allowed action schema
""",

    "forge": """\
You are FORGE, the Remediation Engineer for NexaStack. HOLMES has investigated \
and will send you a hypothesis. Your mission: apply the correct remediation action \
to the root-cause service to shrink the blast radius and restore availability.

Remediation priority:
  1. ScaleService — for saturation / connection pool exhaustion
  2. RollbackDeployment — for bad_deployment failures
  3. DrainTraffic — for overloaded or cascading services
  4. ModifyRateLimit — for cache_miss_storm / traffic spikes
  5. RestartService — ONLY for genuinely unavailable services (availability=False)
  6. CloseIncident — when blast_radius == 0

NEVER RestartService on a healthy service.
Prefer blast-radius reduction over aggressive intervention.
""",

    "argus": """\
You are ARGUS, the monitoring agent for NexaStack. Your role is to surface \
anomalies and alert patterns that help HOLMES identify root causes quickly.
Focus on services with error_rate > 0.05 or latency_ms > 300.
""",

    "oracle": """\
You are ORACLE, the incident-closure and escalation agent.
Close only when the blast radius is empty and services look recovered.
Escalate when the team has low-confidence diagnosis after sustained investigation.
""",
}


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_prompt(
    obs: dict[str, Any],
    agent_role: str = "holmes",
    step_number: int = 0,
) -> str:
    """Convert a Gymnasium observation dict into a chat-formatted text prompt.

    Args:
        obs: Raw observation dict from Sentinel_Env.step() / reset().
        agent_role: Which agent is acting ("holmes", "forge", "argus", "oracle").
        step_number: Current episode step (injected into prompt for context).

    Returns:
        Full prompt string ready for tokenizer.apply_chat_template().
    """
    system = _SYSTEM_PROMPTS.get(agent_role, _SYSTEM_PROMPTS["holmes"])

    # Build the user observation block
    user_block = _format_observation(obs, step_number)

    # Assemble in Llama-3 / Qwen2.5 chat format
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}\n\n"
        f"{_ACTION_SCHEMA}"
        f"<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_block}"
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"```json\n{{"
    )
    return prompt


def build_messages(
    obs: dict[str, Any],
    agent_role: str = "holmes",
    step_number: int = 0,
) -> list[dict[str, str]]:
    """Return messages list for tokenizer.apply_chat_template(messages, ...).

    The assistant turn is pre-filled with '```json\n{' which forces
    Llama-3 / Qwen2.5 to complete a JSON block rather than prose.
    """
    system = _SYSTEM_PROMPTS.get(agent_role, _SYSTEM_PROMPTS["holmes"])
    user_block = _format_observation(obs, step_number)
    return [
        {"role": "system",    "content": f"{system}\n\n{_ACTION_SCHEMA}"},
        {"role": "user",      "content": user_block},
        # Pre-fill assistant with opening JSON fence → forces structured output
        {"role": "assistant", "content": "```json\n{"},
    ]


# ---------------------------------------------------------------------------
# Observation formatter
# ---------------------------------------------------------------------------

def _format_observation(obs: dict[str, Any], step_number: int) -> str:
    """Convert the obs dict to a concise, readable text block."""
    lines: list[str] = [f"=== NexaStack Incident — Step {step_number} ===\n"]

    # --- Incident context ---
    inc_raw = obs.get("incident_context", "{}")
    inc = _parse_json_field(inc_raw)
    if inc:
        lines.append("## Incident Context")
        if "template_id" in inc:
            lines.append(f"  Template : {inc['template_id']}")
        br = inc.get("current_blast_radius", [])
        lines.append(f"  Blast radius ({len(br)} services): {', '.join(br[:8])}" +
                     (" ..." if len(br) > 8 else ""))
        hyps = inc.get("active_hypotheses", [])
        if hyps:
            lines.append("  Hypotheses:")
            for h in hyps[:3]:
                if isinstance(h, dict):
                    lines.append(f"    - {h.get('service','?')} "
                                 f"({h.get('failure_type','?')}) "
                                 f"conf={h.get('confidence',0):.2f}")
        lines.append("")

    # --- Active alerts (top 6) ---
    alerts_raw = obs.get("active_alerts", "[]")
    alerts = _parse_json_field(alerts_raw)
    if alerts:
        lines.append("## Active Alerts (top 6)")
        for a in alerts[:6]:
            if isinstance(a, dict):
                svc = a.get("service", "?")
                metric = a.get("metric", a.get("metric_name", "?"))
                val = a.get("value", a.get("current_value", "?"))
                lines.append(f"  [{a.get('severity','?')}] {svc} — {metric}={val}")
        lines.append("")

    # --- Degraded service metrics ---
    metrics_raw = obs.get("metrics_snapshot", obs.get("service_metrics", "{}"))
    metrics = _parse_json_field(metrics_raw)
    degraded = _find_degraded(metrics)
    if degraded:
        lines.append("## Degraded Services (error_rate > 0.05 or latency > 300ms)")
        for svc, m in degraded[:8]:
            avail = "DOWN" if not m.get("availability", True) else "UP"
            lines.append(
                f"  {svc:30s} [{avail}] "
                f"cpu={m.get('cpu',0):.2f} "
                f"err={m.get('error_rate',0):.3f} "
                f"lat={m.get('latency_ms',0):.0f}ms "
                f"sat={m.get('saturation',0):.2f}"
            )
        lines.append("")

    # --- Recent logs (top 5 errors) ---
    logs_raw = obs.get("recent_logs", "[]")
    logs = _parse_json_field(logs_raw)
    error_logs = [l for l in logs if isinstance(l, dict) and
                  l.get("level", "").upper() in ("ERROR", "WARN")][:5]
    if error_logs:
        lines.append("## Recent Error Logs")
        for l in error_logs:
            lines.append(f"  [{l.get('service','?')}] {l.get('message','')[:80]}")
        lines.append("")

    # --- SLA state ---
    sla_raw = obs.get("sla_state", "{}")
    sla = _parse_json_field(sla_raw)
    if sla:
        service_state = sla.get("services", {})
        lines.append("## SLA State")
        lines.append(
            f"  Breached: {sla.get('breached', False)} | "
            f"MTTR so far: {sla.get('current_mttr', step_number)} steps | "
            f"Blast radius: {sla.get('blast_radius', 0)}"
        )
        if isinstance(service_state, dict):
            unhealthy = [svc for svc, healthy in service_state.items() if not healthy]
            if unhealthy:
                lines.append(f"  Unhealthy services: {', '.join(unhealthy[:8])}" +
                             (" ..." if len(unhealthy) > 8 else ""))
        lines.append("")

    lines.append(
        "Return ONLY a JSON code block. Do not explain your reasoning.\n"
        "Choose one action that is valid for the acting role and grounded in the observation.\n"
        "Example:\n"
        "```json\n"
        "{\"agent\": \"holmes\", \"category\": \"investigative\", "
        "\"name\": \"QueryLogs\", \"params\": {\"service\": \"postgres-primary\", "
        "\"time_range\": [0, 300]}}\n"
        "```"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_field(raw: Any) -> Any:
    """Parse a field that may be a JSON string or already a dict/list."""
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _find_degraded(metrics: Any) -> list[tuple[str, dict]]:
    """Return (service, metric_dict) pairs that look degraded, sorted by error_rate."""
    if not isinstance(metrics, dict):
        return []
    results = []
    for svc, m in metrics.items():
        if not isinstance(m, dict):
            continue
        err = m.get("error_rate", 0) or 0
        lat = m.get("latency_ms", 0) or 0
        avail = m.get("availability", True)
        if err > 0.05 or lat > 300 or not avail:
            results.append((svc, m))
    results.sort(key=lambda x: -(x[1].get("error_rate", 0) or 0))
    return results
