"""SENTINEL validation dashboard for Hugging Face Spaces."""
from __future__ import annotations

import json
from typing import Any

from models import SentinelAction
from sentinel.config import load_config
from server.sentinel_environment import SentinelEnvironment

try:
    import gradio as gr
    _GRADIO_AVAILABLE = True
except ImportError:
    gr = None  # type: ignore[assignment]
    _GRADIO_AVAILABLE = False

_adapter: SentinelEnvironment | None = None
_last_observation: dict[str, Any] = {}
_last_step_result: dict[str, Any] = {}
_action_log: list[str] = []
_INCIDENT_IDS = ["E1", "E2", "E3", "M1", "M2", "M3", "M4", "H1", "H2", "H3"]

_PRESET_ACTIONS: dict[str, dict[str, Any]] = {
    "QueryLogs": {
        "agent": "holmes",
        "category": "investigative",
        "name": "QueryLogs",
        "params": {"service": "cart-service", "time_range": [0, 60]},
    },
    "QueryMetrics": {
        "agent": "argus",
        "category": "investigative",
        "name": "QueryMetrics",
        "params": {"service": "order-service", "metric_name": "cpu", "time_range": [0, 60]},
    },
    "RestartService": {
        "agent": "forge",
        "category": "remediation",
        "name": "RestartService",
        "params": {"service": "cart-service"},
    },
    "EscalateToHuman": {
        "agent": "oracle",
        "category": "meta",
        "name": "EscalateToHuman",
        "params": {"reason": "Validator smoke test escalation"},
    },
}


def _ensure_adapter() -> SentinelEnvironment:
    global _adapter
    if _adapter is None:
        _adapter = SentinelEnvironment()
    return _adapter


def _json_blob(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _append_log(message: str) -> None:
    _action_log.append(message)
    del _action_log[:-20]


def _health_summary() -> str:
    if _adapter is None:
        return (
            "### Runtime Status\n"
            "- Adapter: not initialized\n"
            "- API health: `/health` should return `initialized: false` until reset\n"
        )

    state = _adapter.state.model_dump()
    return (
        "### Runtime Status\n"
        f"- Episode id: `{state['episode_id']}`\n"
        f"- Step count: `{state['step_count']}`\n"
        f"- Incident id: `{state.get('incident_id')}`\n"
        f"- Terminated: `{state.get('terminated')}`\n"
        f"- Truncated: `{state.get('truncated')}`\n"
    )


def _service_health_html() -> str:
    if _adapter is None:
        return "<p>No environment initialized yet.</p>"

    env = _adapter._env
    incident_state = env.world_state.incident_state
    blast_radius = incident_state.current_blast_radius if incident_state is not None else set()

    cells: list[str] = []
    for service, metrics in env.world_state.services.items():
        bg = "#0f766e" if metrics.availability else "#b91c1c"
        border = "3px solid #f59e0b" if service in blast_radius else "1px solid #334155"
        cells.append(
            "<div style='"
            f"background:{bg};border:{border};border-radius:10px;padding:8px 10px;margin:4px;"
            "display:inline-block;width:158px;vertical-align:top;color:#fff;font:12px/1.4 monospace;'>"
            f"<b>{service}</b><br>"
            f"CPU {metrics.cpu * 100:.1f}%<br>"
            f"Err {metrics.error_rate * 100:.2f}%<br>"
            f"Latency {metrics.latency_ms:.0f} ms"
            "</div>"
        )

    return (
        "<div style='background:#0f172a;padding:12px;border-radius:12px;'>"
        + "".join(cells)
        + "</div>"
    )


def _render_snapshot() -> str:
    if _adapter is None:
        return "(no render available yet)"
    rendered = _adapter._env.render()
    return rendered or "(render returned no output)"


def _current_state_json() -> str:
    if _adapter is None:
        return _json_blob({"initialized": False})
    return _json_blob(_adapter.state.model_dump())


def _current_log_text() -> str:
    if not _action_log:
        return "(no actions yet)"
    return "\n".join(reversed(_action_log))


def _snapshot(status: str) -> tuple[str, str, str, str, str, str, str]:
    return (
        status,
        _health_summary(),
        _service_health_html(),
        _current_state_json(),
        _json_blob(_last_observation),
        _json_blob(_last_step_result),
        _current_log_text(),
    )


def _reset_env(seed: int, incident_id: str) -> tuple[str, str, str, str, str, str, str]:
    global _last_observation, _last_step_result

    adapter = _ensure_adapter()
    env = adapter._env
    original_templates = env.incident_generator._templates
    matching = [tpl for tpl in original_templates if tpl.id == incident_id]
    if matching:
        env.incident_generator._templates = matching

    try:
        observation = adapter.reset(seed=seed)
    finally:
        env.incident_generator._templates = original_templates

    _last_observation = observation.model_dump()
    _last_step_result = {
        "event": "reset",
        "seed": seed,
        "incident_id": adapter.state.incident_id,
        "done": observation.done,
    }
    _append_log(f"reset(seed={seed}, incident_id={adapter.state.incident_id})")
    return _snapshot(
        f"Reset succeeded. Episode `{adapter.state.episode_id}` is ready at step `0`."
    )


def _run_action(action_name: str) -> tuple[str, str, str, str, str, str, str]:
    global _last_observation, _last_step_result

    adapter = _ensure_adapter()
    payload = _PRESET_ACTIONS[action_name]
    action = SentinelAction(**payload)
    observation = adapter.step(action)
    _last_observation = observation.model_dump()
    _last_step_result = {
        "action": payload,
        "reward": observation.reward,
        "done": observation.done,
        "info": observation.info,
    }
    _append_log(
        f"{payload['agent']}::{payload['name']} -> reward={observation.reward:.3f}, done={observation.done}"
    )
    return _snapshot(
        f"Action `{payload['name']}` executed. Reward: `{observation.reward:.3f}`. Done: `{observation.done}`."
    )


def _run_custom_action(action_json: str) -> tuple[str, str, str, str, str, str, str]:
    global _last_observation, _last_step_result

    adapter = _ensure_adapter()
    try:
        payload = json.loads(action_json)
        action = SentinelAction(**payload)
    except Exception as exc:
        return _snapshot(f"Custom action parse failed: `{exc}`")

    observation = adapter.step(action)
    _last_observation = observation.model_dump()
    _last_step_result = {
        "action": payload,
        "reward": observation.reward,
        "done": observation.done,
        "info": observation.info,
    }
    _append_log(
        f"{payload['agent']}::{payload['name']} -> reward={observation.reward:.3f}, done={observation.done}"
    )
    return _snapshot(
        f"Custom action `{payload['name']}` executed. Reward: `{observation.reward:.3f}`. Done: `{observation.done}`."
    )


def _run_smoke_test(seed: int, incident_id: str) -> tuple[str, str, str, str, str, str, str]:
    global _last_observation, _last_step_result

    adapter = SentinelEnvironment()
    env = adapter._env
    original_templates = env.incident_generator._templates
    matching = [tpl for tpl in original_templates if tpl.id == incident_id]
    if matching:
        env.incident_generator._templates = matching

    try:
        observation = adapter.reset(seed=seed)
    finally:
        env.incident_generator._templates = original_templates

    query_logs = adapter.step(SentinelAction(**_PRESET_ACTIONS["QueryLogs"]))
    query_metrics = adapter.step(SentinelAction(**_PRESET_ACTIONS["QueryMetrics"]))

    required_keys = {
        "metrics_snapshot",
        "causal_graph_snapshot",
        "active_alerts",
        "recent_logs",
        "active_traces",
        "incident_context",
        "sla_state",
        "reward",
        "done",
        "info",
    }
    missing = sorted(required_keys.difference(query_metrics.model_dump().keys()))

    ok = not missing and adapter.state.step_count >= 2
    _adapter_state = adapter.state.model_dump()
    _last_observation = query_metrics.model_dump()
    _last_step_result = {
        "smoke_test": "pass" if ok else "fail",
        "reset_incident_id": observation.info.get("incident_id"),
        "query_logs_reward": query_logs.reward,
        "query_metrics_reward": query_metrics.reward,
        "state": _adapter_state,
        "missing_keys": missing,
    }
    _append_log(
        f"smoke_test(seed={seed}, incident={incident_id}) -> {'PASS' if ok else 'FAIL'}"
    )

    status = (
        "Smoke test PASSED. Reset, step, state, and observation schema are working."
        if ok
        else f"Smoke test FAILED. Missing keys: {missing}"
    )
    return _snapshot(status)


def build_dashboard() -> Any:
    if not _GRADIO_AVAILABLE:
        return None

    cfg = load_config()
    default_action = _json_blob(cfg.training.placeholder_action)

    with gr.Blocks(title="SENTINEL Validation Console", theme=gr.themes.Default()) as dashboard:
        gr.Markdown(
            """
# SENTINEL Validation Console

This Space is designed to prove the submission requirements directly:

1. the public Space boots correctly,
2. the OpenEnv adapter can `reset`, `step`, and expose `state`,
3. the environment returns structured observations,
4. validators can inspect the live API at `/health` and `/docs`.

Use `Reset Episode`, then a preset action or `Run Validator Smoke Test`.
"""
        )

        with gr.Row():
            with gr.Column(scale=2):
                status = gr.Markdown("Ready. Initialize the environment with `Reset Episode`.")
                runtime = gr.Markdown(_health_summary())
            with gr.Column(scale=1):
                gr.Markdown(
                    """
### Public Endpoints
- `/health`
- `/docs`
- `/dashboard`

### Recommended Check
Run the smoke test after reset. It validates `reset`, two `step` calls, and observation keys.
"""
                )

        with gr.Row():
            seed = gr.Number(value=cfg.demo.seed, precision=0, label="Seed")
            incident_id = gr.Dropdown(choices=_INCIDENT_IDS, value=_INCIDENT_IDS[0], label="Incident")
            reset_btn = gr.Button("Reset Episode", variant="primary")
            smoke_btn = gr.Button("Run Validator Smoke Test", variant="secondary")

        with gr.Row():
            query_logs_btn = gr.Button("Query Logs")
            query_metrics_btn = gr.Button("Query Metrics")
            restart_btn = gr.Button("Restart Service")
            escalate_btn = gr.Button("Escalate To Human")

        health_grid = gr.HTML(value=_service_health_html(), label="Service Health")

        with gr.Row():
            with gr.Column():
                custom_action = gr.Code(
                    value=default_action,
                    language="json",
                    label="Custom Action JSON",
                )
                custom_btn = gr.Button("Run Custom Action")
            with gr.Column():
                action_log = gr.Textbox(
                    value=_current_log_text(),
                    label="Action Log",
                    lines=12,
                    interactive=False,
                )

        with gr.Row():
            state_json = gr.Code(value=_current_state_json(), language="json", label="Adapter State")
            latest_step = gr.Code(value=_json_blob(_last_step_result), language="json", label="Latest Step Result")

        with gr.Row():
            observation_json = gr.Code(value=_json_blob(_last_observation), language="json", label="Latest Observation")

        render_output = gr.Textbox(
            value=_render_snapshot(),
            label="Environment Render Snapshot",
            lines=8,
            interactive=False,
        )

        outputs = [status, runtime, health_grid, state_json, observation_json, latest_step, action_log]

        reset_btn.click(fn=_reset_env, inputs=[seed, incident_id], outputs=outputs)
        smoke_btn.click(fn=_run_smoke_test, inputs=[seed, incident_id], outputs=outputs)
        query_logs_btn.click(fn=lambda: _run_action("QueryLogs"), outputs=outputs)
        query_metrics_btn.click(fn=lambda: _run_action("QueryMetrics"), outputs=outputs)
        restart_btn.click(fn=lambda: _run_action("RestartService"), outputs=outputs)
        escalate_btn.click(fn=lambda: _run_action("EscalateToHuman"), outputs=outputs)
        custom_btn.click(fn=_run_custom_action, inputs=[custom_action], outputs=outputs)

        for trigger in (
            reset_btn,
            smoke_btn,
            query_logs_btn,
            query_metrics_btn,
            restart_btn,
            escalate_btn,
            custom_btn,
        ):
            trigger.click(fn=_render_snapshot, outputs=[render_output])

    return dashboard


demo = build_dashboard()

if __name__ == "__main__" and demo is not None:
    demo.launch(share=False)
