"""SENTINEL Gradio Dashboard.

Visualizes NexaStack health, agent actions, and training progress.
Compatible with HuggingFace Spaces (demo.launch() at module level).
"""
from __future__ import annotations

import json
import time
from typing import Any

from sentinel.config import load_config
from sentinel.training.pipeline import get_placeholder_action

try:
    import gradio as gr
    _GRADIO_AVAILABLE = True
except ImportError:
    gr = None  # type: ignore[assignment]
    _GRADIO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_action_log: list[dict] = []          # last 20 actions
_metrics_log: list[dict] = []         # last 50 episode metrics
_oracle_gap: str = "No episodes completed yet."
_env: Any = None  # Sentinel_Env | None

# Incident IDs from the library
_INCIDENT_IDS = ["E1", "E2", "E3", "M1", "M2", "M3", "M4", "H1", "H2", "H3"]

# ---------------------------------------------------------------------------
# Helper: seed demo state
# ---------------------------------------------------------------------------

def _seed_demo_state(env: Any) -> None:
    """Run 5 steps with seed=42 to populate demo state."""
    cfg = load_config()
    seed = cfg.demo.seed
    action = get_placeholder_action()
    env.reset(seed=seed)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(action)
        _action_log.append({
            "timestamp": time.time(),
            "agent": action.get("agent", "holmes"),
            "name": action.get("name", "QueryLogs"),
            "params": action.get("params", {}),
        })
        if terminated:
            break


def _create_demo_env() -> Any:
    """Create and seed a demo Sentinel_Env, returning None on failure."""
    try:
        from sentinel.env import Sentinel_Env
        env = Sentinel_Env()
        _seed_demo_state(env)
        templates = getattr(env.incident_generator, "_templates", [])
        if templates:
            global _INCIDENT_IDS
            _INCIDENT_IDS = [tpl.id for tpl in templates]
        return env
    except Exception as e:
        print(f"Warning: Could not create demo env: {e}")
        return None


# ---------------------------------------------------------------------------
# Dashboard component builders
# ---------------------------------------------------------------------------

def build_health_grid(env: Any) -> str:
    """Return an HTML string with a 30-service color-coded grid."""
    if env is None:
        return "<p style='color:gray'>No environment available.</p>"

    blast_radius: set[str] = set()
    if env.world_state.incident_state is not None:
        blast_radius = env.world_state.incident_state.current_blast_radius

    cells = []
    for svc, metrics in env.world_state.services.items():
        bg = "#22c55e" if metrics.availability else "#ef4444"
        border = "3px solid #f97316" if svc in blast_radius else "1px solid #374151"
        cpu_pct = round(metrics.cpu * 100, 1)
        err_pct = round(metrics.error_rate * 100, 2)
        cell = (
            f"<div style='"
            f"background:{bg};border:{border};border-radius:6px;"
            f"padding:6px 4px;margin:3px;display:inline-block;"
            f"width:140px;vertical-align:top;font-size:11px;color:#fff;"
            f"font-family:monospace;'>"
            f"<b>{svc}</b><br>"
            f"CPU {cpu_pct}% | Err {err_pct}%"
            f"</div>"
        )
        cells.append(cell)

    return (
        "<div style='background:#111827;padding:10px;border-radius:8px;"
        "line-height:1.6;'>"
        + "".join(cells)
        + "</div>"
    )


def build_action_feed(action_log: list[dict]) -> str:
    """Return last 20 entries from action_log as a formatted string (newest first)."""
    recent = action_log[-20:][::-1]
    if not recent:
        return "(no actions yet)"
    lines = []
    for entry in recent:
        ts = time.strftime("%H:%M:%S", time.localtime(entry.get("timestamp", 0)))
        agent = entry.get("agent", "?").upper()
        name = entry.get("name", "?")
        params = entry.get("params", {})
        # Summarise params: show first key=value pair only
        if params:
            first_key = next(iter(params))
            params_summary = f"{first_key}={params[first_key]}"
        else:
            params_summary = ""
        lines.append(f"[{ts}] {agent}: {name} ({params_summary})")
    return "\n".join(lines)


def get_training_data(metrics_log: list[dict]) -> Any:
    """Return last 50 entries as a pandas DataFrame for LinePlot."""
    try:
        import pandas as pd
    except ImportError:
        return None

    recent = metrics_log[-50:]
    if not recent:
        return pd.DataFrame(columns=["episode", "total_reward", "mttr", "r1", "r2", "r3", "r4"])

    rows = []
    for i, m in enumerate(recent):
        rows.append({
            "episode": m.get("episode", i),
            "total_reward": m.get("total_reward", 0.0),
            "mttr": m.get("mttr", 0.0),
            "r1": m.get("r1", 0.0),
            "r2": m.get("r2", 0.0),
            "r3": m.get("r3", 0.0),
            "r4": m.get("r4", 0.0),
        })
    return pd.DataFrame(rows)


def build_oracle_display(oracle_gap: str) -> str:
    """Return formatted HTML for the ORACLE capability gap."""
    return (
        "<div style='background:#1e293b;border-radius:8px;padding:12px;"
        "font-family:monospace;color:#e2e8f0;font-size:12px;'>"
        "<b style='color:#a78bfa'>ORACLE Capability Gap</b><br><br>"
        f"{oracle_gap}"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Inject incident handler
# ---------------------------------------------------------------------------

def inject_incident(incident_id: str) -> str:
    """Inject a named incident into the environment."""
    global _env
    if _env is None:
        return "No environment available."
    try:
        # Filter the incident generator to only sample the chosen incident
        original_templates = _env.incident_generator._templates
        matching = [inc for inc in original_templates if inc.id == incident_id]
        if not matching:
            return f"Incident '{incident_id}' not found in library."

        _env.incident_generator._templates = matching
        cfg = load_config()
        _env.reset(seed=cfg.demo.seed)
        _env.incident_generator._templates = original_templates

        blast = _env.world_state.incident_state.current_blast_radius if _env.world_state.incident_state else set()
        return (
            f"Injected incident {incident_id}. "
            f"Blast radius: {len(blast)} services ({', '.join(sorted(blast)[:5])}{'...' if len(blast) > 5 else ''})"
        )
    except Exception as e:
        return f"Error injecting incident: {e}"


# ---------------------------------------------------------------------------
# Refresh callback
# ---------------------------------------------------------------------------

def _refresh() -> tuple[str, str, str, Any]:
    """Return updated dashboard data for all auto-refresh components."""
    health_html = build_health_grid(_env)
    feed_text = build_action_feed(_action_log)
    oracle_html = build_oracle_display(_oracle_gap)
    df = get_training_data(_metrics_log)
    return health_html, feed_text, oracle_html, df


# ---------------------------------------------------------------------------
# build_dashboard
# ---------------------------------------------------------------------------

def build_dashboard(env: Any = None) -> Any:
    """Build and return the Gradio dashboard.

    If env is None, creates a pre-seeded Sentinel_Env for demo purposes.
    Returns None if Gradio is not installed.
    """
    if not _GRADIO_AVAILABLE:
        return None

    global _env
    if env is not None:
        _env = env
    elif _env is None:
        _env = _create_demo_env()

    with gr.Blocks(title="SENTINEL — Multi-Agent Incident Response") as dashboard:
        gr.Markdown("# SENTINEL — Multi-Agent Incident Response")

        # Row 1: Health grid
        with gr.Row():
            health_grid = gr.HTML(
                value=build_health_grid(_env),
                label="NexaStack Service Health",
            )

        # Row 2: Action feed | ORACLE gap
        with gr.Row():
            with gr.Column():
                action_feed = gr.Textbox(
                    value=build_action_feed(_action_log),
                    label="Agent Action Feed (last 20)",
                    lines=12,
                    interactive=False,
                )
            with gr.Column():
                oracle_display = gr.HTML(
                    value=build_oracle_display(_oracle_gap),
                    label="ORACLE Capability Gap",
                )

        # Row 3: Training progress
        with gr.Row():
            try:
                import pandas as pd
                training_plot = gr.LinePlot(
                    value=get_training_data(_metrics_log),
                    x="episode",
                    y="total_reward",
                    label="Training Progress (Episode Reward)",
                )
            except Exception:
                training_plot = gr.Dataframe(
                    value=get_training_data(_metrics_log),
                    label="Training Metrics",
                )

        # Row 4: Inject incident
        with gr.Row():
            incident_dropdown = gr.Dropdown(
                choices=_INCIDENT_IDS,
                value=_INCIDENT_IDS[0],
                label="Incident ID",
            )
            inject_btn = gr.Button("Inject Incident")
            inject_status = gr.Textbox(
                value="",
                label="Status",
                interactive=False,
            )

        inject_btn.click(
            fn=inject_incident,
            inputs=[incident_dropdown],
            outputs=[inject_status],
        )

        # Auto-refresh every 2 seconds
        try:
            timer = gr.Timer(value=2.0)
            timer.tick(
                fn=_refresh,
                outputs=[health_grid, action_feed, oracle_display, training_plot],
            )
        except Exception:
            # Fallback: use every= parameter if gr.Timer is unavailable
            health_grid.change(fn=_refresh, outputs=[health_grid, action_feed, oracle_display, training_plot])

    return dashboard


# ---------------------------------------------------------------------------
# Module-level dashboard object (import-safe)
# ---------------------------------------------------------------------------

_env = _create_demo_env()
demo = build_dashboard(_env)

if __name__ == "__main__" and demo is not None:
    demo.launch(share=False)
