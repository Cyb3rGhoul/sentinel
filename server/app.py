"""FastAPI entrypoint for the OpenEnv validator and Hugging Face Space runtime."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from demo.app import build_dashboard
from sentinel.api.server import app as api_app

_HAS_DASHBOARD = False


def _mount_dashboard(app: FastAPI) -> FastAPI:
    """Mount the Gradio dashboard under /dashboard when Gradio is available."""
    global _HAS_DASHBOARD
    dashboard = build_dashboard()
    if dashboard is None:
        return app

    try:
        import gradio as gr
    except ImportError:
        return app

    _HAS_DASHBOARD = True
    return gr.mount_gradio_app(app, dashboard, path="/dashboard")


app = _mount_dashboard(api_app)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Route Space visitors to the public dashboard."""
    return RedirectResponse(url="/dashboard" if _HAS_DASHBOARD else "/docs")
