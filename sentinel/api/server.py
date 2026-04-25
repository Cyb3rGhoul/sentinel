"""SENTINEL OpenEnv FastAPI server.

Exposes HTTP endpoints that proxy to Sentinel_Env:
  POST /reset   -> ResetResponse
  POST /step    -> StepResponse   (HTTP 422 on malformed action)
  GET  /render  -> RenderResponse
  POST /close   -> CloseResponse
  GET  /health  -> HealthResponse

HTTP 503 is returned when Sentinel_Env has not been initialized yet (before
the first /reset call).  HTTP 500 with traceback is returned in debug mode
on unhandled exceptions.
"""
from __future__ import annotations

import os
import traceback
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sentinel.env import Sentinel_Env
from sentinel.models import Action

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(title="SENTINEL OpenEnv API")

# ---------------------------------------------------------------------------
# Shared environment state (module-level singleton)
# ---------------------------------------------------------------------------

_env: Sentinel_Env | None = None

DEBUG: bool = os.getenv("SENTINEL_DEBUG", "").lower() in ("1", "true", "yes")


def _get_env() -> Sentinel_Env:
    """Return the shared env instance, raising HTTP 503 if not yet created."""
    if _env is None:
        raise _EnvNotInitialized()
    return _env


class _EnvNotInitialized(Exception):
    """Raised when the env singleton has not been created yet."""


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(_EnvNotInitialized)
async def _handle_not_initialized(request: Request, exc: _EnvNotInitialized) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Sentinel_Env is not initialized. Call /reset first."},
    )


@app.exception_handler(Exception)
async def _handle_unhandled(request: Request, exc: Exception) -> JSONResponse:
    content: dict[str, Any] = {"detail": str(exc)}
    if DEBUG:
        content["traceback"] = traceback.format_exc()
    return JSONResponse(status_code=500, content=content)


# ---------------------------------------------------------------------------
# Request / Response models (Pydantic v2)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: int | None = None
    options: dict[str, Any] | None = None


class ResetResponse(BaseModel):
    observation: dict[str, Any]
    info: dict[str, Any]


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class RenderResponse(BaseModel):
    render: str | None


class CloseResponse(BaseModel):
    status: str = "closed"


class HealthResponse(BaseModel):
    status: str
    initialized: bool
    step_count: int | None = None
    episode_id: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=ResetResponse)
async def reset(body: ResetRequest = ResetRequest()) -> ResetResponse:
    """Reset the environment and start a new episode."""
    global _env

    # Create the env on first call if it doesn't exist yet
    if _env is None:
        _env = Sentinel_Env()

    obs, info = _env.reset(seed=body.seed, options=body.options)

    # obs contains numpy arrays; convert to plain Python for JSON serialisation
    serialisable_obs = _serialise_obs(obs)

    return ResetResponse(observation=serialisable_obs, info=info)


@app.post("/step", response_model=StepResponse)
async def step(body: StepRequest) -> StepResponse:
    """Apply an action and advance the environment by one step.

    FastAPI / Pydantic automatically returns HTTP 422 when the request body
    does not conform to StepRequest (e.g. missing fields, wrong enum values).
    """
    env = _get_env()

    obs, reward, terminated, truncated, info = env.step(body.action.model_dump())

    return StepResponse(
        observation=_serialise_obs(obs),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


@app.get("/render", response_model=RenderResponse)
async def render() -> RenderResponse:
    """Return a human-readable or JSON render of the current environment state."""
    env = _get_env()
    rendered = env.render()
    return RenderResponse(render=rendered)


@app.post("/close", response_model=CloseResponse)
async def close() -> CloseResponse:
    """Close the environment and release resources."""
    global _env
    env = _get_env()
    env.close()
    _env = None
    return CloseResponse(status="closed")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return server health and environment initialisation status."""
    if _env is None:
        return HealthResponse(status="ok", initialized=False)

    return HealthResponse(
        status="ok",
        initialized=True,
        step_count=_env.step_count,
        episode_id=_env._episode_id or None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialise_obs(obs: dict) -> dict[str, Any]:
    """Convert a gymnasium observation dict to a JSON-serialisable dict.

    The observation may contain numpy arrays (causal_graph_snapshot) and
    JSON strings for the Text-space fields.  We normalise everything to
    plain Python objects so FastAPI can serialise the response.
    """
    import json

    result: dict[str, Any] = {}
    for key, value in obs.items():
        if hasattr(value, "tolist"):
            # numpy array → Python list
            result[key] = value.tolist()
        elif isinstance(value, str):
            # Text-space fields are already JSON strings; try to decode them
            # so the response body is a proper nested object rather than a
            # double-encoded string.
            try:
                result[key] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                result[key] = value
        else:
            result[key] = value
    return result
