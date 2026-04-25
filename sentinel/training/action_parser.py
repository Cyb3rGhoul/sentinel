"""Action parser for SENTINEL LLM agent.

The model is instructed to emit a single JSON object and nothing else.
This module extracts the first valid action-shaped object it can find,
repairs minor formatting issues when possible, and otherwise falls back
to a role-safe default action.
"""
from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_VALID_AGENTS: frozenset[str] = frozenset(
    ["holmes", "forge", "argus", "hermes", "oracle"]
)

_VALID_CATEGORIES: frozenset[str] = frozenset(
    ["investigative", "remediation", "deployment", "meta"]
)

_INVESTIGATIVE_ACTIONS: frozenset[str] = frozenset([
    "QueryLogs", "QueryMetrics", "QueryTrace", "FormHypothesis",
])

_REMEDIATION_ACTIONS: frozenset[str] = frozenset([
    "RestartService", "ScaleService", "RollbackDeployment", "DrainTraffic",
    "ModifyRateLimit", "ModifyConfig",
])

_DEPLOYMENT_ACTIONS: frozenset[str] = frozenset([
    "CanaryDeploy", "FullDeploy", "Rollback",
])

_META_ACTIONS: frozenset[str] = frozenset([
    "CloseIncident", "EscalateToHuman", "GenerateNewScenario",
])

_ALL_ACTIONS: frozenset[str] = (
    _INVESTIGATIVE_ACTIONS | _REMEDIATION_ACTIONS | _DEPLOYMENT_ACTIONS | _META_ACTIONS
)

_CATEGORY_ACTIONS: dict[str, frozenset[str]] = {
    "investigative": _INVESTIGATIVE_ACTIONS,
    "remediation": _REMEDIATION_ACTIONS,
    "deployment": _DEPLOYMENT_ACTIONS,
    "meta": _META_ACTIONS,
}

_AGENT_CATEGORY: dict[str, str] = {
    "holmes": "investigative",
    "argus": "investigative",
    "forge": "remediation",
    "hermes": "deployment",
    "oracle": "meta",
}

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_JSON_PREFIX_RE = re.compile(r"^\s*json\s*[:\-]?\s*", re.IGNORECASE)


def extract_think(llm_output: str) -> str:
    match = _THINK_RE.search(llm_output)
    if not match:
        return ""
    inner = re.sub(r"^<think>|</think>$", "", match.group(0), flags=re.IGNORECASE)
    return inner.strip()


def parse_llm_action(
    llm_output: str,
    fallback_agent: str = "holmes",
) -> dict[str, Any]:
    action, _ = parse_llm_action_result(llm_output, fallback_agent=fallback_agent)
    return action


def parse_llm_action_result(
    llm_output: str,
    fallback_agent: str = "holmes",
) -> tuple[dict[str, Any], bool]:
    """Return (action_dict, parsed_ok)."""
    cleaned = _normalize_output(_THINK_RE.sub("", llm_output).strip())

    action = _extract_from_code_block(cleaned)
    if action is None:
        action = _extract_raw_json(cleaned)
    if action is None:
        action = _extract_by_keyword(cleaned)
    if action is None:
        action = _extract_by_brace_walk(cleaned)
    if action is None:
        action = _extract_by_repair(cleaned)

    if action is None:
        snippet = re.sub(r"\s+", " ", llm_output).strip()[:160]
        logger.warning(
            "parse_llm_action: could not extract JSON from output (len=%d, snippet=%r). Falling back to safe action.",
            len(llm_output),
            snippet,
        )
        return _safe_fallback(fallback_agent), False

    return _validate_and_repair(action, fallback_agent), True


def _extract_from_code_block(text: str) -> dict[str, Any] | None:
    match = _CODE_BLOCK_RE.search(text)
    if not match:
        return None
    return _try_parse(_normalize_output(match.group(1).strip()))


def _extract_raw_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start == -1:
        return None
    return _extract_balanced_object(text, start)


def _extract_by_keyword(text: str) -> dict[str, Any] | None:
    idx = text.find('"agent"')
    if idx == -1:
        idx = text.find("'agent'")
    if idx == -1:
        return None
    start = text.rfind("{", 0, idx)
    if start == -1:
        return None
    return _extract_balanced_object(text, start)


def _extract_by_brace_walk(text: str) -> dict[str, Any] | None:
    for i, ch in enumerate(text):
        if ch == "{":
            candidate = _extract_balanced_object(text, i)
            if candidate and "agent" in candidate and "name" in candidate:
                return candidate
    return None


def _extract_by_repair(text: str) -> dict[str, Any] | None:
    candidates: list[str] = []

    start = text.find("{")
    if start != -1:
        fragment = text[start:].rstrip().rstrip("` \n\r\t")
        candidates.append(fragment)

    jsonish = _jsonish_fragment(text)
    if jsonish:
        candidates.append(jsonish)

    for fragment in candidates:
        for extra in ("", "}", "}}", "}}}"):
            candidate = _try_parse(fragment + extra)
            if candidate and "agent" in candidate and "name" in candidate:
                return candidate
    return None


def _jsonish_fragment(text: str) -> str | None:
    stripped = _normalize_output(text)
    if not stripped:
        return None
    if "agent" not in stripped or "name" not in stripped:
        return None
    if "{" not in stripped:
        stripped = "{" + stripped
    return stripped.rstrip(", \n\r\t")


def _normalize_output(text: str) -> str:
    text = text.strip()
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = _JSON_PREFIX_RE.sub("", text)
    if text.lower().startswith("assistant:"):
        text = text.split(":", 1)[1].strip()
    return text


def _try_python_dict(raw: str) -> dict[str, Any] | None:
    try:
        obj = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _repair_json_like(raw: str) -> str:
    repaired = raw.strip().rstrip("` \n\r\t,")
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
    if repaired.startswith('"agent"') or repaired.startswith("'agent'"):
        repaired = "{" + repaired
    if repaired.startswith("agent"):
        repaired = '{"' + repaired[5:]
    if repaired.count("{") > repaired.count("}"):
        repaired += "}" * (repaired.count("{") - repaired.count("}"))
    return repaired


def _coerce_dict(obj: dict[str, Any]) -> dict[str, Any]:
    if "params" not in obj:
        obj["params"] = {}
    return obj


def _try_parse_loose(raw: str) -> dict[str, Any] | None:
    repaired = _repair_json_like(raw)
    candidate = _try_python_dict(repaired)
    if candidate is not None:
        return _coerce_dict(candidate)

    candidate = _try_python_dict(repaired.replace("null", "None").replace("true", "True").replace("false", "False"))
    if candidate is not None:
        return _coerce_dict(candidate)

    try:
        obj = json.loads(repaired.replace("'", '"'))
        if isinstance(obj, dict):
            return _coerce_dict(obj)
    except (json.JSONDecodeError, ValueError):
        return None
    return None


def _extract_balanced_object(text: str, start: int) -> dict[str, Any] | None:
    depth = 0
    in_string = False
    string_quote = ""
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch in ('"', "'"):
            if not in_string:
                in_string = True
                string_quote = ch
                continue
            if ch == string_quote:
                in_string = False
                string_quote = ""
                continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return _try_parse(text[start : i + 1])
    return None


def _try_parse(raw: str) -> dict[str, Any] | None:
    raw = _normalize_output(raw)
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return _try_parse_loose(raw)


def _validate_and_repair(action: dict[str, Any], fallback_agent: str) -> dict[str, Any]:
    agent = action.get("agent", fallback_agent)
    if agent not in _VALID_AGENTS:
        agent = fallback_agent

    name = action.get("name", "")
    if name not in _ALL_ACTIONS:
        lowered = str(name).lower()
        match = next((candidate for candidate in _ALL_ACTIONS if candidate.lower() == lowered), None)
        name = match or "QueryLogs"

    category = action.get("category", "")
    if category not in _VALID_CATEGORIES:
        category = _infer_category(name)

    if name not in _CATEGORY_ACTIONS.get(category, frozenset()):
        category = _infer_category(name)

    params = action.get("params", {})
    if not isinstance(params, dict):
        params = {}

    return {
        "agent": agent,
        "category": category,
        "name": name,
        "params": params,
    }


def _infer_category(name: str) -> str:
    if name in _INVESTIGATIVE_ACTIONS:
        return "investigative"
    if name in _REMEDIATION_ACTIONS:
        return "remediation"
    if name in _DEPLOYMENT_ACTIONS:
        return "deployment"
    return "meta"


def _safe_fallback(agent: str) -> dict[str, Any]:
    if agent == "forge":
        return {
            "agent": "forge",
            "category": "remediation",
            "name": "ScaleService",
            "params": {"service": "api-gateway", "replicas": 2},
        }
    if agent == "hermes":
        return {
            "agent": "hermes",
            "category": "deployment",
            "name": "Rollback",
            "params": {"service": "api-gateway"},
        }
    if agent == "oracle":
        return {
            "agent": "oracle",
            "category": "meta",
            "name": "EscalateToHuman",
            "params": {"reason": "LLM parse failure"},
        }
    return {
        "agent": "holmes",
        "category": "investigative",
        "name": "QueryLogs",
        "params": {"service": "api-gateway", "time_range": [0, 300]},
    }
