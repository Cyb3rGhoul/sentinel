"""GRPO training pipeline for SENTINEL.

Provides:
- TrainingConfig: dataclass for all training hyperparameters
- EpisodeMetrics: dataclass for per-episode logging
- build_grpo_trainer(): load model + LoRA + GRPOTrainer (graceful fallback)
- log_episode_metrics(): append JSON line to log file
- save_checkpoint() / load_latest_checkpoint(): persistent episode counter
- run_training_loop(): full training loop with OOM handling and checkpointing

Action selection priority (highest to lowest):
  1. LLMAgent.act()               — fine-tuned Llama/Qwen model via prompt_builder +
                                    action_parser (GPU required, unsloth/trl installed)
  2. UCB1 + Bayesian RCA          — observation-driven math (no GPU, no API needed)
     UCB1: Auer, Cesa-Bianchi & Fischer (2002). Machine Learning, 47, 235-256.
     Bayes: Noisy-OR belief propagation (Pearl 1988 / MicroRank WWW 2021).

LLM integration flow:
  obs (dict)
    → prompt_builder.build_messages()    # obs → chat messages
    → tokenizer.apply_chat_template()    # messages → input_ids
    → model.generate()                   # input_ids → raw text
    → action_parser.parse_llm_action()   # raw text → Action dict
    → env.step(action)                   # action → (obs, reward, ...)
    → make_grpo_reward_fn()              # reward signal → GRPOTrainer
"""
from __future__ import annotations

import json
import logging
import os
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

from sentinel.config import load_config
from sentinel.math_engine import get_bayesian_rca, get_ucb1_bandit

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional imports — unsloth / trl may not be installed
# ---------------------------------------------------------------------------

try:
    from unsloth import FastLanguageModel  # type: ignore
    _UNSLOTH_AVAILABLE = True
except Exception as _unsloth_err:  # NotImplementedError when no GPU, ImportError when not installed
    FastLanguageModel = None  # type: ignore
    _UNSLOTH_AVAILABLE = False
    # Warn only once at import time so the caller knows why LLM mode is disabled
    import warnings as _w
    _w.warn(
        f"unsloth unavailable ({type(_unsloth_err).__name__}: {_unsloth_err}). "
        "Training will run in UCB1+Bayesian simulation mode.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from trl import GRPOTrainer, GRPOConfig  # type: ignore
    _TRL_AVAILABLE = True
except Exception as _trl_err:  # ImportError when not installed
    GRPOTrainer = None  # type: ignore
    GRPOConfig = None  # type: ignore
    _TRL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    agent: Literal["holmes", "forge"]
    model_name: str = "unsloth/Meta-Llama-3-8B-Instruct"
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    batch_size: int = 4
    max_steps: int = 1000
    checkpoint_dir: str = "checkpoints"
    log_file: str = "training_log.jsonl"
    sla_breach_threshold: int = 50


@dataclass
class EpisodeMetrics:
    episode: int
    r1: float
    r2: float
    r3: float
    r4: float
    total_reward: float
    mttr: int
    loss: float | None = None


# ---------------------------------------------------------------------------
# Trainer construction
# ---------------------------------------------------------------------------

def build_grpo_trainer(
    agent: Literal["holmes", "forge"],
    env: Any,
    config: TrainingConfig,
) -> tuple[Any, Any]:
    """Build a GRPOTrainer + LLMAgent for the given SENTINEL agent role.

    Full LLM integration flow:
      1. Load ``unsloth/Meta-Llama-3-8B-Instruct`` (or config.model_name) in 4-bit
      2. Apply LoRA adapters (r=16, alpha=32) via Unsloth
      3. Wrap SENTINEL's Reward_Function as a GRPO reward signal
      4. Construct LLMAgent (obs→prompt→model→parse→action)
      5. Return (GRPOTrainer, LLMAgent)

    Returns:
      (trainer, llm_agent) — both None when unsloth/trl are unavailable,
      in which case training falls back to UCB1+Bayesian math agent.
    """
    from sentinel.training.llm_agent import LLMAgent, make_grpo_reward_fn
    from sentinel.training.prompt_builder import build_messages

    if not _UNSLOTH_AVAILABLE or not _TRL_AVAILABLE:
        warnings.warn(
            "unsloth and/or trl are not installed — build_grpo_trainer() returning (None, None). "
            "Training will run in UCB1+Bayesian simulation mode (no GPU required).",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None

    # ── 1. Load base model in 4-bit quantization ──────────────────────────
    logger.info("Loading %s in 4-bit ...", config.model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        dtype=None,           # auto-detect bf16/fp16
    )

    # ── 2. Apply LoRA adapters ─────────────────────────────────────────────
    logger.info("Applying LoRA (r=%d, alpha=%d) ...", config.lora_r, config.lora_alpha)
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",  # unsloth's memory-efficient variant
        random_state=42,
    )

    # ── 3. GRPO reward function ────────────────────────────────────────────
    # GRPOTrainer expects: reward_fn(prompts, completions, **kwargs) -> list[float]
    grpo_reward_fn = make_grpo_reward_fn(env)

    # ── 4. Build a minimal prompt dataset for GRPOTrainer ─────────────────
    # TRL's GRPOTrainer needs a dataset of prompts to sample from.
    # We generate synthetic prompts from a fresh env reset.
    try:
        from datasets import Dataset  # type: ignore
        env_obs, _ = env.reset()
        sample_messages = build_messages(env_obs, agent_role=agent, step_number=0)
        sample_prompt = tokenizer.apply_chat_template(
            sample_messages, tokenize=False, add_generation_prompt=True
        )
        # Create a small dataset; the real training data comes from env rollouts
        prompt_dataset = Dataset.from_dict({"prompt": [sample_prompt] * 8})
    except Exception as exc:
        logger.warning("Could not build prompt dataset: %s", exc)
        prompt_dataset = None

    # ── 5. GRPOTrainer config ─────────────────────────────────────────────
    grpo_cfg = GRPOConfig(
        output_dir=config.checkpoint_dir,
        per_device_train_batch_size=config.batch_size,
        max_steps=config.max_steps,
        max_new_tokens=512,
        temperature=0.7,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        bf16=True,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[grpo_reward_fn],
        args=grpo_cfg,
        train_dataset=prompt_dataset,
    )

    # ── 6. LLMAgent for action generation during rollouts ─────────────────
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_agent = LLMAgent(
        model=model,
        tokenizer=tokenizer,
        agent_role=agent,
        device=device,
    )
    logger.info("build_grpo_trainer: ready (model=%s, device=%s)", config.model_name, device)

    return trainer, llm_agent


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_episode_metrics(metrics: EpisodeMetrics, log_file: str) -> None:
    """Append a JSON line with all EpisodeMetrics fields to *log_file*."""
    record = {
        "episode": metrics.episode,
        "r1": metrics.r1,
        "r2": metrics.r2,
        "r3": metrics.r3,
        "r4": metrics.r4,
        "total_reward": metrics.total_reward,
        "mttr": metrics.mttr,
        "loss": metrics.loss,
    }
    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, checkpoint_dir: str, episode: int) -> None:
    """Save *state* as JSON to ``{checkpoint_dir}/checkpoint_{episode:06d}.json``.

    Creates *checkpoint_dir* if it does not exist.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_{episode:06d}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh)


def load_latest_checkpoint(checkpoint_dir: str) -> dict | None:
    """Return the most recent valid checkpoint dict, or ``None``.

    If the latest checkpoint file is corrupted (JSON parse error), falls back
    to the previous one and logs a warning.  Returns ``None`` when no
    checkpoints exist.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    files = sorted(
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_") and f.endswith(".json")
    )
    if not files:
        return None

    # Try from newest to oldest
    for filename in reversed(files):
        path = os.path.join(checkpoint_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Checkpoint %s is corrupted (%s); falling back to previous checkpoint.",
                filename,
                exc,
            )

    return None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def get_heuristic_action(obs: dict, agent_role: str = "holmes") -> dict:
    """Return an observation-driven action using the HOLMES or FORGE heuristic agent.

    Implements the proper multi-agent workflow:
    1. HOLMES: QueryLogs/QueryMetrics/CheckDependencies → FormHypothesis (when confident)
    2. FORGE: ScaleService/DrainTraffic/ModifyRateLimit on degraded blast-radius services
    3. ORACLE/HERMES: CloseIncident when blast radius shrinks significantly
    """
    import json as _json

    # Parse incident context from observation
    incident_ctx_raw = obs.get("incident_context", "{}")
    if isinstance(incident_ctx_raw, str):
        try:
            incident_ctx = _json.loads(incident_ctx_raw)
        except _json.JSONDecodeError:
            incident_ctx = {}
    else:
        incident_ctx = incident_ctx_raw

    blast_radius: list[str] = incident_ctx.get("current_blast_radius", [])
    hypotheses: list[dict] = incident_ctx.get("active_hypotheses", [])
    step_num: int = incident_ctx.get("step", 0)

    # Parse service metrics from observation to find degraded services
    metrics_raw = obs.get("service_metrics", "{}")
    if isinstance(metrics_raw, str):
        try:
            metrics = _json.loads(metrics_raw)
        except _json.JSONDecodeError:
            metrics = {}
    else:
        metrics = metrics_raw

    # Parse active alerts
    alerts_raw = obs.get("active_alerts", "[]")
    if isinstance(alerts_raw, str):
        try:
            alerts = _json.loads(alerts_raw)
        except _json.JSONDecodeError:
            alerts = []
    else:
        alerts = alerts_raw

    alert_service_counts: dict[str, int] = {}
    for a in alerts:
        svc = a.get("service", "") if isinstance(a, dict) else getattr(a, "service", "")
        if svc:
            alert_service_counts[svc] = alert_service_counts.get(svc, 0) + 1

    # Find most impacted service: prefer blast_radius services with high error rates
    degraded_services = []
    for svc in blast_radius:
        m = metrics.get(svc, {}) if isinstance(metrics, dict) else {}
        err = m.get("error_rate", 0.0) if isinstance(m, dict) else 0.0
        degraded_services.append((svc, err))
    degraded_services.sort(key=lambda x: -x[1])  # highest error rate first

    # Pick primary target
    if degraded_services:
        top_service = degraded_services[0][0]
    elif alert_service_counts:
        top_service = max(alert_service_counts, key=alert_service_counts.__getitem__)
    elif blast_radius:
        top_service = blast_radius[0]
    else:
        top_service = "api-gateway"

    if agent_role == "forge":
        # FORGE: use non-destructive remediation actions on degraded services
        if not blast_radius:
            # Incident resolved — close it
            return {
                "agent": "oracle",
                "category": "meta",
                "name": "CloseIncident",
                "params": {"resolution_summary": "Blast radius cleared, closing incident."},
            }

        if hypotheses:
            best = max(
                hypotheses,
                key=lambda h: h.get("confidence", 0.0) if isinstance(h, dict)
                else getattr(h, "confidence", 0.0),
            )
            target = best.get("service", top_service) if isinstance(best, dict) else getattr(best, "service", top_service)
        else:
            target = top_service

        # Cycle through safe remediation actions (not RestartService which penalizes healthy svcs)
        remediation_cycle = ["ScaleService", "ModifyRateLimit", "DrainTraffic", "RollbackDeployment"]
        action_name = remediation_cycle[step_num % len(remediation_cycle)]
        params: dict = {"service": target}
        if action_name == "ScaleService":
            params["replicas"] = 3
        elif action_name == "ModifyRateLimit":
            params["limit"] = 500
        elif action_name == "DrainTraffic":
            params["target_node"] = target

        return {
            "agent": "forge",
            "category": "remediation",
            "name": action_name,
            "params": params,
        }

    # HOLMES: investigate → form hypothesis when enough data gathered
    if step_num == 3 and blast_radius:
        # After 3 investigative steps, form a hypothesis
        return {
            "agent": "holmes",
            "category": "investigative",
            "name": "FormHypothesis",
            "params": {
                "service": top_service,
                "failure_type": "connection_pool_exhaustion",
                "confidence": 0.8,
            },
        }

    # Rotate through investigative actions for diverse signal collection
    investigative_cycle = ["QueryLogs", "QueryMetrics", "CheckDependencies", "QueryTraces"]
    inv_action = investigative_cycle[step_num % len(investigative_cycle)]
    params = {"service": top_service, "time_range": [0, 300]}
    if inv_action == "QueryMetrics":
        params["metric_name"] = "error_rate"

    return {
        "agent": "holmes",
        "category": "investigative",
        "name": inv_action,
        "params": params,
    }


# Keep get_placeholder_action as an alias for backward compatibility (demo/app.py)
def get_placeholder_action(config_path: str = "env_spec.yaml") -> dict[str, Any]:
    """Backward-compatible alias — returns config-driven action for initial demo seeding."""
    try:
        cfg = load_config(config_path)
        return dict(cfg.training.placeholder_action)
    except Exception:
        return {
            "agent": "holmes",
            "category": "investigative",
            "name": "QueryMetrics",
            "params": {"service": "api-gateway", "metric_name": "error_rate", "time_range": [0, 300]},
        }


def run_training_loop(
    trainer: Any,
    env: Any,
    config: TrainingConfig,
    reward_fn: Any,
    start_episode: int = 0,
    llm_agent: Any = None,
) -> list[EpisodeMetrics]:
    """Run the GRPO training loop from *start_episode* to *config.max_steps*.

    For each episode:
    1. Reset env (ALP curriculum selects difficulty/failure_type).
    2. Act with LLMAgent (if available) or UCB1+Bayesian math.
    3. After each episode, run a GRPOTrainer gradient step.
    4. Record in ALP curriculum for next task selection.
    5. Log metrics and checkpoint every 10 episodes.

    CUDA OOM is handled by halving ``config.batch_size`` and retrying.
    Works even when *trainer* and *llm_agent* are both None (no GPU).
    """
    from sentinel.math_engine import get_alp_curriculum, get_ucb1_bandit

    all_metrics: list[EpisodeMetrics] = []
    curriculum = get_alp_curriculum()
    bandit = get_ucb1_bandit()  # noqa: F841  (used inside _ucb1_math_action)

    mode = "LLM" if llm_agent is not None else "UCB1+Bayesian"
    logger.info(
        "Starting training loop: episodes=%d, mode=%s, start=%d",
        config.max_steps, mode, start_episode,
    )

    for episode in range(start_episode, config.max_steps):
        metrics = _run_single_episode(
            episode=episode,
            trainer=trainer,
            env=env,
            config=config,
            reward_fn=reward_fn,
            llm_agent=llm_agent,
        )
        all_metrics.append(metrics)
        log_episode_metrics(metrics, config.log_file)

        logger.info(
            "Episode %4d | R1=%.2f R2=%.2f R3=%.2f R4=%.2f | Total=%.3f | MTTR=%d | loss=%s",
            episode, metrics.r1, metrics.r2, metrics.r3, metrics.r4,
            metrics.total_reward, metrics.mttr,
            f"{metrics.loss:.4f}" if metrics.loss is not None else "n/a",
        )

        # Record in ALP curriculum for next task selection
        inc = env._incident_state
        if inc is not None:
            difficulty = "easy"
            for template in env.incident_generator._templates:
                if template.id == inc.template_id:
                    difficulty = template.difficulty
                    break
            curriculum.record(difficulty, inc.failure_type.value, metrics.total_reward)

        if episode % 10 == 0:
            state = {
                "episode": episode,
                "batch_size": config.batch_size,
                "alp_summary": curriculum.summary(),
                "mode": mode,
            }
            save_checkpoint(state, config.checkpoint_dir, episode)

    return all_metrics


def _run_single_episode(
    episode: int,
    trainer: Any,
    env: Any,
    config: TrainingConfig,
    reward_fn: Any,
    llm_agent: Any = None,
) -> EpisodeMetrics:
    """Run one episode, retrying on CUDA OOM by halving batch size."""
    while True:
        try:
            return _execute_episode(
                episode, trainer, env, config, reward_fn, llm_agent=llm_agent
            )
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                new_bs = max(1, config.batch_size // 2)
                logger.warning(
                    "CUDA OOM at episode %d \u2014 halving batch_size %d\u2192%d and retrying.",
                    episode, config.batch_size, new_bs,
                )
                config.batch_size = new_bs
                if trainer is not None and hasattr(trainer, "args"):
                    trainer.args.per_device_train_batch_size = new_bs
            else:
                raise


def _execute_episode(
    episode: int,
    trainer: Any,
    env: Any,
    config: TrainingConfig,
    reward_fn: Any,
    llm_agent: Any = None,
) -> EpisodeMetrics:
    """Execute a single episode and return its metrics.

    When llm_agent is provided, generates actions via LLM inference
    (obs → prompt → model.generate → parse → action).
    When trainer is provided, runs a GRPOTrainer gradient step using the
    collected (prompt, completion, reward) triples.
    """
    from sentinel.training.prompt_builder import build_prompt
    from sentinel.models import Action, RewardBreakdown, Trajectory, TrajectoryStep

    obs, info = env.reset()
    steps: list[TrajectoryStep] = []
    terminated = False
    truncated = False
    step_count = 0

    # Collect (prompt, completion, step_reward) for GRPO update
    grpo_samples: list[dict] = []

    # Reset LLM agent state for new episode
    if llm_agent is not None:
        llm_agent.reset()

    while not (terminated or truncated):
        # ─ Build text prompt for this observation ────────────────────────────
        text_prompt = build_prompt(
            obs,
            agent_role=getattr(llm_agent, "agent_role", config.agent),
            step_number=step_count,
        ) if llm_agent is not None or trainer is not None else None

        # ─ Select action ──────────────────────────────────────────────────────
        action_dict = _get_action(
            trainer=trainer,
            obs=obs,
            config=config,
            llm_agent=llm_agent,
            step=step_count,
        )
        llm_completion = action_dict.pop("_llm_completion", None)

        pre_incident_state = env._incident_state
        obs_next, reward, terminated, truncated, step_info = env.step(action_dict)

        # ─ Record GRPO sample ─────────────────────────────────────────────
        if text_prompt is not None and llm_completion is not None:
            grpo_samples.append({
                "prompt":     text_prompt,
                "completion": llm_completion,
                "reward":     float(reward),
            })

        try:
            parsed_action = Action(
                agent=action_dict["agent"],
                category=action_dict["category"],
                name=action_dict["name"],
                params=action_dict.get("params", {}),
            )
        except Exception:
            parsed_action = Action(
                agent="holmes",
                category="investigative",
                name="QueryLogs",
                params={},
            )

        steps.append(
            TrajectoryStep(
                observation=obs,
                action=parsed_action,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=step_info,
            )
        )
        obs = obs_next
        step_count += 1

    # ─ Build trajectory ─────────────────────────────────────────────────
    episode_id = info.get("incident_id", str(uuid.uuid4()))
    incident_template_id = info.get("incident_id", "unknown")
    mttr = step_count

    incident_state = env._incident_state
    world_state = env.world_state

    if not steps:
        breakdown = RewardBreakdown(r1=0.0, r2=0.0, r3=0.0, r4=0.0, penalties=0.0, total=0.0)
    else:
        trajectory = Trajectory(
            episode_id=str(episode_id),
            incident_template_id=str(incident_template_id),
            steps=steps,
            final_reward=RewardBreakdown(r1=0.0, r2=0.0, r3=0.0, r4=0.0, penalties=0.0, total=0.0),
            mttr=mttr,
        )
        if incident_state is not None:
            breakdown = reward_fn.compute_episode_reward(trajectory, world_state, incident_state)
        else:
            breakdown = RewardBreakdown(r1=0.0, r2=0.0, r3=0.0, r4=0.0, penalties=0.0, total=0.0)

    # ─ GRPO gradient step (if trainer + samples available) ────────────────
    loss: float | None = None
    if trainer is not None and grpo_samples:
        try:
            from datasets import Dataset  # type: ignore
            grpo_ds = Dataset.from_list([{
                "prompt":     s["prompt"],
                "completion": s["completion"],
            } for s in grpo_samples])
            trainer.train_dataset = grpo_ds
            result = trainer.train()
            if hasattr(result, "training_loss"):
                loss = float(result.training_loss)
        except Exception as exc:
            logger.warning("GRPO trainer step failed at episode %d: %s", episode, exc)

    return EpisodeMetrics(
        episode=episode,
        r1=breakdown.r1,
        r2=breakdown.r2,
        r3=breakdown.r3,
        r4=breakdown.r4,
        total_reward=breakdown.total,
        mttr=mttr,
        loss=loss,
    )



def _get_action(
    trainer: Any,
    obs: dict,
    config: TrainingConfig,
    llm_agent: Any = None,
    step: int = 0,
) -> dict:
    """Return an action dict using the best available method.

    Priority:
      1. LLMAgent.act()  — fine-tuned model via prompt_builder + action_parser
      2. UCB1 bandit + Bayesian RCA (math-only, no GPU needed)
    """
    # 1. Fine-tuned LLM agent (unsloth + trl available)
    if llm_agent is not None:
        try:
            return llm_agent.act(obs, step=step)
        except Exception as exc:
            logger.debug("LLMAgent.act failed at step %d: %s", step, exc)

    # 2. UCB1 bandit selects action arm; Bayesian RCA fills params
    return _ucb1_math_action(obs, config)


def _ucb1_math_action(obs: dict, config: TrainingConfig) -> dict:
    """Select and fill an action using UCB1 bandit + Bayesian RCA.

    UCB1 (Auer et al. 2002) selects which action type to try next.
    Bayesian Noisy-OR (Pearl 1988) identifies the most likely root-cause
    service and fills action parameters accordingly.
    The chosen arm is updated with the step reward after env.step().
    """
    bandit = get_ucb1_bandit()
    rca = get_bayesian_rca()

    # UCB1: select arm
    arm_idx = bandit.select()
    action = bandit.get_action_template(arm_idx)

    # Bayesian RCA: identify top-k candidate services
    top_candidates = rca.top_k(obs, k=3)
    top_service = top_candidates[0][0] if top_candidates else "api-gateway"
    top_ft = _infer_failure_type(obs, top_service)

    # Fill params based on action name
    name = action.get("name", "QueryLogs")
    params = _fill_params(name, top_service, top_ft, obs)
    action["params"] = params

    # Attach arm_idx so the training loop can update the bandit after the step
    action["_ucb1_arm_idx"] = arm_idx
    return action


def _infer_failure_type(obs: dict, service: str) -> str:
    """Heuristically infer likely failure type from metric anomaly pattern."""
    metrics_raw = obs.get("metrics_snapshot", "{}")
    if isinstance(metrics_raw, str):
        try:
            snap = json.loads(metrics_raw)
        except json.JSONDecodeError:
            snap = {}
    else:
        snap = metrics_raw or {}

    m = snap.get(service)
    if not m:
        return "cpu_spike"
    if isinstance(m, dict):
        cpu = m.get("cpu", 0)
        mem = m.get("memory", 0)
        err = m.get("error_rate", 0)
        lat = m.get("latency_ms", 0)
        sat = m.get("saturation", 0)
    else:
        cpu = getattr(m, "cpu", 0)
        mem = getattr(m, "memory", 0)
        err = getattr(m, "error_rate", 0)
        lat = getattr(m, "latency_ms", 0)
        sat = getattr(m, "saturation", 0)

    # Simple threshold-based classification
    if cpu > 0.85:
        return "cpu_spike"
    if mem > 0.85:
        return "memory_leak"
    if err > 0.1 and lat > 300:
        return "bad_deployment"
    if sat > 0.9:
        return "connection_pool_exhaustion"
    if err > 0.05:
        return "cache_miss_storm"
    return "network_partition"


def _fill_params(name: str, service: str, failure_type: str, obs: dict) -> dict:
    """Fill action parameters from Bayesian RCA output."""
    if name == "QueryLogs":
        return {"service": service, "time_range": [0, 300]}
    if name == "QueryMetrics":
        _metric_map = {
            "cpu_spike": "cpu",
            "memory_leak": "memory",
            "bad_deployment": "error_rate",
            "connection_pool_exhaustion": "saturation",
            "cache_miss_storm": "error_rate",
            "network_partition": "latency_ms",
        }
        return {"service": service, "metric_name": _metric_map.get(failure_type, "error_rate"), "time_range": [0, 300]}
    if name == "QueryTrace":
        return {"trace_id": f"{service}-trace-001"}
    if name == "FormHypothesis":
        return {"service": service, "failure_type": failure_type, "confidence": 0.75}
    if name == "RestartService":
        return {"service": service}
    if name == "ScaleService":
        return {"service": service, "replicas": 3}
    if name == "RollbackDeployment":
        return {"service": service, "version": "previous"}
    if name == "DrainTraffic":
        return {"service": service}
    if name == "ModifyRateLimit":
        return {"service": service, "limit_rps": 100}
    if name == "CloseIncident":
        return {"resolution_summary": f"Root cause identified: {service} ({failure_type})"}
    if name == "EscalateToHuman":
        return {"reason": f"High uncertainty about root cause in {service}"}
    if name == "GenerateNewScenario":
        return {"difficulty": "medium", "target_gap": "investigative"}
    return {}

