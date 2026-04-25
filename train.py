#!/usr/bin/env python
"""SENTINEL LLM Training Entry Point.

One-command launcher for GRPO fine-tuning of Llama-3 / Qwen2.5 on the
SENTINEL multi-agent incident-response environment.

Usage
-----
# Full LLM training (requires GPU + unsloth + trl):
    python train.py --agent holmes --episodes 500

# Simulation mode (no GPU needed — UCB1+Bayesian agent):
    python train.py --agent holmes --episodes 100 --sim-only

# Resume from checkpoint:
    python train.py --agent forge --resume --checkpoint-dir checkpoints/forge

# Evaluate after training:
    python train.py --agent holmes --eval-only --episodes 60

Environment
-----------
    pip install unsloth trl datasets
    # GPU: CUDA 12+ recommended; 16 GB VRAM for 4-bit Llama-3-8B

Training Flow (with GPU)
------------------------
  env.reset()
      │
      ▼
  prompt_builder.build_messages(obs)      ← obs → chat messages
      │
      ▼
  tokenizer.apply_chat_template(messages) ← messages → input_ids
      │
      ▼
  model.generate(input_ids)               ← LLM inference (4-bit LoRA)
      │
      ▼
  action_parser.parse_llm_action(output)  ← raw text → Action dict
      │
      ▼
  env.step(action) → (obs, reward, ...)
      │
      ▼
  GRPOTrainer.train(prompts, completions) ← GRPO gradient step
      │
      ▼
  ALPCurriculum.record(reward)            ← adapt difficulty
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

# Suppress noisy RuntimeWarning emitted by unsloth/trl import guards when
# no GPU is present (we handle that case explicitly in main()).
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Logging setup (before any sentinel imports so config is inherited)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sentinel.train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a SENTINEL LLM agent via GRPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--agent", choices=["holmes", "forge", "argus", "oracle"],
        default="holmes",
        help="Which agent role to train.",
    )
    p.add_argument(
        "--model",
        default="unsloth/Meta-Llama-3-8B-Instruct",
        help="HuggingFace / Unsloth model name or local path.",
    )
    p.add_argument(
        "--episodes", type=int, default=200,
        help="Number of training episodes (= GRPO steps).",
    )
    p.add_argument(
        "--batch-size", type=int, default=4,
        help="Per-device GRPO batch size.",
    )
    p.add_argument(
        "--lora-r", type=int, default=16,
        help="LoRA rank.",
    )
    p.add_argument(
        "--lora-alpha", type=int, default=32,
        help="LoRA alpha.",
    )
    p.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="Directory for saving checkpoints.",
    )
    p.add_argument(
        "--log-file", default="training_log.jsonl",
        help="JSONL file to append per-episode metrics.",
    )
    p.add_argument(
        "--env-spec", default="env_spec.yaml",
        help="Path to env_spec.yaml.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from the latest checkpoint in --checkpoint-dir.",
    )
    p.add_argument(
        "--sim-only", action="store_true",
        help="Skip LLM loading; run UCB1+Bayesian agent only (no GPU needed).",
    )
    p.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; run evaluation and print report.",
    )
    p.add_argument(
        "--eval-episodes", type=int, default=20,
        help="Episodes per difficulty tier in evaluation.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ── Imports ─────────────────────────────────────────────────────────────
    logger.info("Importing SENTINEL modules …")
    from sentinel.env import Sentinel_Env
    from sentinel.training.pipeline import (
        TrainingConfig,
        build_grpo_trainer,
        load_latest_checkpoint,
        run_training_loop,
    )
    from sentinel.training.evaluate import print_eval_report, run_evaluation

    # ── Environment ──────────────────────────────────────────────────────────
    logger.info("Initialising environment from %s …", args.env_spec)
    env = Sentinel_Env(config_path=args.env_spec)
    reward_fn = env.reward_function

    # ── Training config ──────────────────────────────────────────────────────
    config = TrainingConfig(
        agent=args.agent,
        model_name=args.model,
        batch_size=args.batch_size,
        max_steps=args.episodes,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        checkpoint_dir=str(Path(args.checkpoint_dir) / args.agent),
        log_file=args.log_file,
    )

    # ── Eval-only mode ───────────────────────────────────────────────────────
    if args.eval_only:
        logger.info("Eval-only mode: running %d episodes per tier …", args.eval_episodes)
        results = run_evaluation(
            env, reward_fn, episodes_per_tier=args.eval_episodes, seed=args.seed
        )
        print_eval_report(results)
        return 0

    # ── Auto-detect GPU / sim mode ───────────────────────────────────────────
    _has_gpu = False
    try:
        import torch
        _has_gpu = torch.cuda.is_available()
    except ImportError:
        pass

    trainer = None
    llm_agent = None

    if args.sim_only or not _has_gpu:
        if args.sim_only:
            reason = "--sim-only flag set"
        else:
            reason = "no CUDA GPU detected"
        logger.info("=" * 54)
        logger.info("  SENTINEL Training Mode : UCB1 + Bayesian RCA (sim)")
        logger.info("  Reason  : %s", reason)
        logger.info("  GPU run : python train.py --agent holmes")
        logger.info("            (needs CUDA GPU + pip install unsloth trl)")
        logger.info("=" * 54)
    else:
        logger.info("=" * 54)
        logger.info("  SENTINEL Training Mode : LLM (GRPO)")
        logger.info("  Model  : %s", args.model)
        logger.info("  GPU    : %s", torch.cuda.get_device_name(0))
        logger.info("=" * 54)
        logger.info("Building GRPOTrainer + LLMAgent for agent=%s ...", args.agent)
        trainer, llm_agent = build_grpo_trainer(
            agent=args.agent,
            env=env,
            config=config,
        )
        if trainer is None:
            logger.warning(
                "build_grpo_trainer returned None -- falling back to UCB1+Bayesian sim.\n"
                "Check that unsloth and trl are installed correctly."
            )



    # ── Resume ───────────────────────────────────────────────────────────────
    start_episode = 0
    if args.resume:
        ckpt = load_latest_checkpoint(config.checkpoint_dir)
        if ckpt:
            start_episode = ckpt.get("episode", 0) + 1
            logger.info("Resuming from episode %d …", start_episode)
        else:
            logger.info("No checkpoint found in %s; starting from episode 0.", config.checkpoint_dir)

    # ── Training loop ────────────────────────────────────────────────────────
    mode_str = "LLM (GRPO)" if llm_agent is not None else "UCB1+Bayesian (sim)"
    logger.info(
        "Starting training | agent=%s | mode=%s | episodes=%d | start=%d",
        args.agent, mode_str, args.episodes, start_episode,
    )
    t0 = time.perf_counter()

    all_metrics = run_training_loop(
        trainer=trainer,
        env=env,
        config=config,
        reward_fn=reward_fn,
        start_episode=start_episode,
        llm_agent=llm_agent,
    )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Training complete: %d episodes in %.1f s (%.2f s/ep)",
        len(all_metrics), elapsed, elapsed / max(len(all_metrics), 1),
    )

    # ── Print final metrics summary ──────────────────────────────────────────
    if all_metrics:
        last_10 = all_metrics[-10:]
        avg_reward = sum(m.total_reward for m in last_10) / len(last_10)
        avg_mttr   = sum(m.mttr        for m in last_10) / len(last_10)
        logger.info(
            "Last 10 episodes | avg_reward=%.3f | avg_MTTR=%.1f",
            avg_reward, avg_mttr,
        )

    # ── Post-training evaluation ─────────────────────────────────────────────
    logger.info("Running post-training evaluation (%d eps/tier) …", args.eval_episodes)
    results = run_evaluation(
        env, reward_fn, episodes_per_tier=args.eval_episodes, seed=args.seed
    )
    print_eval_report(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
