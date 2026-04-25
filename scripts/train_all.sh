#!/usr/bin/env bash
set -euo pipefail

EPISODES="${EPISODES:-500}"
BATCH_SIZE="${BATCH_SIZE:-2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"

EPISODES="$EPISODES" BATCH_SIZE="$BATCH_SIZE" CHECKPOINT_DIR="$CHECKPOINT_DIR" LOG_FILE="${HOLMES_LOG_FILE:-holmes_training_log.jsonl}" ./scripts/train_holmes.sh
EPISODES="$EPISODES" BATCH_SIZE="$BATCH_SIZE" CHECKPOINT_DIR="$CHECKPOINT_DIR" LOG_FILE="${FORGE_LOG_FILE:-forge_training_log.jsonl}" ./scripts/train_forge.sh
