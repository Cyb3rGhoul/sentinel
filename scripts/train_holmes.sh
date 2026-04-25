#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --agent holmes \
  --episodes "${EPISODES:-500}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --checkpoint-dir "${CHECKPOINT_DIR:-checkpoints}" \
  --log-file "${LOG_FILE:-holmes_training_log.jsonl}"
