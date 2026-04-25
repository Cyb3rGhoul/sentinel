$episodes = if ($env:EPISODES) { $env:EPISODES } else { "500" }
$batchSize = if ($env:BATCH_SIZE) { $env:BATCH_SIZE } else { "2" }
$checkpointDir = if ($env:CHECKPOINT_DIR) { $env:CHECKPOINT_DIR } else { "checkpoints" }
$logFile = if ($env:LOG_FILE) { $env:LOG_FILE } else { "forge_training_log.jsonl" }

python train.py `
  --agent forge `
  --episodes $episodes `
  --batch-size $batchSize `
  --checkpoint-dir $checkpointDir `
  --log-file $logFile
