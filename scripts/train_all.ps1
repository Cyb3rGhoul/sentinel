$env:LOG_FILE = "holmes_training_log.jsonl"
./scripts/train_holmes.ps1

$env:LOG_FILE = "forge_training_log.jsonl"
./scripts/train_forge.ps1
