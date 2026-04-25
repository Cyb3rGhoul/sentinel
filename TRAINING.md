# Training Guide

This repo is ready for **LLM-only** GRPO training. The local workspace is CPU-only, so real runs should happen on a hosted CUDA GPU such as Modal.

## Recommended Hosted Setup

Use:
- `A100-80GB`
- `Qwen/Qwen2.5-7B-Instruct`
- `--no-4bit`

That is the safest path for this codebase right now. It avoids the bitsandbytes CUDA-runtime mismatch that can show up with 4-bit hosted runs.

## Modal Setup

```bash
pip install modal
modal setup
```

If you want a judge-friendly rerunnable notebook, use:
- `sentinel_colab_training.ipynb`

## Modal Smoke Test

Run this first to verify the GPU container is healthy:

```bash
modal run modal_train.py::gpu_sanity
```

## Modal Training

Train the core two agents first:

```bash
modal run modal_train.py::train --agent holmes --episodes 300 --batch-size 2
modal run modal_train.py::train --agent forge --episodes 300 --batch-size 2
```

Then train the supporting agents if you have budget:

```bash
modal run modal_train.py::train --agent hermes --episodes 200 --batch-size 2
modal run modal_train.py::train --agent oracle --episodes 200 --batch-size 2
modal run modal_train.py::train --agent argus --episodes 200 --batch-size 2
```

Evaluate:

```bash
modal run modal_train.py::evaluate --agent holmes --eval-episodes 20
modal run modal_train.py::evaluate --agent forge --eval-episodes 20
modal run modal_train.py::evaluate --agent hermes --eval-episodes 20
modal run modal_train.py::evaluate --agent oracle --eval-episodes 20
modal run modal_train.py::evaluate --agent argus --eval-episodes 20
```

Checkpoints and logs are written to the Modal Volume `sentinel-checkpoints`.

## Agent Priority

Train in this order:

1. `holmes`
2. `forge`
3. `hermes`
4. `oracle`
5. `argus`

What each one does:
- `holmes`: diagnosis and root-cause investigation
- `forge`: remediation and service recovery
- `hermes`: deployment safety and rollout control
- `oracle`: escalation, closure, and scenario control
- `argus`: monitoring support and evidence gathering

Why `holmes` and `forge` first:
- they dominate the benchmark outcome
- `holmes` drives diagnosis quality
- `forge` drives MTTR, recovery, and blast-radius reduction

## Local Or Other GPU Machine

Install:

```bash
git clone <your-repo-url>
cd sentinel
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install unsloth trl datasets
```

Train:

```bash
python train.py --agent holmes --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 500 --batch-size 2
python train.py --agent forge --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 500 --batch-size 2
python train.py --agent hermes --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 300 --batch-size 2
python train.py --agent oracle --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 300 --batch-size 2
python train.py --agent argus --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 300 --batch-size 2
```

Resume:

```bash
python train.py --agent holmes --model Qwen/Qwen2.5-7B-Instruct --no-4bit --resume --checkpoint-dir checkpoints
python train.py --agent forge --model Qwen/Qwen2.5-7B-Instruct --no-4bit --resume --checkpoint-dir checkpoints
```

Evaluate:

```bash
python train.py --agent holmes --model Qwen/Qwen2.5-7B-Instruct --no-4bit --eval-only --eval-episodes 20
python train.py --agent forge --model Qwen/Qwen2.5-7B-Instruct --no-4bit --eval-only --eval-episodes 20
```

## Files To Bring Back

Copy these back after training:
- `checkpoints/holmes/`
- `checkpoints/forge/`
- `checkpoints/hermes/`
- `checkpoints/oracle/`
- `checkpoints/argus/`
- per-agent training logs if you split them

For cleaner logs:

```bash
python train.py --agent holmes --log-file holmes_training_log.jsonl
python train.py --agent forge --log-file forge_training_log.jsonl
```
