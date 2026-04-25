<p align="center">
  <h1 align="center">SENTINEL</h1>
  <p align="center">
    <strong>LLM-First Incident Response Environment for Autonomous Cloud Operations</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/OpenEnv-Hackathon_2026-blue" alt="OpenEnv"/>
    <img src="https://img.shields.io/badge/Python-3.10+-green" alt="Python"/>
    <img src="https://img.shields.io/badge/Gymnasium-1.3.0-orange" alt="Gymnasium"/>
    <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
  </p>
</p>

---

## What You Are Building

SENTINEL is a Gymnasium-compatible environment for training and evaluating LLM agents on realistic cloud incident response.

It simulates:
- a 30-service microservice platform
- cascading failures over a dependency graph
- partial observability with hidden services, missing logs, and red herrings
- role-constrained actions for investigation, remediation, deployment, and incident closure

The active training path is **LLM-only**:
- observations are converted into structured prompts
- the model emits one valid JSON action
- the environment executes that action
- GRPO optimizes the model against the resulting reward signal

There is no active math-policy fallback in the training loop anymore.

---

## Why It Matters

Operational incidents are long-horizon tasks:
- alerts can be misleading
- evidence is incomplete
- wrong actions can expand the blast radius
- the agent has to diagnose and remediate under time pressure

SENTINEL turns that into a trainable benchmark instead of a toy Q&A task.

---

## Core Environment

### Observation

Each step exposes:
- `metrics_snapshot`
- `active_alerts`
- `causal_graph_snapshot`
- `recent_logs`
- `active_traces`
- `incident_context`
- `sla_state`

### Action Roles

| Agent | Purpose | Typical actions |
|-------|---------|-----------------|
| `holmes` | Root-cause investigation | `QueryLogs`, `QueryMetrics`, `QueryTrace`, `FormHypothesis` |
| `forge` | Remediation | `RestartService`, `ScaleService`, `RollbackDeployment`, `DrainTraffic`, `ModifyRateLimit`, `ModifyConfig` |
| `hermes` | Deployment changes | `CanaryDeploy`, `FullDeploy`, `Rollback` |
| `oracle` | Closure / escalation / scenario management | `CloseIncident`, `EscalateToHuman`, `GenerateNewScenario` |
| `argus` | Monitoring support | `QueryLogs`, `QueryMetrics` |

### Reward

Episode reward combines:
- `R1`: diagnosis accuracy
- `R2`: MTTR efficiency
- `R3`: recovery quality
- `R4`: blast-radius minimization
- penalties for harmful or invalid behavior

Step rewards also shape:
- useful investigation
- correct hypotheses
- targeted remediation
- harmful blast-radius expansion
- restarting healthy services

---

## Which Agents To Train

Right now, the highest-value trainable agents are:
- `holmes`: because diagnosis quality determines whether the rest of the workflow is even correct
- `forge`: because remediation quality determines MTTR, recovery quality, and blast-radius reduction

Why not train all agents first:
- `argus` is mostly an observation helper and overlaps heavily with `holmes`
- `hermes` is narrower and can start as a deterministic deployment-safety policy
- `oracle` is meta-control and scenario management, which is useful but less critical than diagnosis + remediation for the main benchmark loop

For a hackathon-grade result, training `holmes` and `forge` first is the correct priority.

If you have GPU budget, the next order is:
1. `hermes`
2. `oracle`
3. `argus`

---

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest -q
python -c "from sentinel.env import Sentinel_Env; env = Sentinel_Env(); obs, info = env.reset(); print(info)"
```

---

## Training

Training requires:
- NVIDIA CUDA GPU
- `unsloth`
- `trl`
- `datasets`
- latest `openenv-core`

Recommended hosted path:

```bash
modal run modal_train.py::gpu_sanity
modal run modal_train.py::train --agent holmes --episodes 300 --batch-size 2
modal run modal_train.py::train --agent forge --episodes 300 --batch-size 2
```

The stable configuration uses `Qwen/Qwen2.5-7B-Instruct` with `--no-4bit` on an `A100-80GB`.

Local or other rented GPU:

```bash
python train.py --agent holmes --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 500 --batch-size 2
python train.py --agent forge --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 500 --batch-size 2
python train.py --agent hermes --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 300 --batch-size 2
python train.py --agent oracle --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 300 --batch-size 2
python train.py --agent argus --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 300 --batch-size 2
```

Detailed hosted-GPU instructions are in `TRAINING.md`.

---

## Submission Assets

Judge-facing materials live here:
- `sentinel_modal_notebook.ipynb`
- `openenv.yaml`
- `sentinel_colab_training.ipynb`
- `TRAINING.md`
- `blog/huggingface_post.md`
- `blog/youtube_script.md`
- `results/`
- `HACKATHON_CHECKLIST.md`

Add these links before submission:
- Hugging Face Space URL
- Hugging Face blog URL or YouTube URL
- final training plots in `results/`
- before/after evaluation summary in `README.md`

---

## Hackathon Fit

SENTINEL is best positioned as:
- primary: `World Modeling`
- secondary: `Long-Horizon Planning`

Why:
- the agent operates inside a partially observable cloud-operations world
- incidents require long multi-step diagnosis and remediation
- actions interact with realistic system state instead of static text tasks

---

## Judge Checklist

- uses the current OpenEnv package path via `openenv-core==0.2.3`
- provides a working TRL / Unsloth training path
- includes a Modal-native notebook for hosted training: `sentinel_modal_notebook.ipynb`
- includes a Colab notebook for hosted training: `sentinel_colab_training.ipynb`
- includes an OpenEnv manifest: `openenv.yaml`
- includes a FastAPI server for deployment: `sentinel/api/server.py`
- includes blog/video draft materials
- includes result artifacts and plotting workflow

Pending before final submission:
- add the real Hugging Face Space URL
- run full multi-episode training and commit final plots
- update README with final before/after metrics

---

## Project Structure

```text
sentinel/
├── sentinel/
│   ├── env.py
│   ├── reward.py
│   ├── models.py
│   ├── world_state.py
│   ├── cascade_engine.py
│   ├── observability.py
│   ├── incident_generator.py
│   ├── config.py
│   ├── agents/
│   ├── training/
│   │   ├── pipeline.py
│   │   ├── llm_agent.py
│   │   ├── prompt_builder.py
│   │   ├── action_parser.py
│   │   └── evaluate.py
│   └── api/
│       └── server.py
├── demo/app.py
├── train.py
├── tests/
├── env_spec.yaml
├── incident_library.yaml
├── requirements.txt
└── TRAINING.md
```

---

## Current Workspace Status

- reward wiring is fixed
- diagnosis metadata flows correctly into episode reward
- training and evaluation are LLM-only
- prompt/action schema matches the actual environment
- demo import side effects were removed
- full tests pass

This workspace is currently CPU-only, so actual GRPO training cannot be run here.

---

## License

MIT
