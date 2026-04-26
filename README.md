---
title: SENTINEL
emoji: "🛡️"
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# SENTINEL

LLM-first incident response environment for autonomous cloud operations.

SENTINEL is a Gymnasium-compatible environment for training and evaluating agents on realistic production outages across a 30-service microservice platform. It includes cascading failures, partial observability, role-constrained actions, an OpenEnv adapter, a public Hugging Face Space, and committed training artifacts.

## Submission Assets

| Deliverable | Link |
|-------------|------|
| GitHub Repository | [github.com/sayantikalaskar/sentinel](https://github.com/sayantikalaskar/sentinel) |
| Hugging Face Space | [huggingface.co/spaces/harry1911/sentinel](https://huggingface.co/spaces/harry1911/sentinel) |
| Live Runtime | [harry1911-sentinel.hf.space](https://harry1911-sentinel.hf.space/) |
| Training Notebook | [`sentinel_colab_training.ipynb`](sentinel_colab_training.ipynb) |
| Training Scripts | [`train.py`](train.py), [`retrain.py`](retrain.py) |
| OpenEnv Manifest | [`openenv.yaml`](openenv.yaml) |
| Blog Write-up | [`Blog.MD`](Blog.MD) |
| Training Curves | [`results/`](results/) |

## Live Validation

The public Space exposes:

- Space page: [huggingface.co/spaces/harry1911/sentinel](https://huggingface.co/spaces/harry1911/sentinel)
- Runtime root: [harry1911-sentinel.hf.space](https://harry1911-sentinel.hf.space/)
- Health endpoint: [harry1911-sentinel.hf.space/health](https://harry1911-sentinel.hf.space/health)
- API docs: [harry1911-sentinel.hf.space/docs](https://harry1911-sentinel.hf.space/docs)

The Space UI is a validation console. It can:

- reset a live episode
- execute valid sample actions
- run a smoke test across `reset`, `step`, and `state`
- display live observation and state JSON

## What It Simulates

Each episode models a cloud incident with:

- a 30-service dependency graph
- cascading failures
- partial observability with hidden services and noisy evidence
- agent roles for investigation, remediation, deployment, and incident closure

Each observation contains:

- `metrics_snapshot`
- `active_alerts`
- `causal_graph_snapshot`
- `recent_logs`
- `active_traces`
- `incident_context`
- `sla_state`

## OpenEnv Structure

The OpenEnv-compatible pieces are:

- environment wrapper: [`server/sentinel_environment.py`](server/sentinel_environment.py)
- manifest: [`openenv.yaml`](openenv.yaml)
- runtime entrypoint: [`server/app.py`](server/app.py)
- Gym-style environment: [`sentinel/env.py`](sentinel/env.py)

The adapter exposes:

- `reset(...)`
- `step(...)`
- `state`

## Training Evidence

Committed plots required for validation are present in [`results/`](results/), including reward and loss curves.

### Comparison Plots

![All Agents Comparison](results/comparison_all_agents.png)

![Loss Comparison](results/comparison_loss.png)

### Individual Curves

![Holmes Training](results/holmes_training_curves.png)
![Holmes Loss](results/holmes_loss_curve.png)

![Forge Training](results/forge_training_curves.png)
![Forge Loss](results/forge_loss_curve.png)

![Argus Training](results/argus_training_curves.png)
![Argus Loss](results/argus_loss_curve.png)

![Hermes Training](results/hermes_training_curves.png)
![Hermes Loss](results/hermes_loss_curve.png)

![Oracle Training](results/oracle_training_curves.png)
![Oracle Loss](results/oracle_loss_curve.png)

## Training

Quick start:

```bash
pip install -r requirements.txt
python -m pytest -q
python -c "from sentinel.env import Sentinel_Env; env = Sentinel_Env(); obs, info = env.reset(); print(info)"
```

Training entrypoints:

```bash
python train.py --agent holmes --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 500 --batch-size 2
python train.py --agent forge --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 500 --batch-size 2
python train.py --agent hermes --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 300 --batch-size 2
python train.py --agent oracle --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 300 --batch-size 2
python train.py --agent argus --model Qwen/Qwen2.5-7B-Instruct --no-4bit --episodes 300 --batch-size 2
```

Hosted GPU instructions are in [`TRAINING.md`](TRAINING.md).

## Validation Checklist

- [x] public, logged-out, cloneable Hugging Face Space
- [x] parseable [`openenv.yaml`](openenv.yaml)
- [x] OpenEnv wrapper with `reset`, `step`, and `state`
- [x] committed reward and loss plots in the repo
- [x] runnable training notebook and scripts
- [x] README links every required deliverable

## Project Structure

```text
sentinel/
├── demo/
├── results/
├── sentinel/
├── server/
├── tests/
├── Blog.MD
├── Dockerfile
├── README.md
├── TRAINING.md
├── client.py
├── env_spec.yaml
├── incident_library.yaml
├── models.py
├── openenv.yaml
├── requirements.txt
├── retrain.py
├── sentinel_colab_training.ipynb
└── train.py
```

## Hackathon Fit

Primary category:

- `World Modeling`

Secondary category:

- `Long-Horizon Planning`

## Authors

Harsh Shukla and Sayantika Laskar

Built for the Meta PyTorch OpenEnv Hackathon 2026.

## License

MIT
