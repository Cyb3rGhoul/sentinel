"""Modal entrypoint for SENTINEL GPU training.

Usage examples:
  modal run modal_train.py::train --agent holmes
  modal run modal_train.py::train --agent forge
  modal run modal_train.py::train --agent hermes
"""
from __future__ import annotations

import modal

app = modal.App("sentinel-training")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu24.04",
        add_python="3.10",
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("unsloth", "trl", "datasets")
    .env({
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONPATH": "/root/sentinel",
    })
)

volume = modal.Volume.from_name("sentinel-checkpoints", create_if_missing=True)

sentinel_mount = modal.Mount.from_local_dir(
    ".",
    remote_path="/root/sentinel",
    condition=lambda path: not any(
        x in path for x in [".git", "__pycache__", ".pytest_cache", "checkpoints", ".venv", ".mypy_cache"]
    ),
)

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 12,
    volumes={"/mnt/checkpoints": volume},
    mounts=[sentinel_mount],
)
def train(
    agent: str = "holmes",
    episodes: int = 500,
    batch_size: int = 2,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    load_in_4bit: bool = False,
) -> str:
    import subprocess

    log_file = f"/mnt/checkpoints/{agent}_training_log.jsonl"
    checkpoint_dir = "/mnt/checkpoints"

    cmd = [
        "python",
        "/root/sentinel/train.py",
        "--agent",
        agent,
        "--model",
        model,
        "--episodes",
        str(episodes),
        "--batch-size",
        str(batch_size),
        "--no-4bit" if not load_in_4bit else "--load-in-4bit",
        "--checkpoint-dir",
        checkpoint_dir,
        "--log-file",
        log_file,
    ]
    subprocess.run(cmd, check=True, cwd="/root/sentinel")
    return f"Finished training {agent}. Outputs written under {checkpoint_dir}/{agent}"


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    volumes={"/mnt/checkpoints": volume},
    mounts=[sentinel_mount],
)
def evaluate(
    agent: str = "holmes",
    eval_episodes: int = 20,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    load_in_4bit: bool = False,
) -> str:
    import subprocess

    checkpoint_dir = "/mnt/checkpoints"
    cmd = [
        "python",
        "/root/sentinel/train.py",
        "--agent",
        agent,
        "--model",
        model,
        "--eval-only",
        "--eval-episodes",
        str(eval_episodes),
        "--no-4bit" if not load_in_4bit else "--load-in-4bit",
        "--checkpoint-dir",
        checkpoint_dir,
    ]
    subprocess.run(cmd, check=True, cwd="/root/sentinel")
    return f"Finished evaluation for {agent}"


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 10,
    mounts=[sentinel_mount],
)
def gpu_sanity() -> str:
    import subprocess

    subprocess.run(["nvidia-smi"], check=True)
    out = subprocess.run(
        [
            "python",
            "-c",
            "import torch; print({'cuda_available': torch.cuda.is_available(), 'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()
