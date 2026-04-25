# Hackathon Submission Checklist

This file maps the repo to the OpenEnv hackathon requirements and highlights what still needs to be filled before submission.

## Non-Negotiables

- OpenEnv package:
  - repo now points to `openenv-core[core]==0.2.3` in `requirements.txt`
  - source: https://meta-pytorch.org/OpenEnv/installation.html
  - source: https://pypi.org/project/openenv-core/0.1.1/

- OpenEnv manifest:
  - present at `openenv.yaml`

- Gym-style environment:
  - `sentinel/env.py`
  - `reset()`, `step()`, `render()`, `close()`

- Client/server deployment surface:
  - FastAPI server at `sentinel/api/server.py`

- Training pipeline:
  - `train.py`
  - `sentinel/training/pipeline.py`
  - `modal_train.py`

- Colab notebook:
  - `sentinel_colab_training.ipynb`

- Modal notebook:
  - `sentinel_modal_notebook.ipynb`

- Short explanation materials:
  - blog draft: `blog/huggingface_post.md`
  - video draft: `blog/youtube_script.md`

- Results folder:
  - `results/`

## Theme Positioning

Best fit:
- `Theme #3.1 World Modeling`
- `Theme #2 Long-Horizon Planning`

Supportive angle:
- role-specialized multi-agent task handling

Avoid overclaiming:
- current runtime is stronger as a world-model / incident-ops environment than as a negotiation-heavy multi-agent interaction benchmark

## What Judges Need To See

- baseline vs trained comparison
- reward curve
- loss curve
- before/after transcript or episode walkthrough
- clear statement of what improved

## Still Required Before Final Submission

- create and link the Hugging Face Space
- add the Hugging Face Space URL to `README.md`
- publish blog/video and link it from `README.md`
- run longer hosted training for `holmes` and `forge`
- commit final plots into `results/`
- add final before/after metrics to `README.md`

## Recommended Run Order

1. `holmes`
2. `forge`
3. `hermes`
4. `oracle`
5. `argus`

## Hosted Training Recommendation

- fastest current path: Modal `A100-80GB`
- best use of Hugging Face credits: Space hosting first, extra training second

## Sources

- OpenEnv installation docs: https://meta-pytorch.org/OpenEnv/installation.html
- OpenEnv GitHub: https://github.com/meta-pytorch/OpenEnv
- Hugging Face Spaces overview: https://huggingface.co/docs/hub/en/spaces
- Hugging Face GPU Spaces: https://huggingface.co/docs/hub/en/spaces-gpus
