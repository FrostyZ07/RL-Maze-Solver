# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`faultwalk`: fault-tolerant locomotion for a simulated Unitree Go2 quadruped. One policy should detect and identify a locked, limp or weak motor from its own joint/IMU history and keep tracking velocity commands. Currently at milestone M0 (feasibility test, policy contract, clean-up); most of the package does not exist yet.

## Where things run

- **Training:** Kaggle/Colab GPU only, PyTorch + mjlab, installed inside the notebooks (`notebooks/`). The `train` extra in `pyproject.toml` is intentionally empty.
- **Everything else:** CPU on the local Windows laptop: plain MuJoCo + ONNX Runtime for evaluation, demos and tests. No local GPU, so do not add torch/mjlab to core dependencies.
- Policies cross that boundary as ONNX files plus a metadata file (the "policy contract").

## Layout

- `src/faultwalk/` — the package (src layout); version in `__init__.py`.
- `tests/` — pytest suite (`testpaths` is set to `tests`, so `legacy/` is never collected).
- `.github/workflows/ci.yml` — ruff + pytest on ubuntu-latest and windows-latest, Python 3.12.
- `legacy/` — the old PyBullet maze PoC. **Frozen:** do not edit, refactor or lint it (ruff excludes it). Its scripts run only from inside `legacy/` with `legacy/requirements.txt`. Its model zips are in `legacy/models/` and are untracked.

## Commands

```
pip install -e ".[dev]"     # core deps: numpy, mujoco, onnxruntime; dev: pytest, ruff
ruff check .
pytest                      # single test: pytest tests/test_import.py::test_version
```

## Conventions

- Python >= 3.10 (CI and `.python-version` use 3.12).
- Model files (`*.zip`, `*.pt`, `*.onnx`, ...), `logs/` and `results/` are gitignored; checkpoints go to Hugging Face Hub. A small ONNX file that must be committed goes under `assets/`, which is allowlisted.
