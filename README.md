# faultwalk

A robot dog that keeps walking when a motor fails. The goal is a single locomotion policy for a simulated Unitree Go2 quadruped that, using only the robot's own joint and IMU readings, notices that a motor has locked, gone limp or weakened, identifies which one, and keeps following velocity commands where a standard policy falls over. Training uses PyTorch with mjlab (GPU-parallel MuJoCo) on free Kaggle/Colab GPUs; evaluation, demos and tests run on CPU with plain MuJoCo and ONNX Runtime.

**Status:** M0 in progress (feasibility test, policy contract, repository clean-up). There are no results yet.

## Planned layout

```
src/faultwalk/   package: fault model, task configs, models, ONNX export, CPU MuJoCo runner, evaluation
notebooks/       Kaggle/Colab training notebooks
web/             browser demo
ros2/            rclpy node + Dockerfile
tests/           pytest suite
legacy/          the earlier PyBullet maze PoC, frozen
```

Not all of these exist yet.

## Development

```
pip install -e ".[dev]"
ruff check .
pytest
```

## Legacy

The previous project in this repository, a PPO agent for PyBullet mazes, is kept unchanged in [`legacy/`](legacy/README.md).

## License

MIT, see [LICENSE](LICENSE).
