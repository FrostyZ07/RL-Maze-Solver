# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PoC for a PPO reinforcement-learning agent navigating a PyBullet 3D maze (plan in `devplanpoc.md`). Stack: Gymnasium, Stable-Baselines3 (CPU torch), PyBullet, Flask, TensorBoard. No README, no linter configured.

## Commands

```
pip install -r requirements.txt
python train.py                 # PPO, 500k steps, headless; checkpoints every 10k -> logs/, final -> ppo_maze_final.zip
python app.py                   # Flask dashboard :5000, auto-spawns TensorBoard :6006 on logs/
python test_agent.py            # GUI rollout of the newest logs/*.zip checkpoint (falls back to ppo_maze_final)
python solve_maze.py            # GUI demo: A* path through env.layout, driven by the env
python manual_test.py           # GUI, arrow-key control
pytest tests/                   # single test: pytest tests/test_env.py::test_maze_env_step
```

Run scripts from the repo root (they import `envs.maze_env` and `app.py` launches `train.py` via relative cwd).

## Architecture

- `envs/maze_env.py` — `MazeEnv`, the only environment. Connects to PyBullet (`GUI` if `render_mode='human'`, else `DIRECT`). `reset()` rebuilds the whole simulation each episode: a hardcoded 9x9 `self.layout` (1=wall) with grid cell (r,c) -> world (c-4, r-4), agent start (-3,-3), goal (3,3). Obs = `[dist_to_target, sin(heading_err), cos(heading_err), 32 lidar distances]` (35-dim). Actions are `Discrete(3)`: forward / turn left / turn right via `resetBaseVelocity`. Reward is shaped (progress, heading, step/stuck/collision penalties, +20 waypoint, +100 goal). `self.waypoints` is currently a single direct-to-goal point `[3,3]`, so the agent navigates by lidar rather than following a path.
- `train.py` — SB3 PPO `MlpPolicy` on CPU, TensorBoard logs under `logs/PPO_run_N`.
- `app.py` — Flask wrapper; `/start_training`, `/stop_training`, `/status` manage `train.py` as a subprocess; UI in `templates/index.html`.
- `solve_maze.py` duplicates the grid<->world conversion (hardcoded offset 4) and drives the env with a continuous-style controller on the layout.

## Gotchas

- `tests/test_env.py` and `manual_test.py` are stale: they assume the old continuous 2-dim action space and 14-dim observation (`action_space.shape[0] == 2`, `obs.shape[0] == 14`), but the env is now `Discrete(3)` with 35-dim obs. Tests will fail until updated to match the env.
- `test_agent.py` picks the checkpoint by `getctime` over `logs/*.zip`, which can differ from the one you just trained.
- Layout/start/goal are hardcoded in `reset()`; changing maze size also requires updating the `-4` offsets in `maze_env.py` and `solve_maze.py`.
