# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PoC for a PPO reinforcement-learning agent navigating a PyBullet 3D maze (plan in `devplanpoc.md`). Stack: Gymnasium, Stable-Baselines3 (CPU torch), PyBullet, Flask, TensorBoard. No README, no linter configured. Python 3.12 (`.python-version`); `requirements.txt` is pinned. pybullet has no Windows wheel, so installing it there needs the MSVC C++ Build Tools.

## Commands

```
pip install -r requirements.txt
python train.py                 # PPO, 5M steps, 8 envs, headless; checkpoints every 10k -> logs/, final -> ppo_maze_final.zip
python train.py --seed 1 --timesteps 2000000
python train.py --resume logs/ppo_maze_100000_steps.zip --timesteps 500000   # continues weights + curriculum level; steps are added on top
python evaluate.py              # fixed eval set: 5 sizes x 200 seeded held-out mazes; Random, NodeCount, Oracle, PPO -> table + results/eval_*.csv/json
python evaluate.py --models a.zip b.zip --sizes 11 15 --episodes 50 --deterministic
python app.py                   # Flask dashboard :5000, auto-spawns TensorBoard :6006 on logs/
python test_agent.py            # GUI rollout of the newest logs/*.zip checkpoint (falls back to ppo_maze_final)
python solve_maze.py            # GUI demo of the PPO model on random mazes
pytest tests/                   # single test: pytest tests/test_env.py::test_spaces
```

Run scripts from the repo root (they import `envs.maze_env` and `app.py` launches `train.py` via relative cwd).

## Architecture

- `envs/maze_env.py` — `MazeEnv`, the only environment, registered as `gym.make("MazeNav-v0", size=..., fixed=...)` in `envs/__init__.py`. Connects to PyBullet (`GUI` if `render_mode='human'`, else `DIRECT`). `reset()` rebuilds the whole simulation each episode: a DFS maze of odd `size` (+ `LOOPS` extra openings) from `layout_seed`, random start, hidden goal >= size//2 tiles away, 2 patrolling obstacles. Tile (r,c) -> world (c+offset, r+offset), `offset = -(size-1)//2`. `reset(seed, options={"size", "layout_seed"})` pins an episode (used by evaluation). Obs (70-dim) = `[time_left, goal_visible, dist, sin, cos` (goal terms are 0 unless in line of sight)`, 4 neighbour visit counts, 8 noisy lidar` (indices 0-16), `has_target, path_len/size, sin, cos` (route cue), `7x7 egocentric map crop]`. The map (`env.known`: -1 unknown / 0 open / 1 wall) is built by the agent from the noisy lidar, 4 extra world-axis scan rays and odometry, never from the true layout; `env.searched` tiles are ones the goal is ruled out from (visited, or on the clear stretch of the agent's row/column while the goal was not visible). The route cue is a BFS over the known map to the remembered goal, else to the nearest tile that has an unknown neighbour or is not yet searched (`env.waypoint()`). Actions `Discrete(3)`: forward / left / right. Reward is event-based (novelty, revisit, wall, obstacle, step penalties, +100 goal). Episode budget `max_steps = STEPS_PER_TILE * open tiles` depends on maze size only (never on the start-goal route, so `time_left` does not leak goal distance).
- `EVAL_SEED_RANGE = 2**20`: training layout seeds are drawn from `[2**20, 2**31)`, evaluation layout seeds from `[0, 2**20)`, so test mazes are never seen in training.
- `train.py` — SB3 PPO `MlpPolicy` on CPU, 8 `SubprocVecEnv` workers + `VecFrameStack(4)`, TensorBoard under `logs/PPO_run_N`. `Curriculum` callback grows the maze (`SIZES` 7->15) at 60% win rate and stores the level in `model.curriculum_level`, which SB3 pickles into every checkpoint zip.
- `evaluate.py` — fixed evaluation set (`episode_seed` = `base + size*10000 + i`), agents `RandomAgent`, `NodeCountAgent` (no map, odometry + lidar), `FrontierAgent` (follows the route cue over its own map; ~100% shows the memory suffices), `OracleAgent` (BFS on the true layout), `PPOAgent` (manual 4-frame stack identical to `VecFrameStack`). Reports success + Wilson 95% CI, steps, SPL, obstacle hits.
- `app.py` — Flask wrapper; `/start_training`, `/stop_training`, `/status` manage `train.py` as a subprocess; UI in `templates/index.html`.
- `solve_maze.py`, `test_agent.py` — GUI demos.

## Gotchas

- `MazeEnv` calls PyBullet without a `physicsClientId`, so only **one env per process** can be alive (tests and evaluation create/close them one at a time; training uses subprocesses).
- Checkpoints trained before the map-memory change have a 17-dim obs and no longer match the 70-dim env (`ppo_maze_old_final.zip`, `ppo_maze_parallel.zip`, existing `logs/*.zip`); retrain. `PROMOTE_AT` is now 0.9.
- Older note: checkpoints trained before the time-budget change saw a different episode budget. Zips from before `curriculum_level` was stored report an unknown level, and `--resume` from them restarts the curriculum at size 7.
- `test_agent.py` is stale (no frame stack, wrong obs print) and picks the checkpoint by `getctime` over `logs/*.zip`, which can differ from the one you just trained.
- `set_size` only sets `next_size`; `reset()` applies it, because the map arrays and `offset` are sized per episode (resizing mid-episode crashed training at the first level-up). Changing the maze size scheme means touching `offset` in `maze_env.py`; `evaluate.py` reads it from the env.
