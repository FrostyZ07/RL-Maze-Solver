# RL Maze Solver

A PPO agent that finds a hidden goal in randomly generated 3D mazes (PyBullet), using only noisy lidar and odometry. Stack: Gymnasium, Stable-Baselines3 (CPU), PyBullet, Flask, TensorBoard.

<!-- TODO: add a GIF/screenshot of the agent solving a maze (python solve_maze.py) -->

## Result

PPO trained for 4.7M steps (one seed, curriculum reached 15x15). Evaluation uses 200 held-out seeded mazes per size, with 95% Wilson confidence intervals.

| Agent | 7 | 9 | 11 | 13 | 15 | All |
|---|---|---|---|---|---|---|
| Random | 23.0% | 23.5% | 15.0% | 10.0% | 8.5% | 16.0% |
| NodeCount (no map, same sensors) | 99.5% | 98.0% | 92.0% | 92.5% | 84.0% | 93.2% |
| Frontier (own map, same sensors) | 100% | 99.5% | 100% | 100% | 100% | 99.9% |
| **PPO** | 99.0% | 98.5% | 94.5% | 87.5% | 87.0% | **93.3%** [91.6, 94.7] |
| Oracle (knows map and goal) | 100% | 100% | 100% | 100% | 100% | 100% |

Other PPO metrics (all sizes): 347 steps on successful runs, SPL 0.65, 7.3 obstacle hits per episode.

Takeaways:
- PPO beats Random by a wide margin, and ties the map-free NodeCount heuristic overall (93.3% vs 93.2%).
- PPO is better than NodeCount at 11x11 and 13x13 and on SPL and obstacle hits, but it degrades on larger mazes (87% at 15x15).
- The Frontier agent reaches ~100% with the same sensors, so the information in the observation is sufficient. The remaining gap is a learning gap, not an information gap.

Raw data: `results/eval_20261018_145222.csv` and `.json`.

## Setup

```
pip install -r requirements.txt
```
Python 3.12. On Windows, pybullet has no wheel, so install the MSVC C++ Build Tools first.

## Usage

```
python train.py                                   # 5M steps, 8 envs, checkpoints every 10k in logs/
python train.py --seed 1 --timesteps 2000000
python train.py --resume logs/ppo_maze_100000_steps.zip --timesteps 500000
# tuning: any PPO hyperparameter is a flag (also applied on --resume); --name keeps ppo_maze_final.zip safe
python train.py --resume logs/ppo_maze_4480000_steps.zip --timesteps 3000000 --name ppo_tuned     --lr-schedule linear --ent-coef 0.001 --gamma 0.998 --gae-lambda 0.97
python evaluate.py                                # 5 sizes x 200 held-out mazes, all agents
python evaluate.py --models ppo_maze_final.zip --sizes 17 21 --episodes 100 --out results/large   # unseen larger mazes
python app.py                                     # dashboard :5000, TensorBoard :6006
python solve_maze.py                              # GUI demo
pytest tests/
```

## Design

**Environment** (`envs/maze_env.py`, `gym.make("MazeNav-v0")`): a DFS maze of odd size plus 6 extra openings (loops), random start, hidden goal at least size//2 tiles away, and 2 patrolling obstacles. The maze is rebuilt every episode. Actions are discrete: forward, left, right.

**Observation (70-dim, 4 frames stacked):**
- time left, goal visibility, and goal distance/bearing (non-zero only in line of sight)
- visit counts of the 4 neighbouring tiles
- 8 noisy lidar rays
- route cue: has-target flag, path length/size, and bearing to the next tile
- a 7x7 egocentric crop of the agent's own map

The map is built only from lidar, 4 extra scan rays and odometry. It never uses the true layout. The route cue is a BFS over that map to the remembered goal, or else to the nearest unexplored or unsearched frontier.

**Reward (event-based, no distance shaping):** +100 goal, +0.1 first visit to a tile, -0.05 revisit, -0.05 wall, -1 obstacle, -0.01 per step. The step budget is 15 steps per open tile, so it depends on maze size only and does not leak goal distance.

**Training:** PPO MlpPolicy [256, 256], gamma 0.995, ent_coef 0.01, n_steps 512 x 8 envs, batch 256. A curriculum grows the maze 7 -> 9 -> 11 -> 13 -> 15 once the win rate over the last 100 episodes reaches 90%.

**Train/eval split:** training layout seeds come from [2^20, 2^31) and evaluation seeds from [0, 2^20), so test mazes are never seen in training.

## Limitations

- Single training seed, so no variance estimate across seeds.
- Success falls with maze size (87% at 15x15). Behaviour on 17 and 21 is untested (see the command above).
- The agent relies on a hand-built route cue and map crop. A hidden-goal agent without them was not tested (no ablation).
- Obstacle hits remain (about 7 per episode), though fewer than the heuristic baselines.
- Only one maze family (DFS plus loops) and a simulated, noisy lidar.
- Checkpoints from before the map-memory change (17-dim obs) are incompatible.
